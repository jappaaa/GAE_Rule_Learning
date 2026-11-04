import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == 'batch':
        return nn.BatchNorm1d(channels)
    if norm_type == 'layer':
        return nn.LayerNorm(channels)
    return nn.Identity()


class GNNEncoder(nn.Module):
    """Homogeneous GNN backbone to be converted to heterogeneous via to_hetero() in the GAE."""

    def __init__(self, hidden_channels: int, latent_channels: int,
                 num_layers: int = 2, conv_type: str = 'sage', norm_type: str = 'none'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            out = latent_channels if i == num_layers - 1 else hidden_channels
            if conv_type == 'gat':
                self.convs.append(GATConv((-1, -1), out, add_self_loops=False)) # can't add self loops to bipartite relations
            else:
                self.convs.append(SAGEConv((-1, -1), out)) # SAGE already concatenates central nodes embedding, therefore no self-loops
            if i < num_layers - 1:
                self.norms.append(_make_norm(norm_type, hidden_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1: # Don't add activation and normalization at the last layer
                x = F.relu(self.norms[i](x))
        return x


class HeteroGNNEncoder(nn.Module):
    """Natively heterogeneous GNN encoder using HeteroConv.

    Used when bidirectional_has_measure=False: value_node only appears as a source
    in measured_by edges and never as a destination, so to_hetero() cannot be used.
    All edge types in the metadata participate in message passing; value_node is
    projected to latent space via a dedicated linear layer after all conv layers.
    """

    # destination node types that are virtual — sensor→virtual edges use GAT when gat_for_sensor_virtual=True
    _VIRTUAL_DST_TYPES = {'type_virtual', 'global_virtual'}

    def __init__(self, hidden_channels: int, latent_channels: int,
                 num_layers: int, conv_type: str, metadata, aggr: str = 'sum',
                 norm_type: str = 'none', gat_for_sensor_virtual: bool = False,
                 use_input_projection: bool = True, value_node_projection_only: bool = False):
        super().__init__()
        node_types, edge_types = metadata
        self.use_input_projection = use_input_projection
        self.value_node_projection_only = value_node_projection_only

        # Input projection: projects raw node features → hidden_channels before message passing.
        # When use_input_projection=False: no projection, conv layers handle raw features via lazy (-1,-1) init.
        # When value_node_projection_only=True: only value_node is projected; other types stay raw for conv.
        if use_input_projection:
            proj_types = ['value_node'] if value_node_projection_only else node_types
            self.input_lins = nn.ModuleDict({
                nt: nn.LazyLinear(hidden_channels) for nt in proj_types
            })

        # HeteroConv layers — one conv per edge type, covering all edges in metadata
        self.convs = nn.ModuleList()
        # One norm per node type (excluding value_node) per intermediate layer
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            out = latent_channels if i == num_layers - 1 else hidden_channels
            conv_dict = {}
            for edge_type in edge_types:
                src_type, _, dst_type = edge_type
                # use GAT for sensor→virtual edges when requested, to learn per-sensor importance weights
                if gat_for_sensor_virtual and src_type == 'sensor' and dst_type in self._VIRTUAL_DST_TYPES:
                    conv_dict[edge_type] = GATConv((-1, -1), out, add_self_loops=False)
                elif conv_type == 'gat':
                    conv_dict[edge_type] = GATConv((-1, -1), out, add_self_loops=False)
                else:
                    conv_dict[edge_type] = SAGEConv((-1, -1), out)
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))
            if i < num_layers - 1:
                # value_node is reinserted manually and never updated by conv, so exclude it
                self.norms.append(nn.ModuleDict({
                    nt: _make_norm(norm_type, hidden_channels)
                    for nt in node_types if nt != 'value_node'
                }))

        # value_node is never updated by message passing; project its fixed representation to latent_channels.
        # When use_input_projection=True the input is always hidden_channels, so a regular Linear
        # is used — this keeps the architecture identical to the pre-flag state and old checkpoints loadable.
        # When False, input dim is unknown (raw features), so LazyLinear is used instead.
        if use_input_projection:
            self.value_node_out = nn.Linear(hidden_channels, latent_channels)
        else:
            self.value_node_out = nn.LazyLinear(latent_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        if self.use_input_projection:
            if self.value_node_projection_only:
                # only project value_node; other types pass raw features into conv via lazy init
                x_dict['value_node'] = F.relu(self.input_lins['value_node'](x_dict['value_node']))
            else:
                x_dict = {nt: F.relu(self.input_lins[nt](x)) for nt, x in x_dict.items()}
        value_node_h = x_dict['value_node']  # stored here; never updated by conv layers

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {nt: F.relu(self.norms[i][nt](x)) for nt, x in x_dict.items()}
                # HeteroConv drops value_node (never a destination); reinsert as source for next layer
                x_dict['value_node'] = value_node_h

        # Final output: project value_node's fixed hidden representation to latent_channels
        x_dict['value_node'] = self.value_node_out(value_node_h)
        return x_dict
