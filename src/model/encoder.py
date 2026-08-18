import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv


class GNNEncoder(nn.Module):
    """Homogeneous GNN backbone to be converted to heterogeneous via to_hetero() in the GAE."""

    def __init__(self, hidden_channels: int, latent_channels: int,
                 num_layers: int = 2, conv_type: str = 'sage'):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out = latent_channels if i == num_layers - 1 else hidden_channels
            if conv_type == 'gat':
                self.convs.append(GATConv((-1, -1), out, add_self_loops=False)) # can't add self loops to bipartite relations
            else:
                self.convs.append(SAGEConv((-1, -1), out)) # SAGE already concatenates central nodes embedding, therefore no self-loops

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1: # Don't add activation at the last layer
                x = F.relu(x)
        return x


class HeteroGNNEncoder(nn.Module):
    """Natively heterogeneous GNN encoder using HeteroConv.

    Used when bidirectional_has_measure=False: value_node only appears as a source
    in measured_by edges and never as a destination, so to_hetero() cannot be used.
    All edge types in the metadata participate in message passing; value_node is
    projected to latent space via a dedicated linear layer after all conv layers.
    """

    def __init__(self, hidden_channels: int, latent_channels: int,
                 num_layers: int, conv_type: str, metadata, aggr: str = 'sum'):
        super().__init__()
        node_types, edge_types = metadata

        # One lazy linear per node type: projects variable raw feature dims → hidden_channels
        self.input_lins = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_channels) for nt in node_types
        })

        # HeteroConv layers — one conv per edge type, covering all edges in metadata
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out = latent_channels if i == num_layers - 1 else hidden_channels
            conv_dict = {}
            for edge_type in edge_types:
                if conv_type == 'gat':
                    conv_dict[edge_type] = GATConv((-1, -1), out, add_self_loops=False)
                else:
                    conv_dict[edge_type] = SAGEConv((-1, -1), out)
            self.convs.append(HeteroConv(conv_dict, aggr=aggr))

        # value_node is never updated by message passing; project to latent_channels once
        self.value_node_out = nn.Linear(hidden_channels, latent_channels)

    def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
        # Project all node types (including value_node) to hidden_channels
        x_dict = {nt: F.relu(self.input_lins[nt](x)) for nt, x in x_dict.items()}
        value_node_h = x_dict['value_node']

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {nt: F.relu(x) for nt, x in x_dict.items()}
                # HeteroConv drops value_node (never a destination); reinsert as source for next layer
                x_dict['value_node'] = value_node_h

        # Final output: project value_node's fixed hidden representation to latent_channels
        x_dict['value_node'] = self.value_node_out(value_node_h)
        return x_dict
