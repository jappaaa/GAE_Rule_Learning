import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class GNNEncoder(nn.Module):
    """Homogeneous GNN backbone to be converted to heterogeneous via to_hetero() in the GAE."""

    def __init__(self, hidden_channels: int, latent_channels: int,
                 num_layers: int = 2, conv_type: str = 'sage'):
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            out = latent_channels if i == num_layers - 1 else hidden_channels
            if conv_type == 'gat':
                self.convs.append(GATConv((-1, -1), out, add_self_loops=False))
            else:
                self.convs.append(SAGEConv((-1, -1), out))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x
