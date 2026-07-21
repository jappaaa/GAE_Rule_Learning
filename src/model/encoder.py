import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super().__init__()

    def forward(self, x, edge_index):
        pass


class GATEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int, heads: int = 4):
        super().__init__()

    def forward(self, x, edge_index):
        pass
