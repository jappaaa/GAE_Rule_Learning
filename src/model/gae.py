import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.encoder import GCNEncoder, GATEncoder
from src.model.decoder import InnerProductDecoder
from src.utils.config import Config


class GraphAutoEncoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pass

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode reconstruction scores for given edge pairs."""
        pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pass

    def loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Binary cross-entropy over positive and negative edge samples."""
        pass
