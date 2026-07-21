import torch
import torch.nn as nn


class InnerProductDecoder(nn.Module):
    """Reconstructs adjacency matrix via inner product of node embeddings."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed adjacency (sigmoid of Z @ Z.T)."""
        pass

    def forward_pair(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Return reconstruction scores for a specific set of (src, dst) pairs.
        More memory-efficient than computing the full N×N matrix."""
        pass
