import torch
import torch.nn as nn


# PyG provides InnerProductDecoder but its forward() indexes a single z matrix via
# z[edge_index[0]] * z[edge_index[1]], assuming all nodes share one index space.
# For our heterogeneous case, sensor and value_node embeddings live in separate
# z_dict entries with independent indices, so we take z_src and z_dst directly.
class InnerProductDecoder(nn.Module):
    """Computes pairwise logit scores between two sets of node embeddings.

    Returns raw logits (no activation) — the caller applies softmax for rule
    extraction or passes directly to F.cross_entropy during training.
    Keeping the matrix multiplication here makes it easy to swap in alternative
    decoders (bilinear, MLP, etc.) for later experiments.
    """

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor, batched: bool = False) -> torch.Tensor:
        """Return logit matrix via inner product.

        Args:
            z_src:   [n_src, latent]          or [n_graphs, n_src, latent] if batched
            z_dst:   [n_dst, latent]          or [n_graphs, n_dst, latent] if batched
            batched: if True use bmm and flatten graphs into the first dim;
                     output is [n_graphs * n_src, n_dst]
        Returns:
            logits  [n_src, n_dst]  or  [n_graphs * n_src, n_dst]
        """
        if batched:
            return torch.bmm(z_src, z_dst.transpose(1, 2)).flatten(0, 1)
        return z_src @ z_dst.T
