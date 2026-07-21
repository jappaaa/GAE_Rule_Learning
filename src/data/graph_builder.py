import pandas as pd
import torch
from torch_geometric.data import Data


class GraphBuilder:
    """Builds one PyG Data object per timestamp.

    Node set (fixed across all timestamps):
      - KG nodes (sensors, junctions, pipes, ...)
      - Value nodes: one per (sensor × discrete bin) combination

    Edge set:
      - Static KG edges (topology, semantic relations) — same every timestamp
      - Dynamic has_measure edges — connect each sensor to its current bin node
    """

    def __init__(self, static_edge_index: torch.Tensor, node_ids: list, value_node_ids: list):
        self.static_edge_index = static_edge_index
        self.node_ids = node_ids              # KG nodes
        self.value_node_ids = value_node_ids  # (sensor, bin) value nodes

    def build(self, sensor_row: pd.Series) -> Data:
        """Build one Data object for a single timestamp row."""
        pass

    def build_all(self, sensor_df: pd.DataFrame) -> list[Data]:
        """Build one Data object per timestamp row."""
        pass

    def _make_has_measure_edges(self, sensor_row: pd.Series) -> torch.Tensor:
        """Return edge_index for has_measure edges at one timestamp."""
        pass
