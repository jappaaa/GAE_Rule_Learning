import torch
from torch_geometric.data import HeteroData

from src.data.graph_builder import GraphBuilder
from src.utils.config import Config

SUPPORTED_STRATEGIES = {'remove', 'all_bins', 'random_bin'}


class Masker:
    """Applies edge masking to measured_by edges during training.

    Strategies:
      remove     — drop masked edges entirely; sensor receives no reading signal
      all_bins   — replace masked edge with edges to all bins of the sensor's type;
                   sensor is maximally uncertain about its reading
      random_bin — replace masked edge with an edge to a randomly chosen bin;
                   sensor receives a corrupted (likely wrong) reading
    """

    def __init__(self, graph_builder: GraphBuilder, config: Config):
        if config.masking_strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown masking_strategy '{config.masking_strategy}'. "
                f"Supported: {SUPPORTED_STRATEGIES}"
            )
        self.gb = graph_builder
        self.mask_ratio = config.mask_ratio
        self.strategy = config.masking_strategy

        self._sensor_to_type = {
            idx: st
            for st in graph_builder.SENSOR_TYPES
            for idx in graph_builder.sensor_indices_by_type[st].tolist()
        }

    def apply(self, data: HeteroData) -> HeteroData:
        ei = data[('value_node', 'measured_by', 'sensor')].edge_index
        n_keep = ei.shape[1] - max(1, int(ei.shape[1] * self.mask_ratio))
        perm = torch.randperm(ei.shape[1], device=ei.device)
        keep_sensor_indices = perm[:n_keep]
        mask_sensor_indices = perm[n_keep:]

        if self.strategy == 'remove':
            new_ei = ei[:, keep_sensor_indices]
        elif self.strategy == 'all_bins':
            new_ei = self._all_bins(ei, keep_sensor_indices, mask_sensor_indices)
        elif self.strategy == 'random_bin':
            new_ei = self._random_bin(ei, keep_sensor_indices, mask_sensor_indices)

        data[('value_node', 'measured_by', 'sensor')].edge_index = new_ei
        if self.gb.bidirectional:
            data[('sensor', 'has_measure', 'value_node')].edge_index = new_ei.flip(0)
        return data

    def _all_bins(self, ei, keep_sensor_indices, mask_sensor_indices):
        """Replace each masked edge with edges to all bins of the sensor's type."""
        src, dst = [], []
        for s_idx in ei[1, mask_sensor_indices].tolist():
            st = self._sensor_to_type[s_idx]
            for v_idx in self.gb.value_indices_by_type[st].tolist():
                src.append(v_idx)
                dst.append(s_idx)
        masked = torch.tensor([src, dst], dtype=torch.long, device=ei.device)
        return torch.cat([ei[:, keep_sensor_indices], masked], dim=1)

    def _random_bin(self, ei, keep_sensor_indices, mask_sensor_indices):
        """Replace each masked edge with an edge to a randomly chosen bin of the sensor's type. Can be seen as adding noise"""
        src, dst = [], []
        for s_idx in ei[1, mask_sensor_indices].tolist():
            st = self._sensor_to_type[s_idx]
            value_nodes = self.gb.value_indices_by_type[st]
            random_v = value_nodes[torch.randint(len(value_nodes), (1,)).item()].item()
            src.append(random_v)
            dst.append(s_idx)
        masked = torch.tensor([src, dst], dtype=torch.long, device=ei.device)
        return torch.cat([ei[:, keep_sensor_indices], masked], dim=1)
