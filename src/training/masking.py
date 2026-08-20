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
        n_graphs = getattr(data, 'num_graphs', 1)
        ei_full = data[('value_node', 'measured_by', 'sensor')].edge_index

        if n_graphs == 1:
            new_ei = self._apply_single(ei_full)
        else:
            n_s = self.gb.n_sensors
            n_v = self.gb.n_value_nodes
            parts = []
            for g in range(n_graphs):
                # for the current graph obtain the sensor and value node offsets and obtain the indices of the edges in the batch index belonging to the currrent graph
                s_off = g * n_s
                v_off = g * n_v
                in_b = (ei_full[1] >= s_off) & (ei_full[1] < s_off + n_s)

                # needed to subtract the offset from the indices, such that the logic in the masking methods doesn't break, e.g., value_indices_by_type and sensor_by_type
                # utilize indices from 0 to num_nodes, and not the incremented ones by batching from the DataLoader
                offset = torch.tensor([[v_off], [s_off]], dtype=torch.long, device=ei_full.device) 
                ei_local = ei_full[:, in_b] - offset

                # offset is added back to the masked edge_index
                parts.append(self._apply_single(ei_local) + offset)

            new_ei = torch.cat(parts, dim=1)

        data[('value_node', 'measured_by', 'sensor')].edge_index = new_ei
        if self.gb.bidirectional:
            data[('sensor', 'has_measure', 'value_node')].edge_index = new_ei.flip(0)
        return data

    def _apply_single(self, ei: torch.Tensor) -> torch.Tensor:
        """Mask a single graph's measured_by edge index (local indices)."""
        n_keep = ei.shape[1] - max(1, int(ei.shape[1] * self.mask_ratio))
        perm = torch.randperm(ei.shape[1], device=ei.device)
        keep_indices = perm[:n_keep]
        mask_indices = perm[n_keep:]

        if self.strategy == 'remove':
            return ei[:, keep_indices]
        elif self.strategy == 'all_bins':
            return self._all_bins(ei, keep_indices, mask_indices)
        elif self.strategy == 'random_bin':
            return self._random_bin(ei, keep_indices, mask_indices)

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
