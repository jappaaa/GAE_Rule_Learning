import pandas as pd
import torch
from torch_geometric.data import HeteroData


class GraphBuilder:
    """Builds PyG HeteroData objects from discretized sensor and attribute data.

    Node types: junction, reservoir, pipe, sensor, value_node
    Static edges: (junction/reservoir, connected, pipe) bidirectional
                  (junction/reservoir/pipe, has_sensor, sensor)
    Dynamic edges: (sensor, has_measure, value_node) — one per timestamp
    """

    SENSOR_TYPES = ['pressure', 'flow', 'demand']

    def __init__(self, topology: dict, n_bins_per_type: dict, bidirectional: bool = False):
        """
        Args:
            topology: output of LeakDBLoader.load_topology()
            n_bins_per_type: actual number of bins per sensor type after fitting,
                             e.g. {'pressure': 3, 'flow': 3, 'demand': 3}
            bidirectional: if True, has_measure edges are added in both directions
        """
        self.topology = topology
        self.n_bins_per_type = n_bins_per_type
        self.bidirectional = bidirectional

        self._build_node_indices()
        self._build_static_edges()
        self._build_static_node_features()

    def _build_node_indices(self):
        """Assign integer indices from 0 to n_nodes - 1 within each node type. The indices are stored inside dictionaries 
           linking each index to the original node id
        """

        # junction and reservoir indices derived from topology
        junction_ids = [nid for nid, attrs in self.topology['nodes'].items() if attrs['type'] == 'junction']
        reservoir_ids = [nid for nid, attrs in self.topology['nodes'].items() if attrs['type'] == 'reservoir']
        self.junction_idx = {nid: i for i, nid in enumerate(junction_ids)}
        self.reservoir_idx = {nid: i for i, nid in enumerate(reservoir_ids)}

        # pipe indices derived from topology
        self.pipe_idx = {pid: i for i, pid in enumerate(self.topology['pipes'].keys())}

        # sensor indices: pressure + demand at all nodes, flow at all pipes
        all_node_ids = list(self.junction_idx.keys()) + list(self.reservoir_idx.keys())
        self.sensor_idx = {}
        idx = 0
        for nid in all_node_ids:
            self.sensor_idx[('pressure', f'node_{nid}')] = idx
            idx += 1
        for nid in all_node_ids:
            self.sensor_idx[('demand', f'node_{nid}')] = idx
            idx += 1
        for pid in self.pipe_idx:
            self.sensor_idx[('flow', f'link_{pid}')] = idx
            idx += 1


        # value node indices: one per (sensor_type, bin_idx) key using actual n_bins per type (currently share per type)
        self.value_node_idx = {}
        offset = 0
        for sensor_type in self.SENSOR_TYPES:
            for bin_idx in range(self.n_bins_per_type[sensor_type]):
                self.value_node_idx[(sensor_type, bin_idx)] = offset + bin_idx
            offset += self.n_bins_per_type[sensor_type]

        self.n_sensors = len(self.sensor_idx)
        self.n_value_nodes = len(self.value_node_idx)

        self.sensor_indices_by_type = {
            st: torch.tensor([idx for (stype, _), idx in self.sensor_idx.items() if stype == st])
            for st in self.SENSOR_TYPES
        }
        self.value_indices_by_type = {
            st: torch.tensor([idx for (vtype, _), idx in self.value_node_idx.items() if vtype == st])
            for st in self.SENSOR_TYPES
        }

    def _build_static_edges(self):
        """Build edge indices for connected and has_sensor edges.

        Stores two dicts of pre-built edge_index tensors:
          self.connected_edges   — junction/reservoir <-> pipe
          self.has_sensor_edges  — junction/reservoir/pipe -> sensor
        """
        junc_pipe_src, junc_pipe_dst = [], []
        res_pipe_src,  res_pipe_dst  = [], []
        pipe_junc_src, pipe_junc_dst = [], []
        pipe_res_src,  pipe_res_dst  = [], []

        for pid, pipe_attrs in self.topology['pipes'].items():
            p = self.pipe_idx[pid]
            for endpoint in ('start_node', 'end_node'):
                nid = pipe_attrs[endpoint]
                if nid in self.junction_idx:
                    j = self.junction_idx[nid]
                    junc_pipe_src.append(j); junc_pipe_dst.append(p)
                    pipe_junc_src.append(p); pipe_junc_dst.append(j)
                else:
                    r = self.reservoir_idx[nid]
                    res_pipe_src.append(r); res_pipe_dst.append(p)
                    pipe_res_src.append(p); pipe_res_dst.append(r)

        def _ei(src, dst):
            return torch.tensor([src, dst], dtype=torch.long)

        self.connected_edges = {
            ('junction',   'connected', 'pipe'):       _ei(junc_pipe_src, junc_pipe_dst),
            ('pipe',       'connected', 'junction'):   _ei(pipe_junc_src, pipe_junc_dst),
            ('reservoir',  'connected', 'pipe'):       _ei(res_pipe_src,  res_pipe_dst),
            ('pipe',       'connected', 'reservoir'):  _ei(pipe_res_src,  pipe_res_dst),
        }

        # has_sensor: each junction/reservoir has one pressure + one demand sensor;
        # each pipe has one flow sensor
        junc_s_src, junc_s_dst = [], []
        res_s_src,  res_s_dst  = [], []
        pipe_s_src, pipe_s_dst = [], []

        for nid, j in self.junction_idx.items():
            for st in ('pressure', 'demand'):
                s = self.sensor_idx[(st, f'node_{nid}')]
                junc_s_src.append(j); junc_s_dst.append(s)

        for nid, r in self.reservoir_idx.items():
            for st in ('pressure', 'demand'):
                s = self.sensor_idx[(st, f'node_{nid}')]
                res_s_src.append(r); res_s_dst.append(s)

        for pid, p in self.pipe_idx.items():
            s = self.sensor_idx[('flow', f'link_{pid}')]
            pipe_s_src.append(p); pipe_s_dst.append(s)

        self.has_sensor_edges = {
            ('junction',  'has_sensor',     'sensor'):    _ei(junc_s_src, junc_s_dst),
            ('reservoir', 'has_sensor',     'sensor'):    _ei(res_s_src,  res_s_dst),
            ('pipe',      'has_sensor',     'sensor'):    _ei(pipe_s_src, pipe_s_dst),
            ('sensor',    'located_at', 'junction'):  _ei(junc_s_dst, junc_s_src),
            ('sensor',    'located_at', 'reservoir'): _ei(res_s_dst,  res_s_src),
            ('sensor',    'located_at', 'pipe'):      _ei(pipe_s_dst, pipe_s_src),
        }

    def _build_static_node_features(self):
        """Build fixed node feature tensors for sensor and value_node types.    

        sensor features   (n_sensors,     3): one-hot sensor type
        value_node feats  (n_value_nodes, 4): [normalized_bin_idx, one-hot sensor type]
        """
        type_to_idx = {st: i for i, st in enumerate(self.SENSOR_TYPES)}

        # sensor: one-hot type — shape (n_sensors, 3)
        sensor_x = torch.zeros(self.n_sensors, len(self.SENSOR_TYPES))
        for (st, _), s_i in self.sensor_idx.items():
            sensor_x[s_i, type_to_idx[st]] = 1.0
        self.sensor_x = sensor_x

        # value_node: [normalized bin idx, one-hot type] — shape (n_value_nodes, 4)
        value_x = torch.zeros(self.n_value_nodes, 1 + len(self.SENSOR_TYPES))
        for (st, bin_idx), v_i in self.value_node_idx.items():
            n = self.n_bins_per_type[st]
            value_x[v_i, 0] = bin_idx / max(n - 1, 1)
            value_x[v_i, 1 + type_to_idx[st]] = 1.0
        self.value_node_x = value_x

    def build_base(self, attributes: dict) -> HeteroData:
        """Build the static HeteroData for one scenario (node features + static edges).
        Called once per scenario; result is reused across all its timestamps."""
        data = HeteroData()

        # junction: [elevation, base_demand]
        junction_x = torch.zeros(len(self.junction_idx), 2)
        for nid, j in self.junction_idx.items():
            attrs = attributes['junctions'][nid]
            junction_x[j, 0] = attrs['elevation']
            junction_x[j, 1] = attrs['base_demand']
        data['junction'].x = junction_x

        # reservoir: [head]
        reservoir_x = torch.zeros(len(self.reservoir_idx), 1)
        for nid, r in self.reservoir_idx.items():
            reservoir_x[r, 0] = attributes['reservoirs'][nid]['head']
        data['reservoir'].x = reservoir_x

        # pipe: [length, diameter, roughness]
        pipe_x = torch.zeros(len(self.pipe_idx), 3)
        for pid, p in self.pipe_idx.items():
            attrs = attributes['pipes'][pid]
            pipe_x[p, 0] = attrs['length']
            pipe_x[p, 1] = attrs['diameter']
            pipe_x[p, 2] = attrs['roughness']
        data['pipe'].x = pipe_x

        # sensor and value_node features are fixed across all scenarios
        data['sensor'].x = self.sensor_x
        data['value_node'].x = self.value_node_x

        # static topology edges
        for edge_type, ei in self.connected_edges.items():
            data[edge_type].edge_index = ei
        for edge_type, ei in self.has_sensor_edges.items():
            data[edge_type].edge_index = ei

        return data

    def build(self, base: HeteroData, pressures_row: pd.Series, flows_row: pd.Series, demands_row: pd.Series) -> HeteroData:
        """Clone the scenario base graph and attach dynamic has_measure edges for one timestamp."""
        data = base.clone()
        ei = self._build_has_measure_edges(pressures_row, flows_row, demands_row)
        data[('sensor', 'has_measure', 'value_node')].edge_index = ei
        if self.bidirectional:
            data[('value_node', 'measured_by', 'sensor')].edge_index = ei.flip(0)
        return data

    def _build_has_measure_edges(self, pressures_row: pd.Series, flows_row: pd.Series, demands_row: pd.Series) -> torch.Tensor:
        """Return edge_index [2, n_edges] connecting each sensor node to its current value node.
        Sensors with NaN readings are skipped."""
        src, dst = [], []

        sensor_rows = [
            ('pressure', pressures_row),
            ('demand',   demands_row),
            ('flow',     flows_row),
        ]
        for st, row in sensor_rows:
            for col, bin_idx in row.items():
                if pd.isna(bin_idx):
                    continue
                s = self.sensor_idx[(st, col)]
                v = self.value_node_idx[(st, int(bin_idx))]
                src.append(s)
                dst.append(v)

        return torch.tensor([src, dst], dtype=torch.long)
