import pandas as pd
import torch
from torch_geometric.data import HeteroData


class GraphBuilder:
    """Builds PyG HeteroData objects from discretized sensor and attribute data.

    Node types: junction, reservoir, pipe, sensor, value_node
    Static edges: (junction/reservoir, connected, pipe) bidirectional
                  (junction/reservoir/pipe, has_sensor, sensor)
    Dynamic edges: (value_node, measured_by, sensor) — one per timestamp;
                   optionally also (sensor, has_measure, value_node) if bidirectional=True
    """

    SENSOR_TYPES = ['pressure', 'flow', 'demand']

    def __init__(self, topology: dict, n_bins_per_type: dict, bidirectional: bool = False, virtual_node_mode: str = 'none'):
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
        self.virtual_node_mode = virtual_node_mode

        self._build_node_indices()
        self._build_static_edges()
        self._build_virtual_node_edges()
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

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            self.type_virtual_idx = {st: i for i, st in enumerate(self.SENSOR_TYPES)}
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            self.global_virtual_idx = {0: 0}

    @staticmethod
    def _ei(src: list, dst: list) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

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

        self.connected_edges = {
            ('junction',   'connected', 'pipe'):       self._ei(junc_pipe_src, junc_pipe_dst),
            ('pipe',       'connected', 'junction'):   self._ei(pipe_junc_src, pipe_junc_dst),
            ('reservoir',  'connected', 'pipe'):       self._ei(res_pipe_src,  res_pipe_dst),
            ('pipe',       'connected', 'reservoir'):  self._ei(pipe_res_src,  pipe_res_dst),
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
            ('junction',  'has_sensor',  'sensor'):    self._ei(junc_s_src, junc_s_dst),
            ('reservoir', 'has_sensor',  'sensor'):    self._ei(res_s_src,  res_s_dst),
            ('pipe',      'has_sensor',  'sensor'):    self._ei(pipe_s_src, pipe_s_dst),
            ('sensor',    'located_at',  'junction'):  self._ei(junc_s_dst, junc_s_src),
            ('sensor',    'located_at',  'reservoir'): self._ei(res_s_dst,  res_s_src),
            ('sensor',    'located_at',  'pipe'):      self._ei(pipe_s_dst, pipe_s_src),
        }

    def _build_virtual_node_edges(self):
        """Build edge indices connecting sensors to virtual nodes based on virtual_node_mode.

        'global':              sensor <-> global_virtual (all sensors)
        'type_interconnected': sensor <-> type_virtual (by type) + type_virtual <-> type_virtual (all pairs)
        'hierarchical':        sensor <-> type_virtual (by type) + type_virtual <-> global_virtual
        'hierarchical_direct': sensor <-> type_virtual (by type) + type_virtual -> global_virtual + global_virtual -> sensor
        """
        self.virtual_node_edges = {}
        if self.virtual_node_mode == 'none':
            return

        if self.virtual_node_mode == 'global':
            src = list(range(self.n_sensors))
            dst = [0] * self.n_sensors
            self.virtual_node_edges = {
                ('sensor',         'to_global_state',   'global_virtual'): self._ei(src, dst),
                ('global_virtual', 'from_global_state', 'sensor'):         self._ei(dst, src),
            }

        elif self.virtual_node_mode in ('type_interconnected', 'hierarchical', 'hierarchical_direct'):
            s_src, s_dst = [], []
            for (st, _), s_idx in self.sensor_idx.items():
                tv_idx = self.type_virtual_idx[st]
                s_src.append(s_idx)
                s_dst.append(tv_idx)
            self.virtual_node_edges = {
                ('sensor',       'to_state',   'type_virtual'): self._ei(s_src, s_dst),
                ('type_virtual', 'from_state', 'sensor'):       self._ei(s_dst, s_src),
            }

            if self.virtual_node_mode == 'type_interconnected':
                n = len(self.SENSOR_TYPES)
                tv_src = [i for i in range(n) for j in range(n) if i != j]
                tv_dst = [j for i in range(n) for j in range(n) if i != j]
                self.virtual_node_edges[('type_virtual', 'connected_to', 'type_virtual')] = self._ei(tv_src, tv_dst)

            elif self.virtual_node_mode == 'hierarchical':
                n = len(self.SENSOR_TYPES)
                tv_src = list(range(n))
                gv_dst = [0] * n
                self.virtual_node_edges[('type_virtual',   'to_global_state',   'global_virtual')] = self._ei(tv_src, gv_dst)
                self.virtual_node_edges[('global_virtual', 'from_global_state', 'type_virtual')]   = self._ei(gv_dst, tv_src)

            elif self.virtual_node_mode == 'hierarchical_direct':
                n = len(self.SENSOR_TYPES)
                tv_src = list(range(n))
                gv_dst = [0] * n
                self.virtual_node_edges[('type_virtual',   'to_global_state',   'global_virtual')] = self._ei(tv_src, gv_dst)
                # global distributes directly to sensors, bypassing type_virtual on the way down
                self.virtual_node_edges[('global_virtual', 'from_global_state', 'sensor')]         = self._ei([0] * self.n_sensors, list(range(self.n_sensors)))

    def _build_static_node_features(self):
        """Build fixed node feature tensors for sensor and value_node types.

        sensor features        (n_sensors,          n_sensor_types): one-hot sensor type
        value_node feats       (n_value_nodes,      n_sensor_types+1): [normalized_bin_idx, one-hot sensor type]
        type_virtual feats     (n_sensor_types,     n_sensor_types): one-hot sensor type (same encoding as sensor)
        global_virtual feats   (1,                  1): zero initialization (identity derived from aggregation)
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

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            type_virtual_x = torch.zeros(len(self.SENSOR_TYPES), len(self.SENSOR_TYPES))
            for st, tv_i in self.type_virtual_idx.items():
                type_virtual_x[tv_i, type_to_idx[st]] = 1.0
            self.type_virtual_x = type_virtual_x

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            self.global_virtual_x = torch.zeros(1, 1)

    def build_sensor_hop_map(self, max_hops: int) -> dict:
        """Return {sensor_idx: frozenset(reachable_sensor_indices)} via BFS on static topology.

        Uses connected + has_sensor/located_at edges only (no measured_by/has_measure),
        so the result is independent of any particular timestamp's dynamic edges.
        Pass max_hops = num_layers - 1 to account for the one hop from antecedent
        value_node to antecedent sensor before traversing topology.
        """
        # Determine the global offsets for each node type
        n_j = len(self.junction_idx)
        n_r = len(self.reservoir_idx)
        n_p = len(self.pipe_idx)
        off_r = n_j
        off_p = n_j + n_r
        off_s = n_j + n_r + n_p
        total = off_s + self.n_sensors

        # set the offsets per node type for easy access
        type_off = {'junction': 0, 'reservoir': off_r, 'pipe': off_p, 'sensor': off_s}

        # from the typespecific edge indexes, create a global one 
        adj = [[] for _ in range(total)]
        for (src_type, _, dst_type), ei in {**self.connected_edges, **self.has_sensor_edges}.items():
            o_src = type_off[src_type]
            o_dst = type_off[dst_type]
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                adj[o_src + s].append(o_dst + d)

        # use a breadth first search approach to traverse the graph up to max_hops for each sensor node
        # to obtain the reachable sensors.
        sensor_hop_map = {}
        for s_idx in range(self.n_sensors):
            start = off_s + s_idx
            dist = {start: 0}
            queue = [start]
            reachable = set() # start without the node itself (these are the antecedents and should not be considered)
            qi = 0
            while qi < len(queue):
                node = queue[qi]; qi += 1
                d = dist[node]
                if d >= max_hops:
                    continue
                for nb in adj[node]:
                    if nb not in dist:
                        dist[nb] = d + 1
                        queue.append(nb)
                        if nb >= off_s:
                            reachable.add(nb - off_s)
            sensor_hop_map[s_idx] = frozenset(reachable)

        return sensor_hop_map

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

        # virtual node features and edges (mode-dependent)
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            data['type_virtual'].x = self.type_virtual_x
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            data['global_virtual'].x = self.global_virtual_x
        for edge_type, ei in self.virtual_node_edges.items():
            data[edge_type].edge_index = ei

        return data

    def build(self, base: HeteroData, pressures_row: pd.Series, flows_row: pd.Series, demands_row: pd.Series) -> HeteroData:
        """Clone the scenario base graph and attach dynamic measured_by edges for one timestamp."""
        data = base.clone()
        ei = self._build_measured_by_edges(pressures_row, flows_row, demands_row)
        data[('value_node', 'measured_by', 'sensor')].edge_index = ei
        if self.bidirectional:
            data[('sensor', 'has_measure', 'value_node')].edge_index = ei.flip(0)
        return data

    def _build_measured_by_edges(self, pressures_row: pd.Series, flows_row: pd.Series, demands_row: pd.Series) -> torch.Tensor:
        """Return edge_index [2, n_edges] connecting each value node to its sensor (value_node → sensor).
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
                src.append(v)
                dst.append(s)

        return self._ei(src, dst)
