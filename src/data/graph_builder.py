from collections import defaultdict

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

    def __init__(self, topology: dict, n_bins_per_type: dict, bidirectional: bool = False,
                 virtual_node_mode: str = 'none', onehot_bin_features: bool = True,
                 sensor_types: tuple = ('pressure', 'flow', 'demand'),
                 node_sensor_types: tuple = ('pressure', 'demand')):
        """
        Args:
            topology: output of a loader's load_topology() — {"nodes": {id: {"type": ...}},
                      "links": {id: {"type": ..., "start_node": ..., "end_node": ...}}}
            n_bins_per_type: actual number of bins per sensor type after fitting,
                             e.g. {'pressure': 3, 'flow': 3, 'demand': 3}
            bidirectional: if True, has_measure edges are added in both directions
            onehot_bin_features: if True, value_node bin feature is one-hot; if False, normalized scalar
            sensor_types: ordered list of signal types to build sensors for (was a hardcoded
                          class attribute SENSOR_TYPES); defaults to LeakDB's 3 signals
            node_sensor_types: subset of sensor_types measured once per node, across every
                              node type present in topology (e.g. pressure/demand); every other
                              sensor_types entry is assumed measured once per link instead,
                              across every link type (e.g. flow) — matches both LeakDB and
                              DiTEC-WDN, where sensor signals are organized by node/link
                              category rather than by component sub-type (unlike static
                              attributes, which do differ per sub-type)
        """
        self.topology = topology
        self.n_bins_per_type = n_bins_per_type
        self.bidirectional = bidirectional
        self.virtual_node_mode = virtual_node_mode
        self.onehot_bin_features = onehot_bin_features
        self.SENSOR_TYPES = list(sensor_types)
        self.node_sensor_types = set(node_sensor_types)

        self._build_node_indices()
        self._build_static_edges()
        self._build_virtual_node_edges()
        self._build_static_node_features()

    def _build_node_indices(self):
        """Assign integer indices from 0 to n_nodes - 1 within each node type, and within each
        link type, derived generically from whatever types are present in `topology` (not
        hardcoded to junction/reservoir/pipe, so this also works for DiTEC-WDN's additional
        tank/pump/prv/... types). The indices are stored inside dictionaries linking each
        index to the original node/link id.
        """

        # node/link type indices derived from topology
        node_types = sorted({info['type'] for info in self.topology['nodes'].values()})
        link_types = sorted({info['type'] for info in self.topology['links'].values()})

        self.node_idx = {}
        for nt in node_types:
            ids = [nid for nid, info in self.topology['nodes'].items() if info['type'] == nt]
            self.node_idx[nt] = {nid: i for i, nid in enumerate(ids)}

        self.link_idx = {}
        for lt in link_types:
            ids = [lid for lid, info in self.topology['links'].items() if info['type'] == lt]
            self.link_idx[lt] = {lid: i for i, lid in enumerate(ids)}

        # backward-compat aliases for LeakDB's two node types / one link type -- used by the
        # rest of this class (generalized incrementally, see docs/experiment_plan.md) and by
        # src/utils/visualize.py
        self.junction_idx = self.node_idx.get('junction', {})
        self.reservoir_idx = self.node_idx.get('reservoir', {})
        self.pipe_idx = self.link_idx.get('pipe', {})

        # sensor indices: self.SENSOR_TYPES in order, each either over every node (if listed in
        # node_sensor_types) or every link (otherwise) -- preserves LeakDB's original insertion
        # order (pressure, flow, demand) exactly, since flow sits between the two node sensors.
        # Keys are (signal_type, raw_id) -- no "node_"/"link_" tag, since the signal type
        # already disambiguates a node-measured id from a link-measured one (loaders are
        # responsible for giving sensor dataframe columns the same raw id as topology).
        all_node_ids = [nid for ids in self.node_idx.values() for nid in ids]
        all_link_ids = [lid for ids in self.link_idx.values() for lid in ids]

        self.sensor_idx = {}
        idx = 0
        for st in self.SENSOR_TYPES:
            if st in self.node_sensor_types:
                for nid in all_node_ids:
                    self.sensor_idx[(st, nid)] = idx
                    idx += 1
            else:
                for lid in all_link_ids:
                    self.sensor_idx[(st, lid)] = idx
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
        """Build edge indices for connected and has_sensor edges, generalized over whatever
        node/link types are present in topology (not hardcoded to junction/reservoir/pipe).

        Stores two dicts of pre-built edge_index tensors:
          self.connected_edges   — (node_type, connected, link_type) bidirectional, one pair
                                    per (node_type, link_type) combination that actually occurs
          self.has_sensor_edges  — (component_type, has_sensor, sensor) + reverse located_at,
                                    one pair per component type that has a sensor of some kind
                                    (following the same node_sensor_types/link split as
                                    _build_node_indices)
        """
        # node id -> its node type, to classify each link endpoint below
        node_type_of = {nid: nt for nt, ids in self.node_idx.items() for nid in ids}

        connected_pairs = defaultdict(lambda: ([], []))  # (node_type, link_type) -> (node_side, link_side)
        for lt, ids in self.link_idx.items():
            for lid, l in ids.items():
                endpoints = self.topology['links'][lid]
                for endpoint in ('start_node', 'end_node'):
                    nid = endpoints[endpoint]
                    nt = node_type_of[nid]
                    n = self.node_idx[nt][nid]
                    n_side, l_side = connected_pairs[(nt, lt)]
                    n_side.append(n); l_side.append(l)

        self.connected_edges = {}
        for (nt, lt), (n_side, l_side) in connected_pairs.items():
            self.connected_edges[(nt, 'connected', lt)] = self._ei(n_side, l_side)
            self.connected_edges[(lt, 'connected', nt)] = self._ei(l_side, n_side)

        # has_sensor: one (component_type, sensor) pair per SENSOR_TYPES entry that applies to
        # that component type, following the node_sensor_types/link split from _build_node_indices
        sensor_pairs = defaultdict(lambda: ([], []))  # component_type -> (comp_side, sensor_side)
        for st in self.SENSOR_TYPES:
            if st in self.node_sensor_types:
                for nt, ids in self.node_idx.items():
                    for nid, n in ids.items():
                        s = self.sensor_idx[(st, nid)]
                        c_side, s_side = sensor_pairs[nt]
                        c_side.append(n); s_side.append(s)
            else:
                for lt, ids in self.link_idx.items():
                    for lid, l in ids.items():
                        s = self.sensor_idx[(st, lid)]
                        c_side, s_side = sensor_pairs[lt]
                        c_side.append(l); s_side.append(s)

        self.has_sensor_edges = {}
        for comp_type, (c_side, s_side) in sensor_pairs.items():
            self.has_sensor_edges[(comp_type, 'has_sensor', 'sensor')] = self._ei(c_side, s_side)
            self.has_sensor_edges[('sensor', 'located_at', comp_type)] = self._ei(s_side, c_side)

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
                tv_src, tv_dst = [], []
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            tv_src.append(i)
                            tv_dst.append(j)
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
        value_node feats       (n_value_nodes,      max_bins+n_sensor_types) or (n_value_nodes, 1+n_sensor_types): [one-hot or scalar bin_idx, one-hot sensor type]
        type_virtual feats     (n_sensor_types,     n_sensor_types): one-hot sensor type (same encoding as sensor)
        global_virtual feats   (1,                  1): zero initialization (identity derived from aggregation)
        """
        type_to_idx = {st: i for i, st in enumerate(self.SENSOR_TYPES)}

        # sensor: one-hot type — shape (n_sensors, 3)
        sensor_x = torch.zeros(self.n_sensors, len(self.SENSOR_TYPES))
        for (st, _), s_i in self.sensor_idx.items():
            sensor_x[s_i, type_to_idx[st]] = 1.0
        self.sensor_x = sensor_x

        # value_node features — two modes controlled by onehot_bin_features:
        #   True:  one-hot bin  + one-hot type — shape (n_value_nodes, max_bins + n_sensor_types)
        #   False: scalar bin   + one-hot type — shape (n_value_nodes, 1 + n_sensor_types)
        if self.onehot_bin_features:
            max_bins = max(self.n_bins_per_type.values())
            value_x = torch.zeros(self.n_value_nodes, max_bins + len(self.SENSOR_TYPES))
            for (st, bin_idx), v_i in self.value_node_idx.items():
                value_x[v_i, bin_idx] = 1.0                          # one-hot bin position
                value_x[v_i, max_bins + type_to_idx[st]] = 1.0      # one-hot sensor type
        else:
            value_x = torch.zeros(self.n_value_nodes, 1 + len(self.SENSOR_TYPES))
            for (st, bin_idx), v_i in self.value_node_idx.items():
                n = self.n_bins_per_type[st]
                value_x[v_i, 0] = bin_idx / max(n - 1, 1)           # min-max normalized scalar
                value_x[v_i, 1 + type_to_idx[st]] = 1.0             # one-hot sensor type
        self.value_node_x = value_x

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            type_virtual_x = torch.zeros(len(self.SENSOR_TYPES), len(self.SENSOR_TYPES))
            for st, tv_i in self.type_virtual_idx.items():
                type_virtual_x[tv_i, type_to_idx[st]] = 1.0
            self.type_virtual_x = type_virtual_x

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            self.global_virtual_x = torch.zeros(1, 1)

    def _component_offsets(self) -> tuple:
        """Return ({component_type: global_offset}, total_count), spanning every node type
        then every link type present in topology (not hardcoded to junction/reservoir/pipe).
        Shared by build_sensor_distance_map/build_sensor_hop_map, which flatten all component
        types into one adjacency array for BFS.
        """
        type_off = {}
        offset = 0
        for type_idx in (self.node_idx, self.link_idx):
            for ct, ids in type_idx.items():
                type_off[ct] = offset
                offset += len(ids)
        return type_off, offset

    def build_sensor_distance_map(self) -> dict:
        """Return {sensor_idx: {other_sensor_idx: network_hop_distance}}.

        Distance is the shortest path between sensor *location nodes* (junctions/pipes)
        in the water network graph, traversing only connected edges:
          0 = co-located sensors (same junction or pipe)
          1 = directly connected locations (junction ↔ adjacent pipe)
          2 = two steps apart (junction → pipe → junction)
        This is more interpretable than raw graph hops because it counts structural
        water network components rather than PyG graph edges.
        Note: build_sensor_hop_map uses raw graph hops — kept separate because it
        serves the GNN receptive field filter where graph layers are the natural unit.
        """
        type_off, n_network_nodes = self._component_offsets()

        # adjacency on network nodes only — no sensor nodes, no measured_by edges
        adj = [[] for _ in range(n_network_nodes)]
        for (src_type, _, dst_type), ei in self.connected_edges.items():
            if src_type not in type_off or dst_type not in type_off:
                continue
            o_src, o_dst = type_off[src_type], type_off[dst_type]
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                adj[o_src + s].append(o_dst + d)

        # map each sensor to its location node (junction or pipe)
        sensor_location = {}  # sensor_idx -> global network node index
        for (src_type, _, dst_type), ei in self.has_sensor_edges.items():
            if dst_type != 'sensor' or src_type not in type_off:
                continue
            o_src = type_off[src_type]
            for loc, s in zip(ei[0].tolist(), ei[1].tolist()):
                sensor_location[s] = o_src + loc

        # reverse: network node -> list of sensors at that location
        loc_to_sensors = defaultdict(list)
        for s_idx, loc in sensor_location.items():
            loc_to_sensors[loc].append(s_idx)

        # BFS from each sensor's location through the water network
        distance_map = {}
        for s_idx in range(self.n_sensors):
            if s_idx not in sensor_location:
                distance_map[s_idx] = {}
                continue
            start = sensor_location[s_idx]
            dist = {start: 0}
            queue = [start]
            qi = 0
            while qi < len(queue):
                node = queue[qi]; qi += 1
                for nb in adj[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            distance_map[s_idx] = {
                other_s: d
                for loc, d in dist.items()
                for other_s in loc_to_sensors.get(loc, [])
                if other_s != s_idx
            }
        return distance_map

    def build_sensor_hop_map(self, max_hops: int) -> dict:
        """Return {sensor_idx: frozenset(reachable_sensor_indices)} via BFS on static topology.

        Uses connected + has_sensor/located_at edges only (no measured_by/has_measure),
        so the result is independent of any particular timestamp's dynamic edges.
        Pass max_hops = num_layers - 1 to account for the one hop from antecedent
        value_node to antecedent sensor before traversing topology.
        """
        # Determine the global offsets for each component type, plus sensors at the end
        type_off, off_s = self._component_offsets()
        type_off = {**type_off, 'sensor': off_s}
        total = off_s + self.n_sensors

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

    def build_base(self, attrs_dfs: dict) -> HeteroData:
        """Build the static HeteroData for one scenario (node/link features + static edges).
        Called once per scenario; result is reused across all its timestamps.

        Args:
            attrs_dfs: {component_type: df}, one DataFrame per node/link type present in
                       topology, indexed by component id, one column per static attribute
                       (already normalized upstream) — same shape
                       DiTECWDNLoader.load_attributes_as_df() returns for one scenario. A
                       type with no entry, or an empty df (e.g. LeakDB's reservoir, whose
                       head is excluded as constant across scenarios), gets a 1-column
                       zero placeholder instead.
        """
        data = HeteroData()

        for type_idx in (self.node_idx, self.link_idx):
            for ct, ids in type_idx.items():
                df = attrs_dfs.get(ct)
                if df is None or df.shape[1] == 0:
                    x = torch.zeros(len(ids), 1)
                else:
                    cols = list(df.columns)
                    x = torch.zeros(len(ids), len(cols))
                    for cid, i in ids.items():
                        x[i] = torch.tensor(df.loc[cid, cols].values, dtype=torch.float32)
                data[ct].x = x

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

    def build(self, base: HeteroData, sensor_rows: dict) -> HeteroData:
        """Clone the scenario base graph and attach dynamic measured_by edges for one timestamp.

        Args:
            sensor_rows: {signal_type: pd.Series}, one row per entry of self.SENSOR_TYPES —
                         the discretized bin index for each sensor at this timestamp, indexed
                         by the same raw node/link id topology uses (no "node_"/"link_" tag).
        """
        data = base.clone()
        ei = self._build_measured_by_edges(sensor_rows)
        data[('value_node', 'measured_by', 'sensor')].edge_index = ei
        if self.bidirectional:
            data[('sensor', 'has_measure', 'value_node')].edge_index = ei.flip(0)
        return data

    def _build_measured_by_edges(self, sensor_rows: dict) -> torch.Tensor:
        """Return edge_index [2, n_edges] connecting each value node to its sensor (value_node → sensor).
        Sensors with NaN readings are skipped."""
        src, dst = [], []

        for st in self.SENSOR_TYPES:
            for loc, bin_idx in sensor_rows[st].items():
                if pd.isna(bin_idx):
                    continue
                s = self.sensor_idx[(st, loc)]
                v = self.value_node_idx[(st, int(bin_idx))]
                src.append(v)
                dst.append(s)

        return self._ei(src, dst)
