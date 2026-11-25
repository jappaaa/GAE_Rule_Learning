"""Builds PyG HeteroData objects for the LBNL Building 59 graph. Does NOT subclass
GraphBuilder -- GraphBuilder's node/sensor indexing assumes every node of a type shares
the same sensor types uniformly (true for WDN junctions/pipes, false here: RTU/Zone/VAV
each have disjoint, non-uniform point sets). Reworked piece by piece instead, reusing
GraphBuilder's patterns (node indexing, static-edge building, build_base/build split)
where they still apply -- see docs/experiment_plan.md.

Run from the project root to check all steps (node/sensor indices, static edges, node
features, build_base()/build(), hop maps):
    python -m src.data.lbnl_b59.graph_builder
"""
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.lbnl_b59.loader import LBNLLoader

# Structural-to-structural Brick relations materialized as graph edges (hasPoint/isPointOf
# are handled separately, via has_sensor/located_at -- see _build_static_edges). Confirmed
# these 4 predicates only ever connect two structural entities, never a point type entity.
STRUCTURAL_RELATIONS = {'isPartOf', 'hasLocation', 'feeds', 'isFedBy'}


class LBNLGraphBuilder:
    """Node types: Zone, VAV, Rooftop_Unit, HVAC_Zone, Floor, Building (+ sensor,
    value_node once later steps add them)."""

    def __init__(self, entities: dict, relations: list, raw_ids_by_type: dict,
                 n_bins_per_type: dict, raw_id_owner: dict, bidirectional: bool = False,
                 virtual_node_mode: str = 'none', onehot_bin_features: bool = True):
        self.entities = entities
        self.relations = relations
        self.n_bins_per_type = n_bins_per_type
        self.bidirectional = bidirectional
        self.virtual_node_mode = virtual_node_mode
        self.onehot_bin_features = onehot_bin_features

        self._build_node_indices()
        self._build_sensor_indices(raw_ids_by_type)
        self._build_virtual_node_indices()
        self._build_static_edges(raw_id_owner)
        self._build_virtual_node_edges()
        self._build_static_node_features()

    def _build_node_indices(self):
        """Assign integer indices from 0 to n_nodes - 1 within each structural entity
        type, derived generically from classify_entities()'s structural_ids (not
        hardcoded to Zone/VAV/...), each entity keeping its original ttl id as the key.
        """
        structural_ids, _ = LBNLLoader.classify_entities(self.entities, self.relations)

        node_types = sorted({self.entities[eid]['type'][0] for eid in structural_ids})
        self.node_idx = {}
        for nt in node_types:
            ids = sorted(eid for eid in structural_ids if self.entities[eid]['type'][0] == nt)
            self.node_idx[nt] = {eid: i for i, eid in enumerate(ids)}

    def _build_sensor_indices(self, raw_ids_by_type: dict):
        """Assign integer indices to sensors and value nodes.

        Unlike GraphBuilder, self.sensor_idx is built directly from raw_ids_by_type
        (LBNLDataset.process()'s Step 1/3 output) rather than reconstructed from a
        "every node of a type has the same sensor types" assumption -- RTU/Zone/VAV
        each have disjoint, non-uniform point sets here, so that assumption doesn't
        hold. self.SENSOR_TYPES is sorted for a deterministic ordering (raw_ids_by_type
        comes from a dict whose insertion order isn't itself meaningful).

        self.value_node_idx / n_sensors / n_value_nodes / sensor_indices_by_type /
        value_indices_by_type are otherwise identical in shape and construction to
        GraphBuilder's -- these are driven only by self.SENSOR_TYPES + sensor_idx/
        value_node_idx, with no WDN-specific assumption baked in.
        """
        self.SENSOR_TYPES = sorted(raw_ids_by_type)

        self.sensor_idx = {}
        idx = 0
        for st in self.SENSOR_TYPES:
            for raw_id in raw_ids_by_type[st]:
                self.sensor_idx[(st, raw_id)] = idx
                idx += 1

        self.value_node_idx = {}
        offset = 0
        for st in self.SENSOR_TYPES:
            for bin_idx in range(self.n_bins_per_type[st]):
                self.value_node_idx[(st, bin_idx)] = offset + bin_idx
            offset += self.n_bins_per_type[st]

        self.n_sensors = len(self.sensor_idx)
        self.n_value_nodes = len(self.value_node_idx)

        # these are required for loss calculation of the model
        self.sensor_indices_by_type = {
            st: torch.tensor([idx for (stype, _), idx in self.sensor_idx.items() if stype == st])
            for st in self.SENSOR_TYPES
        }
        self.value_indices_by_type = {
            st: torch.tensor([idx for (vtype, _), idx in self.value_node_idx.items() if vtype == st])
            for st in self.SENSOR_TYPES
        }

    def _build_virtual_node_indices(self):
        """Assign integer indices to virtual nodes, if enabled by virtual_node_mode -- kept
        separate from _build_sensor_indices (same reasoning as splitting node indices from
        sensor indices into their own methods), even though both are no-ops when
        virtual_node_mode == 'none'.
        """
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            self.type_virtual_idx = {st: i for i, st in enumerate(self.SENSOR_TYPES)}
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            self.global_virtual_idx = {0: 0}

    @staticmethod
    def _ei(src: list, dst: list) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

    def _build_static_edges(self, raw_id_owner: dict):
        """Build bidirectional edge_index tensors for every static relation in the graph.

        self.structural_edges: one bidirectional pair per (subj_type, predicate, obj_type)
        combination actually present among STRUCTURAL_RELATIONS -- built generically from
        self.relations, not hardcoded (e.g. 'feeds' alone covers both
        Rooftop_Unit->HVAC_Zone and VAV->Zone, so it naturally yields 2 pairs). The reverse
        direction is named 'rev_<predicate>' rather than reusing a real Brick inverse
        predicate name (e.g. 'hasPart' for isPartOf) -- bidirectional edges are a GNN
        message-passing design choice here, not a ttl-asserted relation (same reasoning
        already applied to has_measure/measured_by), so the reverse name shouldn't imply
        otherwise.

        self.has_sensor_edges: one (component_type, 'has_sensor', 'sensor') + reverse
        ('sensor', 'located_at', component_type) pair per structural type that owns at
        least one sensor -- same naming as the shared GraphBuilder. raw_id_owner (LBNLDataset
        .process()'s raw_id_owner.json) gives each sensor's owner directly, already resolved
        from hasPoint/isPointOf plus the rename/zone-prefix-fallback cases -- not re-derived
        from self.relations here.
        """
        node_type_of = {eid: nt for nt, ids in self.node_idx.items() for eid in ids}

        rel_pairs = defaultdict(lambda: ([], []))  # (subj_type, predicate, obj_type) -> (subj_side, obj_side)
        for s, p, o in self.relations:
            if p not in STRUCTURAL_RELATIONS:
                continue
            st, ot = node_type_of[s], node_type_of[o]
            s_idx, o_idx = self.node_idx[st][s], self.node_idx[ot][o]
            s_side, o_side = rel_pairs[(st, p, ot)]
            s_side.append(s_idx); o_side.append(o_idx)

        self.structural_edges = {}
        for (st, p, ot), (s_side, o_side) in rel_pairs.items():
            self.structural_edges[(st, p, ot)] = self._ei(s_side, o_side)
            self.structural_edges[(ot, f'rev_{p}', st)] = self._ei(o_side, s_side)

        sensor_pairs = defaultdict(lambda: ([], []))  # component_type -> (comp_side, sensor_side)
        for (st, raw_id), s_idx in self.sensor_idx.items():
            owner = raw_id_owner[raw_id]
            ot = node_type_of[owner]
            o_idx = self.node_idx[ot][owner]
            c_side, s_side = sensor_pairs[ot]
            c_side.append(o_idx); s_side.append(s_idx)

        self.has_sensor_edges = {}
        for ct, (c_side, s_side) in sensor_pairs.items():
            self.has_sensor_edges[(ct, 'has_sensor', 'sensor')] = self._ei(c_side, s_side)
            self.has_sensor_edges[('sensor', 'located_at', ct)] = self._ei(s_side, c_side)

    def _build_virtual_node_edges(self):
        """Build edge indices connecting sensors to virtual nodes based on virtual_node_mode.
        Ported verbatim from GraphBuilder -- driven only by self.virtual_node_mode,
        self.sensor_idx, self.SENSOR_TYPES and self.n_sensors, none of which carry any
        WDN-specific (junction/pipe) assumption, so no rework is needed here.

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
        """Build fixed node feature tensors for sensor and value_node types (+ virtual
        node types, if enabled). Ported verbatim from GraphBuilder -- driven only by
        self.SENSOR_TYPES, self.sensor_idx, self.value_node_idx and self.n_bins_per_type,
        none of which carry any WDN-specific assumption. Structural node types (Zone/VAV/
        Rooftop_Unit/HVAC_Zone/Floor/Building) are NOT built here -- same as GraphBuilder,
        those come from attrs_dfs in build_base() instead (Step 5), since that's where the
        zero-placeholder fallback for featureless types is decided.

        sensor features        (n_sensors,          n_sensor_types): one-hot sensor type
        value_node feats       (n_value_nodes,      max_bins+n_sensor_types) or (n_value_nodes, 1+n_sensor_types)
        type_virtual feats     (n_sensor_types,     n_sensor_types): one-hot sensor type (same encoding as sensor)
        global_virtual feats   (1,                  1): zero initialization (identity derived from aggregation)
        """
        # helper to guarantee one hot features represent same type uniformly
        type_to_idx = {st: i for i, st in enumerate(self.SENSOR_TYPES)}

        sensor_x = torch.zeros(self.n_sensors, len(self.SENSOR_TYPES))
        for (st, _), s_i in self.sensor_idx.items():
            sensor_x[s_i, type_to_idx[st]] = 1.0
        self.sensor_x = sensor_x

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
        """Return ({component_type: global_offset}, total_count), spanning self.node_idx --
        no node_idx/link_idx split here, unlike GraphBuilder (LBNL has no pipe-like link
        type). Shared by build_sensor_distance_map/build_sensor_hop_map, which flatten
        every structural component type into one adjacency array for BFS.
        """
        type_off = {}
        offset = 0
        for ct, ids in self.node_idx.items():
            type_off[ct] = offset
            offset += len(ids)
        return type_off, offset

    def build_sensor_distance_map(self) -> dict:
        """Return {sensor_idx: {other_sensor_idx: structural_hop_distance}}.

        Distance is the shortest path between sensor *location* nodes (the structural
        entity each sensor is attached to, via has_sensor_edges/raw_id_owner) in the Brick
        structural graph, traversing only structural_edges (isPartOf/hasLocation/feeds/
        isFedBy, both directions):
          0 = co-located sensors (same structural entity, e.g. two points on the same Zone)
          1 = directly connected structural entities (e.g. a Zone and its RTU's HVAC_Zone)
          2 = two structural hops apart, etc.
        More interpretable than raw graph hops because it counts Brick structural
        relations, not PyG edges (sensor/value_node hops aren't counted). Mirrors
        GraphBuilder.build_sensor_distance_map() exactly, just over self.structural_edges
        instead of self.connected_edges and with no node_idx/link_idx split. Note:
        build_sensor_hop_map uses raw graph hops -- kept separate, same reasoning as
        GraphBuilder's docstring (serves the GNN receptive-field filter, where graph
        layers are the natural unit).
        """
        type_off, n_network_nodes = self._component_offsets()

        # adjacency on structural nodes only -- no sensor nodes, no measured_by edges
        adj = [[] for _ in range(n_network_nodes)]
        for (src_type, _, dst_type), ei in self.structural_edges.items():
            o_src, o_dst = type_off[src_type], type_off[dst_type]
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                adj[o_src + s].append(o_dst + d)

        # map each sensor to its location node (the structural entity it's attached to)
        sensor_location = {}
        for (src_type, _, dst_type), ei in self.has_sensor_edges.items():
            if dst_type != 'sensor':
                continue
            o_src = type_off[src_type]
            for loc, s in zip(ei[0].tolist(), ei[1].tolist()):
                sensor_location[s] = o_src + loc

        # reverse: structural node -> list of sensors at that location
        loc_to_sensors = defaultdict(list)
        for s_idx, loc in sensor_location.items():
            loc_to_sensors[loc].append(s_idx)

        # BFS from each sensor's location through the structural graph
        distance_map = {}
        for s_idx in range(self.n_sensors):
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

        Uses structural_edges + has_sensor_edges only (no measured_by/has_measure), so the
        result is independent of any particular timestamp's dynamic edges. Pass
        max_hops = num_layers - 1 to account for the one hop from antecedent value_node to
        antecedent sensor before traversing topology. Mirrors
        GraphBuilder.build_sensor_hop_map() exactly, just with no node_idx/link_idx split.
        """
        type_off, off_s = self._component_offsets()
        type_off = {**type_off, 'sensor': off_s}
        total = off_s + self.n_sensors

        adj = [[] for _ in range(total)]
        for (src_type, _, dst_type), ei in {**self.structural_edges, **self.has_sensor_edges}.items():
            o_src = type_off[src_type]
            o_dst = type_off[dst_type]
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                adj[o_src + s].append(o_dst + d)

        sensor_hop_map = {}
        for s_idx in range(self.n_sensors):
            start = off_s + s_idx
            dist = {start: 0}
            queue = [start]
            reachable = set()  # start without the node itself (these are the antecedents and should not be considered)
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
        """Build the static HeteroData for the (one and only) LBNL base graph. Unlike
        LeakDB/DiTEC-WDN, there's no per-scenario variation here (see
        docs/experiment_plan.md), so this is called once total rather than once per
        scenario -- otherwise the same structure as GraphBuilder.build_base(), just over
        a single self.node_idx instead of a node_idx/link_idx split (LBNL has no pipe-like
        link type).

        Args:
            attrs_dfs: {component_type: df}, one DataFrame per structural type with a
                       derivable feature (LBNLLoader.derive_structural_attributes()'s
                       output: Zone, VAV), indexed by entity id. A type with no entry
                       (Rooftop_Unit/HVAC_Zone/Floor/Building) gets a 1-column zero
                       placeholder instead -- same convention as GraphBuilder's (e.g.
                       LeakDB's reservoir).
        """
        data = HeteroData()

        for ct, ids in self.node_idx.items():
            df = attrs_dfs.get(ct)
            if df is None or df.shape[1] == 0:
                x = torch.zeros(len(ids), 1)
            else:
                # creates the tensor where the rows from df indexed by OG entity id are placed at the corresponding PyG id row in the tensor
                cols = list(df.columns)
                x = torch.zeros(len(ids), len(cols))
                for eid, i in ids.items():
                    x[i] = torch.tensor(df.loc[eid, cols].values, dtype=torch.float32)
            data[ct].x = x

        # sensor and value_node features are fixed, independent of attrs_dfs
        data['sensor'].x = self.sensor_x
        data['value_node'].x = self.value_node_x

        for edge_type, ei in self.structural_edges.items():
            data[edge_type].edge_index = ei
        for edge_type, ei in self.has_sensor_edges.items():
            data[edge_type].edge_index = ei

        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'type_interconnected'):
            data['type_virtual'].x = self.type_virtual_x
        if self.virtual_node_mode in ('hierarchical', 'hierarchical_direct', 'global'):
            data['global_virtual'].x = self.global_virtual_x
        for edge_type, ei in self.virtual_node_edges.items():
            data[edge_type].edge_index = ei

        return data

    def build(self, base: HeteroData, sensor_rows: dict) -> HeteroData:
        """Clone the base graph and attach dynamic measured_by edges for one timestamp.

        Args:
            sensor_rows: {sensor_type: pd.Series}, one row per entry of self.SENSOR_TYPES --
                         the discretized bin index for each sensor at this timestamp,
                         indexed by raw_id (the same keys self.sensor_idx uses, no
                         "node_"/"link_" tag needed since LBNL's raw_ids are already
                         globally unique).
        """
        data = base.clone()
        ei = self._build_measured_by_edges(sensor_rows)
        data[('value_node', 'measured_by', 'sensor')].edge_index = ei
        if self.bidirectional:
            data[('sensor', 'has_measure', 'value_node')].edge_index = ei.flip(0)
        return data

    def _build_measured_by_edges(self, sensor_rows: dict) -> torch.Tensor:
        """Return edge_index [2, n_edges] connecting each value node to its sensor
        (value_node -> sensor). Sensors with NaN readings are skipped."""
        src, dst = [], []

        for st in self.SENSOR_TYPES:
            for raw_id, bin_idx in sensor_rows[st].items():
                if pd.isna(bin_idx):
                    continue
                s = self.sensor_idx[(st, raw_id)]
                v = self.value_node_idx[(st, int(bin_idx))]
                src.append(v)
                dst.append(s)

        return self._ei(src, dst)


if __name__ == '__main__':
    loader = LBNLLoader(os.path.join('data', 'lbnl_b59', 'raw'))
    result = loader.load_entities_and_relations()
    entities, relations = result['entities'], result['relations']

    processed_dir = os.path.join('data', 'lbnl_b59', 'processed')
    with open(os.path.join(processed_dir, 'raw_ids_by_type.json')) as f:
        raw_ids_by_type = json.load(f)
    with open(os.path.join(processed_dir, 'metadata.json')) as f:
        n_bins_per_type = json.load(f)['n_bins_per_type']
    with open(os.path.join(processed_dir, 'raw_id_owner.json')) as f:
        raw_id_owner = json.load(f)

    gb = LBNLGraphBuilder(entities, relations, raw_ids_by_type, n_bins_per_type, raw_id_owner)

    print(f"node types: {sorted(gb.node_idx)}")
    for nt, ids in sorted(gb.node_idx.items()):
        print(f"  {nt:15s} n={len(ids)}  e.g. {list(ids.items())[:3]}")

    print(f"\n{len(gb.SENSOR_TYPES)} sensor types, n_sensors={gb.n_sensors}, n_value_nodes={gb.n_value_nodes}")
    print(f"sensor_idx matches raw_ids_by_type count: {gb.n_sensors == sum(len(v) for v in raw_ids_by_type.values())}")
    print(f"value_node_idx matches n_bins_per_type sum: {gb.n_value_nodes == sum(n_bins_per_type.values())}")

    for st in gb.SENSOR_TYPES[:3]:
        print(f"  {st:45s} sensor_indices_by_type len={len(gb.sensor_indices_by_type[st])}  "
              f"value_indices_by_type len={len(gb.value_indices_by_type[st])} (n_bins={n_bins_per_type[st]})")

    print(f"\n--- static edges ---")
    print(f"structural_edges: {sorted(gb.structural_edges)}")
    for edge_type, ei in sorted(gb.structural_edges.items(), key=str):
        print(f"  {edge_type}: {ei.shape[1]} edges")

    print(f"\nhas_sensor_edges component types: {sorted({ct for ct, _, dt in gb.has_sensor_edges if dt == 'sensor'})}")
    n_has_sensor = sum(ei.shape[1] for (ct, rel, dt), ei in gb.has_sensor_edges.items() if rel == 'has_sensor')
    n_located_at = sum(ei.shape[1] for (ct, rel, dt), ei in gb.has_sensor_edges.items() if rel == 'located_at')
    print(f"total has_sensor edges: {n_has_sensor}  |  total located_at edges: {n_located_at}  "
          f"|  both == n_sensors ({gb.n_sensors}): {n_has_sensor == n_located_at == gb.n_sensors}")

    print(f"\n--- static node features (virtual_node_mode='none') ---")
    print(f"sensor_x shape: {gb.sensor_x.shape} (expect ({gb.n_sensors}, {len(gb.SENSOR_TYPES)}))")
    print(f"value_node_x shape: {gb.value_node_x.shape} "
          f"(expect ({gb.n_value_nodes}, {max(n_bins_per_type.values()) + len(gb.SENSOR_TYPES)}))")
    print(f"sensor_x rows sum to 1 (one-hot): {(gb.sensor_x.sum(dim=1) == 1).all().item()}")
    print(f"value_node_x rows sum to 2 (one-hot bin + one-hot type): {(gb.value_node_x.sum(dim=1) == 2).all().item()}")
    print(f"virtual_node_edges (expect empty): {gb.virtual_node_edges}")

    print(f"\n--- re-built with virtual_node_mode='hierarchical' ---")
    gb_vn = LBNLGraphBuilder(entities, relations, raw_ids_by_type, n_bins_per_type, raw_id_owner,
                              virtual_node_mode='hierarchical')
    print(f"virtual_node_edges: {sorted(gb_vn.virtual_node_edges)}")
    for edge_type, ei in sorted(gb_vn.virtual_node_edges.items(), key=str):
        print(f"  {edge_type}: {ei.shape[1]} edges")
    print(f"type_virtual_x shape: {gb_vn.type_virtual_x.shape}  |  global_virtual_x shape: {gb_vn.global_virtual_x.shape}")

    print(f"\n--- build_base() / build() ---")
    attrs_dfs = loader.derive_structural_attributes(entities, relations)
    base = gb.build_base(attrs_dfs)
    base.validate(raise_on_error=True)
    print(base)
    for ct in gb.node_idx:
        print(f"  {ct:15s} x shape: {base[ct].x.shape}")

    # dataset_categorical.parquet's sensor columns are "sensor_type;raw_id" (see
    # LBNLDataset.process()'s Step 3 -- needed so Aerial's export.py can tell item columns
    # apart from metadata columns), not the bare raw_id gb.build() expects, so each
    # sensor_type's slice is renamed back to bare raw_id, same as LeakDBDataset.process()
    # does for its own ";"-prefixed sensor columns.
    cat_df = pd.read_parquet(os.path.join(processed_dir, 'dataset_categorical.parquet'))
    row = cat_df.iloc[0]
    sensor_rows = {
        st: row[[f'{st};{rid}' for rid in raw_ids_by_type[st]]].rename(index=lambda c: c.split(';')[1])
        for st in gb.SENSOR_TYPES
    }
    graph = gb.build(base, sensor_rows)
    graph.validate(raise_on_error=True)
    n_measured_by = graph[('value_node', 'measured_by', 'sensor')].edge_index.shape[1]
    print(f"\nmeasured_by edges for 1 timestamp: {n_measured_by} (expect n_sensors={gb.n_sensors}, since dataset_categorical has 0 NaN)")
    print(f"has_measure edges present (expect False, bidirectional=False by default): "
          f"{('sensor', 'has_measure', 'value_node') in graph.edge_types}")

    gb_bidir = LBNLGraphBuilder(entities, relations, raw_ids_by_type, n_bins_per_type, raw_id_owner, bidirectional=True)
    base_bidir = gb_bidir.build_base(attrs_dfs)
    graph_bidir = gb_bidir.build(base_bidir, sensor_rows)
    graph_bidir.validate(raise_on_error=True)
    print(f"\nwith bidirectional=True: has_measure edges: {graph_bidir[('sensor', 'has_measure', 'value_node')].edge_index.shape[1]} "
          f"(expect == measured_by's {n_measured_by})")

    print(f"\n--- hop maps ---")
    distance_map = gb.build_sensor_distance_map()
    print(f"build_sensor_distance_map(): {len(distance_map)} sensors covered "
          f"(expect n_sensors={gb.n_sensors}): {len(distance_map) == gb.n_sensors}")
    all_dists = [d for other in distance_map.values() for d in other.values()]
    print(f"max structural hop distance between any two sensors: {max(all_dists)}")
    print(f"min/mean: {min(all_dists)} / {sum(all_dists) / len(all_dists):.2f}")
    s0 = 0
    print(f"sensor 0 ({list(gb.sensor_idx)[0]}): "
          f"{len(distance_map[s0])} other sensors reachable, "
          f"{sum(1 for d in distance_map[s0].values() if d == 0)} co-located (0 hops)")

    hop_map_1 = gb.build_sensor_hop_map(max_hops=1)
    hop_map_5 = gb.build_sensor_hop_map(max_hops=5)
    avg_reach_1 = sum(len(v) for v in hop_map_1.values()) / len(hop_map_1)
    avg_reach_5 = sum(len(v) for v in hop_map_5.values()) / len(hop_map_5)
    print(f"\nbuild_sensor_hop_map(max_hops=1): avg reachable sensors per sensor = {avg_reach_1:.1f} "
          f"(out of {gb.n_sensors - 1} others)")
    print(f"build_sensor_hop_map(max_hops=5): avg reachable sensors per sensor = {avg_reach_5:.1f}")
    hop_map_full = gb.build_sensor_hop_map(max_hops=100)  # effectively unbounded for this graph's size
    avg_reach_full = sum(len(v) for v in hop_map_full.values()) / len(hop_map_full)
    n_fully_connected = sum(1 for v in hop_map_full.values() if len(v) == gb.n_sensors - 1)
    print(f"build_sensor_hop_map(max_hops=100, effectively unbounded): avg reachable = {avg_reach_full:.1f}  "
          f"(all {gb.n_sensors} sensors reach every other sensor: {n_fully_connected == gb.n_sensors})")
