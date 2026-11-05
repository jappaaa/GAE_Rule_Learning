import os
import webbrowser

from torch_geometric.data import HeteroData

from src.data.graph_builder import GraphBuilder

# Fixed colors for the two sensor-abstraction node types, which exist regardless of dataset.
_FIXED_NODE_COLORS = {
    'sensor':     '#E67E22',
    'value_node': '#E74C3C',
}

# Palette cycled across whatever physical node/link types a given topology actually has
# (junction/reservoir/pipe for LeakDB, plus tank/pump/prv/... for DiTEC-WDN) — assigned in
# _node_colors_for() rather than hardcoded here, since the set of types varies by dataset.
_COMPONENT_COLOR_PALETTE = ['#4A90D9', '#27AE60', '#95A5A6', '#8E44AD', '#16A085', '#D35400', '#2C3E50', '#C0392B']

# Colors keyed by relation name only (not the full (src, rel, dst) triple), so any new
# component type reuses the same coloring for 'connected'/'has_sensor'/'located_at' edges.
_EDGE_COLORS = {
    'measured_by':  '#E74C3C',
    'has_measure':  '#C0392B',
    'has_sensor':   '#F39C12',
    'located_at':   '#F8C471',
    'connected':    '#85C1E9',
}


def _node_colors_for(gb: GraphBuilder) -> dict:
    """{node_type: color}, covering every component type present in gb.node_idx/link_idx
    (cycled from _COMPONENT_COLOR_PALETTE) plus the two fixed sensor-abstraction types."""
    colors = dict(_FIXED_NODE_COLORS)
    component_types = sorted(gb.node_idx.keys()) + sorted(gb.link_idx.keys())
    for i, ct in enumerate(component_types):
        colors[ct] = _COMPONENT_COLOR_PALETTE[i % len(_COMPONENT_COLOR_PALETTE)]
    return colors


def visualize_graph(data: HeteroData, gb: GraphBuilder,
                    title: str = "Graph", output_path: str = "graph.html",
                    open_browser: bool = True) -> None:
    """Save an interactive pyvis HTML visualization of a HeteroData graph.

    Nodes are colored by type; measured_by edges are drawn thick and red so the
    dynamic part of the graph stands out immediately.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis is required for graph visualization: pip install pyvis")

    # {node_type: {local_idx: original_id}}, covering every component type (not just
    # junction/reservoir/pipe) so labels work for any dataset GraphBuilder was built from.
    idx_to_component = {}
    for type_idx in (gb.node_idx, gb.link_idx):
        for ct, ids in type_idx.items():
            idx_to_component[ct] = {v: k for k, v in ids.items()}
    idx_to_sensor = {v: k for k, v in gb.sensor_idx.items()}
    idx_to_value  = {v: k for k, v in gb.value_node_idx.items()}

    node_colors = _node_colors_for(gb)

    net = Network(height='850px', width='100%', directed=True, heading=title)
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=120, spring_strength=0.05)

    node_map = {}
    global_id = 0

    for node_type in data.node_types:
        n = data[node_type].num_nodes
        color = node_colors.get(node_type, '#9B59B6')
        for i in range(n):
            if node_type in idx_to_component:
                label = f"{node_type[:3].upper()}:{idx_to_component[node_type].get(i, i)}"
            elif node_type == 'sensor':
                st, sname = idx_to_sensor.get(i, ('?', str(i)))
                label = f"{st[:3]}:{sname}"
            elif node_type == 'value_node':
                st, bin_idx = idx_to_value.get(i, ('?', i))
                label = f"{st[:3]}:b{bin_idx}"
            else:
                label = f"{node_type}:{i}"

            node_map[(node_type, i)] = global_id
            net.add_node(global_id, label=label, color=color,
                         title=f"[{node_type}] {label}", group=node_type,
                         size=12 if node_type in ('sensor', 'value_node') else 16)
            global_id += 1

    for edge_type in data.edge_types:
        src_type, rel, dst_type = edge_type
        edge_store = data[edge_type]
        if not hasattr(edge_store, 'edge_index') or edge_store.edge_index is None:
            continue
        ei = edge_store.edge_index
        if ei.numel() == 0:
            continue
        color = _EDGE_COLORS.get(rel, '#BDC3C7')
        width = 4 if rel == 'measured_by' else 1
        for j in range(ei.shape[1]):
            src_g = node_map.get((src_type, ei[0, j].item()))
            dst_g = node_map.get((dst_type, ei[1, j].item()))
            if src_g is not None and dst_g is not None:
                net.add_edge(src_g, dst_g, color=color, width=width, title=rel)

    net.show_buttons(filter_=['physics'])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    net.save_graph(output_path)
    print(f"  Saved → {output_path}")
    if open_browser:
        webbrowser.open(f"file:///{os.path.abspath(output_path)}")
