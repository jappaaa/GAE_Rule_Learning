import os
import webbrowser

from torch_geometric.data import HeteroData

from src.data.graph_builder import GraphBuilder

_NODE_COLORS = {
    'junction':   '#4A90D9',
    'reservoir':  '#27AE60',
    'pipe':       '#95A5A6',
    'sensor':     '#E67E22',
    'value_node': '#E74C3C',
}

_EDGE_COLORS = {
    ('value_node', 'measured_by',  'sensor'):    '#E74C3C',
    ('sensor',     'has_measure',  'value_node'): '#C0392B',
    ('junction',   'has_sensor',   'sensor'):    '#F39C12',
    ('reservoir',  'has_sensor',   'sensor'):    '#F39C12',
    ('pipe',       'has_sensor',   'sensor'):    '#F39C12',
    ('sensor',     'located_at',   'junction'):  '#F8C471',
    ('sensor',     'located_at',   'reservoir'): '#F8C471',
    ('sensor',     'located_at',   'pipe'):      '#F8C471',
    ('junction',   'connected',    'pipe'):      '#85C1E9',
    ('pipe',       'connected',    'junction'):  '#85C1E9',
    ('reservoir',  'connected',    'pipe'):      '#1A5276',
    ('pipe',       'connected',    'reservoir'): '#1A5276',
}


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

    idx_to_junction  = {v: k for k, v in gb.junction_idx.items()}
    idx_to_reservoir = {v: k for k, v in gb.reservoir_idx.items()}
    idx_to_pipe      = {v: k for k, v in gb.pipe_idx.items()}
    idx_to_sensor    = {v: k for k, v in gb.sensor_idx.items()}
    idx_to_value     = {v: k for k, v in gb.value_node_idx.items()}

    net = Network(height='850px', width='100%', directed=True, heading=title)
    net.barnes_hut(gravity=-4000, central_gravity=0.3, spring_length=120, spring_strength=0.05)

    node_map = {}
    global_id = 0

    for node_type in data.node_types:
        n = data[node_type].num_nodes
        color = _NODE_COLORS.get(node_type, '#9B59B6')
        for i in range(n):
            if node_type == 'junction':
                label = f"J:{idx_to_junction.get(i, i)}"
            elif node_type == 'reservoir':
                label = f"R:{idx_to_reservoir.get(i, i)}"
            elif node_type == 'pipe':
                label = f"P:{idx_to_pipe.get(i, i)}"
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
        color = _EDGE_COLORS.get(edge_type, '#BDC3C7')
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
