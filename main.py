import dataclasses
import json
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataset import LeakDBDataset
from src.model.gae import GraphAutoEncoder
from src.training.trainer import Trainer
from src.rules.extractor import RuleExtractor
from src.rules.evaluator import RuleEvaluator
from src.utils.config import Config


def main():
    config = Config()
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = LeakDBDataset(config)
    print(f"Dataset ready — {len(dataset)} graphs | {len(dataset.train_indices)} train / {len(dataset.val_indices)} val / {len(dataset.test_indices)} test")

    print("Initialising model...")
    sample = dataset[0]
    model = GraphAutoEncoder(config, sample.metadata(), dataset.gb)
    model.to(device)
    print("Model ready")

    if config.train_model:
        print("Starting training...")
        trainer = Trainer(model, dataset, config, device)
        trainer.train()
        print("Training complete")
    else:
        print(f"Loading checkpoint from {config.checkpoint_path}...")
        checkpoint = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    print("\nProbing model sensitivity to antecedent...")
    gb = dataset.gb
    probe_st = 'pressure'
    probe_sname = next(sname for (st, sname) in gb.sensor_idx if st == probe_st)
    n_bins = gb.n_bins_per_type[probe_st]
    scenario = config.scenarios[0]
    model.eval()

    def _make_query(bin_idx):
        g = dataset.base_graphs[scenario].clone().to(device)
        v = gb.value_node_idx[(probe_st, bin_idx)]
        s = gb.sensor_idx[(probe_st, probe_sname)]
        g[('value_node', 'measured_by', 'sensor')].edge_index = torch.tensor([[v], [s]], dtype=torch.long, device=device)
        return g

    g_low  = _make_query(0)
    g_high = _make_query(n_bins - 1)
    with torch.no_grad():
        probs_low  = model(g_low.x_dict,  g_low.edge_index_dict)
        probs_high = model(g_high.x_dict, g_high.edge_index_dict)


    print(f"  Probe sensor: ({probe_st}, {probe_sname}) — comparing bin 0 vs bin {n_bins - 1}")
    for st in gb.SENSOR_TYPES:
        diff = (probs_low[st] - probs_high[st]).abs().mean().item()
        print(f"  mean |Δprob| for {st} sensors: {diff:.4f}")

    # Probe sensor's own predicted distribution (it has the measured_by edge directly)
    probe_s_idx = gb.sensor_idx[(probe_st, probe_sname)]
    probe_row = (gb.sensor_indices_by_type[probe_st] == probe_s_idx).nonzero(as_tuple=True)[0][0].item()
    low_p  = [round(p, 3) for p in probs_low[probe_st][probe_row].cpu().tolist()]
    high_p = [round(p, 3) for p in probs_high[probe_st][probe_row].cpu().tolist()]
    print(f"  Probe sensor bin probs (antecedent=bin 0):        {low_p}")
    print(f"  Probe sensor bin probs (antecedent=bin {n_bins-1}): {high_p}")

    # Group other sensors by GNN hop distance from the antecedent value_node:
    #   same junction   ≈ 3 hops  (value_node→sensor→junction→sensor)
    #   1 pipe away     ≈ 5 hops  (…→junction→pipe→junction→sensor)
    #   beyond          > 6 hops  (outside receptive field with num_layers=6)
    junc_hs_ei   = gb.has_sensor_edges[('junction', 'has_sensor', 'sensor')]
    pipe_hs_ei   = gb.has_sensor_edges[('pipe',     'has_sensor', 'sensor')]
    junc_pipe_ei = gb.connected_edges[('junction',  'connected',  'pipe')]
    pipe_junc_ei = gb.connected_edges[('pipe',      'connected',  'junction')]

    sensors_same_junc, sensors_1pipe = set(), set()
    probe_junc_mask = junc_hs_ei[1] == probe_s_idx
    if probe_junc_mask.any():
        probe_junc = junc_hs_ei[0, probe_junc_mask][0].item()
        sensors_same_junc = set(junc_hs_ei[1, junc_hs_ei[0] == probe_junc].tolist()) - {probe_s_idx}
        adj_pipes = junc_pipe_ei[1, junc_pipe_ei[0] == probe_junc].tolist()
        for p in adj_pipes:
            for adj_junc in pipe_junc_ei[1, pipe_junc_ei[0] == p].tolist():
                if adj_junc != probe_junc:
                    sensors_1pipe.update(junc_hs_ei[1, junc_hs_ei[0] == adj_junc].tolist())
            sensors_1pipe.update(pipe_hs_ei[1, pipe_hs_ei[0] == p].tolist())
        sensors_1pipe -= sensors_same_junc | {probe_s_idx}

    groups_diffs = {'same junction (~3 GNN hops)': [], '1 pipe away (~5 GNN hops)': [], 'beyond receptive field': []}
    for st in gb.SENSOR_TYPES:
        for row, s_idx in enumerate(gb.sensor_indices_by_type[st].tolist()):
            if s_idx == probe_s_idx:
                continue
            d = (probs_low[st][row] - probs_high[st][row]).abs().mean().item()
            if s_idx in sensors_same_junc:
                groups_diffs['same junction (~3 GNN hops)'].append(d)
            elif s_idx in sensors_1pipe:
                groups_diffs['1 pipe away (~5 GNN hops)'].append(d)
            else:
                groups_diffs['beyond receptive field'].append(d)

    for label, diffs in groups_diffs.items():
        if diffs:
            print(f"  {label}: mean |Δprob| = {sum(diffs)/len(diffs):.4f}  (n={len(diffs)})")

    _idx_to_sensor = {v: k for k, v in gb.sensor_idx.items()}
    for st in gb.SENSOR_TYPES:
        print(f"\n  Raw probs_low[{st}] per sensor (antecedent=bin 0):")
        for row, s_idx in enumerate(gb.sensor_indices_by_type[st].tolist()):
            _, sname = _idx_to_sensor[s_idx]
            if s_idx == probe_s_idx:
                label = 'PROBE'
            elif s_idx in sensors_same_junc:
                label = 'same_junc'
            elif s_idx in sensors_1pipe:
                label = '1pipe'
            else:
                label = ''
            p = [round(v, 3) for v in probs_low[st][row].cpu().tolist()]
            print(f"    row {row:3d} ({sname:12s}) {label:10s}: {p}")

    print("\nInformation leakage probe — does the model reconstruct impossible antecedents?")
    leakage_cases = [
        ('flow', 'link_1', 9, [0, 5, 8]),   # always bin 0; test impossible bins
        ('flow', 'link_27', 1, [0, 5, 9]),   # always bin 1; test impossible bins
    ]
    for ant_st, ant_sname, real_bin, test_bins in leakage_cases:
        ant_s_idx = gb.sensor_idx[(ant_st, ant_sname)]
        ant_row = (gb.sensor_indices_by_type[ant_st] == ant_s_idx).nonzero(as_tuple=True)[0][0].item()
        print(f"  Sensor ({ant_st}, {ant_sname}) — real bin: {real_bin}")
        for test_bin in [real_bin] + test_bins:
            g = dataset.base_graphs[scenario].clone().to(device)
            v = gb.value_node_idx[(ant_st, test_bin)]
            g[('value_node', 'measured_by', 'sensor')].edge_index = \
                torch.tensor([[v], [ant_s_idx]], dtype=torch.long, device=device)
            with torch.no_grad():
                p = model(g.x_dict, g.edge_index_dict)
            probs_row = [round(x, 3) for x in p[ant_st][ant_row].cpu().tolist()]
            tag = '← real' if test_bin == real_bin else '← impossible'
            print(f"    injected bin {test_bin}: probs={probs_row}  {tag}")

    print("\nBin frequency check across test timestamps:")
    sensors_to_check = [
        ('pressure', 'node_1'),
        ('demand',   'node_1'),
        ('flow',     'link_18'),
        ('flow',     'link_19'),
        ('flow',     'link_27'),
        ('flow',     'link_32'),
        ('demand',   'node_19'),
        ('flow',     'link_1'),
    ]
    item_to_col = {item: col for col, item in enumerate(dataset.test_items)}
    tt = dataset.test_tensor.float()
    for st, sname in sensors_to_check:
        parts = []
        for bin_idx in range(gb.n_bins_per_type[st]):
            key = (st, sname, bin_idx)
            if key in item_to_col:
                frac = tt[:, item_to_col[key]].mean().item()
                parts.append(f"bin{bin_idx}={frac:.3f}")
        print(f"  ({st:8s}, {sname:8s}): {' '.join(parts)}")

    if config.learn_rules:
        print("Extracting rules...")
        extractor = RuleExtractor(model, dataset.gb, dataset, config, device)
        rules = extractor.extract()
        print(f"Extraction complete — {len(rules)} candidate rules")
        Path(config.rules_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.rules_path, 'w') as f:
            json.dump(rules, f)
        print(f"Rules saved to {config.rules_path}")
    else:
        rules_path = Path(config.rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(
                f"learn_rules=False but no rules file found at '{config.rules_path}'. "
                "Set learn_rules=True to extract and save rules first."
            )
        print(f"Loading rules from {config.rules_path}...")
        with open(rules_path) as f:
            raw = json.load(f)
        rules = [
            {
                'antecedent': [tuple(item) for item in rule['antecedent']],
                'consequent': tuple(rule['consequent']),
            }
            for rule in raw
        ]
        print(f"Loaded {len(rules)} rules")

    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")

    print("Evaluating rules...")
    evaluator = RuleEvaluator(dataset, config, device)
    evaluated_rules, averages = evaluator.evaluate(rules)
    if config.filter_rules:
        print(f"Evaluation complete — {len(evaluated_rules)} rules pass support>={config.min_support} and confidence>={config.min_confidence}")
    else:
        print(f"Evaluation complete — {len(evaluated_rules)} rules (no filtering applied)")
    print(f"Averages over all {len(rules)} rules: " + " | ".join(f"{k}={v:.4f}" for k, v in averages.items()))
    for rule in evaluated_rules[:5]:
        print(f"  support={rule['support']:.3f} conf={rule['confidence']:.3f} lift={rule['lift']:.3f} zhang={rule['zhang']:.3f} | {rule['antecedent']} -> {rule['consequent']}")
    print(f"Data coverage: {averages.get('coverage', 'n/a')}")

    results = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': dataclasses.asdict(config),
        'n_rules_total': len(rules),
        'n_rules_after_filter': len(evaluated_rules),
        'averages': averages,
        'top_rules': evaluated_rules[:10],
    }
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"run_{results['timestamp'].replace(':', '-')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
