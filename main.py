import copy
import dataclasses
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from src.data.dataset import LeakDBDataset
from src.model.gae import GraphAutoEncoder
from src.training.trainer import Trainer
from src.training.masking import Masker
from src.rules.extractor import RuleExtractor
from src.rules.fp_extractor import FPExtractor
from src.rules.evaluator import RuleEvaluator, annotate_hop_distances
from src.utils.config import Config
from src.utils.visualize import visualize_graph


def _save_rules_output(evaluated_rules, averages, all_rules, run_dir, tag, config):
    rows = []
    for r in evaluated_rules:
        row = {'antecedent': str(r['antecedent']), 'consequent': str(r['consequent'])}
        for k in ('support', 'support_ant', 'confidence', 'lift', 'zhang', 'hop_distance_min', 'hop_distance_max'):
            if k in r:
                row[k] = r[k]
        rows.append(row)
    rules_df = pd.DataFrame(rows)
    rules_df.to_csv(run_dir / f'rules_{tag}.csv', index=False)
    print(f"Rules CSV saved to {run_dir / f'rules_{tag}.csv'}")

    results = {
        'tag': tag,
        'config': dataclasses.asdict(config),
        'n_rules_total': len(all_rules),
        'n_rules_after_filter': len(evaluated_rules),
        'averages': averages,
        'top_rules': evaluated_rules[:10],
    }
    with open(run_dir / f'run_{tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {run_dir / f'run_{tag}.json'}")


def main():
    config = Config()
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = LeakDBDataset(config)
    print(f"Dataset ready — {len(dataset)} graphs | {len(dataset.train_indices)} train / {len(dataset.val_indices)} val")

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

    print(f"Loading best checkpoint from {config.checkpoint_path}...")
    checkpoint = torch.load(config.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    if config.visualize_graphs:
        print("\nGenerating graph visualizations...")
        gb_viz = dataset.gb
        viz_dir = config.viz_output_dir

        # 1. Original training graph (full measured_by edges)
        orig_idx = dataset.train_indices[0]
        orig_graph = dataset[orig_idx].to('cpu')
        visualize_graph(orig_graph, gb_viz, title="Original graph (training)",
                        output_path=f"{viz_dir}/graph_original.html")

        # 2. Masked graph (same graph after training masking)
        masker = Masker(gb_viz, config)
        masked_graph = masker.apply(copy.deepcopy(orig_graph))
        visualize_graph(masked_graph, gb_viz, title=f"Masked graph (strategy={config.masking_strategy}, ratio={config.mask_ratio})",
                        output_path=f"{viz_dir}/graph_masked.html")

        # 3. Query graph — built via RuleExtractor.build_query_graph so it matches
        #    exactly what the model sees during rule extraction
        query_st    = 'pressure'
        query_sname = next(sname for (st, sname) in gb_viz.sensor_idx if st == query_st)
        query_bin   = 0
        antecedent  = [(query_st, query_sname, query_bin)]
        extractor_viz = RuleExtractor(model, gb_viz, dataset, config, torch.device('cpu'))
        query_graph = extractor_viz.build_query_graph(antecedent)
        visualize_graph(query_graph, gb_viz,
                        title=f"Query graph — antecedent: ({query_st}, {query_sname}, bin {query_bin})",
                        output_path=f"{viz_dir}/graph_query.html")
        print("Visualization complete")

    model.eval()

    timestamp = datetime.now().isoformat(timespec='seconds')
    run_dir = Path(config.results_dir) / f"run_{timestamp.replace(':', '-')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if config.learn_rules or config.evaluate_rules:
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

        if config.evaluate_rules:
            print("Evaluating GAE rules...")
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

            if config.compute_hop_distance:
                hop_avgs = annotate_hop_distances(evaluated_rules, dataset.gb)
                if hop_avgs:
                    averages['avg_hop_distance_min'] = hop_avgs['avg_min']
                    averages['avg_hop_distance_max'] = hop_avgs['avg_max']
                    print(f"Average hop distance — min: {hop_avgs['avg_min']}  max: {hop_avgs['avg_max']}")
            _save_rules_output(evaluated_rules, averages, rules, run_dir, 'gae', config)

    if config.learn_rules_fp or config.evaluate_rules_fp:
        if config.learn_rules_fp:
            print("Extracting rules via FP-growth...")
            fp_extractor = FPExtractor(dataset, config)
            fp_rules = fp_extractor.extract()
            print(f"FP-growth extraction complete — {len(fp_rules)} candidate rules")
            Path(config.rules_path_fp).parent.mkdir(parents=True, exist_ok=True)
            with open(config.rules_path_fp, 'w') as f:
                json.dump(fp_rules, f)
            print(f"FP rules saved to {config.rules_path_fp}")
        else:
            rules_path_fp = Path(config.rules_path_fp)
            if not rules_path_fp.exists():
                raise FileNotFoundError(
                    f"learn_rules_fp=False but no FP rules file found at '{config.rules_path_fp}'. "
                    "Set learn_rules_fp=True to extract and save rules first."
                )
            print(f"Loading FP rules from {config.rules_path_fp}...")
            with open(rules_path_fp) as f:
                raw = json.load(f)
            fp_rules = [
                {
                    'antecedent': [tuple(item) for item in rule['antecedent']],
                    'consequent': tuple(rule['consequent']),
                }
                for rule in raw
            ]
            print(f"Loaded {len(fp_rules)} FP rules")

        for rule in fp_rules[:5]:
            print(f"  {rule['antecedent']} -> {rule['consequent']}")

        if config.evaluate_rules_fp:
            print("Evaluating FP-growth rules...")
            fp_evaluator = RuleEvaluator(dataset, config, device)
            evaluated_fp_rules, fp_averages = fp_evaluator.evaluate(
                fp_rules,
                min_support=config.min_support_fp,
                min_confidence=config.min_confidence_fp,
            )
            if config.filter_rules:
                print(f"FP evaluation complete — {len(evaluated_fp_rules)} rules pass support>={config.min_support} and confidence>={config.min_confidence}")
            else:
                print(f"FP evaluation complete — {len(evaluated_fp_rules)} rules (no filtering applied)")
            print(f"FP averages over all {len(fp_rules)} rules: " + " | ".join(f"{k}={v:.4f}" for k, v in fp_averages.items()))
            for rule in evaluated_fp_rules[:5]:
                print(f"  support={rule['support']:.3f} conf={rule['confidence']:.3f} lift={rule['lift']:.3f} zhang={rule['zhang']:.3f} | {rule['antecedent']} -> {rule['consequent']}")

            if config.compute_hop_distance:
                hop_avgs = annotate_hop_distances(evaluated_fp_rules, dataset.gb)
                if hop_avgs:
                    fp_averages['avg_hop_distance_min'] = hop_avgs['avg_min']
                    fp_averages['avg_hop_distance_max'] = hop_avgs['avg_max']
                    print(f"Average hop distance — min: {hop_avgs['avg_min']}  max: {hop_avgs['avg_max']}")
            _save_rules_output(evaluated_fp_rules, fp_averages, fp_rules, run_dir, 'fp', config)


if __name__ == "__main__":
    main()
