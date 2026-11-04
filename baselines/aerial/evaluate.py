"""Step 4 of the Aerial baseline: evaluate the converted Aerial rules with the project's RuleEvaluator.

Runs in the project environment. Usage, from the project root (after running convert.py):
    python -m baselines.aerial.evaluate --run-dir results/aerial/<run> [--no-hop-distance]

Evaluates on the train split only (split='train'), matching train_aerial.py's default training split, so rules
are evaluated on exactly the rows they were mined from (see RuleEvaluator's split argument) — val is reserved for
a held-out signal elsewhere, not mixed into this comparison.

Writes to --run-dir:
    rules_aerial.csv   one row per rule with its metrics, same columns as rules_gae.csv / rules_fp.csv / rules_rulenaps.csv
    run_aerial.json     averages, rule counts, Aerial params and the top 10 rules
"""
import argparse
import json
import os

import pandas as pd
import torch

from src.data.leakdb.dataset import LeakDBDataset
from src.rules.evaluator import RuleEvaluator, annotate_hop_distances
from src.utils.config import Config

RULE_METRIC_COLS = ('support', 'support_ant', 'confidence', 'lift', 'zhang', 'hop_distance_min', 'hop_distance_max')


def load_rules(run_dir: str) -> list[dict]:
    """Load the converted_rules.json written by convert.py; json stores items as lists, the evaluator expects tuples."""
    with open(os.path.join(run_dir, 'converted_rules.json')) as f:
        raw = json.load(f)
    return [{'antecedent': [tuple(i) for i in r['antecedent']], 'consequent': tuple(r['consequent'])} for r in raw]


def evaluate_rules(rules: list[dict], dataset: LeakDBDataset, config: Config, device: torch.device,
                   compute_hop_distance: bool) -> tuple[list[dict], dict]:
    """Evaluate rules and return the evaluated rules (filtered if config.filter_rules) and the metric averages."""
    if not rules:
        return [], {}
    evaluator = RuleEvaluator(dataset, config, device, split='train')
    evaluated_rules, averages = evaluator.evaluate(rules)
    if compute_hop_distance and evaluated_rules:
        hop_avgs = annotate_hop_distances(evaluated_rules, dataset.gb)
        if hop_avgs:
            averages['avg_hop_distance_min'] = hop_avgs['avg_min']
            averages['avg_hop_distance_max'] = hop_avgs['avg_max']
    return evaluated_rules, averages


def evaluate_run(config: Config, run_dir: str, dataset: LeakDBDataset = None, device: torch.device = None,
                 compute_hop_distance: bool = None, save: bool = True) -> dict:
    """Evaluate the rules of one Aerial run directory and return the metric averages.

    dataset and device can be passed in so the dataset is built only once when many runs are evaluated.
    """
    if device is None:
        device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dataset is None:
        dataset = LeakDBDataset(config)
    if compute_hop_distance is None:
        compute_hop_distance = config.compute_hop_distance

    rules = load_rules(run_dir)
    evaluated_rules, averages = evaluate_rules(rules, dataset, config, device, compute_hop_distance)

    if save:
        rows = [{'antecedent': str(r['antecedent']), 'consequent': str(r['consequent']),
                 **{k: r[k] for k in RULE_METRIC_COLS if k in r}} for r in evaluated_rules]
        pd.DataFrame(rows).to_csv(os.path.join(run_dir, 'rules_aerial.csv'), index=False)

        with open(os.path.join(run_dir, 'run_summary.json')) as f:
            summary = json.load(f)
        results = {
            'tag': 'aerial',
            'aerial_train_params': summary['train_params'],
            'aerial_rule_params': summary['rule_params'],
            'filter_rules': config.filter_rules,
            'n_rules_total': len(rules),
            'n_rules_after_filter': len(evaluated_rules),
            'averages': averages,
            'top_rules': evaluated_rules[:10],
        }
        with open(os.path.join(run_dir, 'run_aerial.json'), 'w') as f:
            json.dump(results, f, indent=2)

    return averages


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate converted Aerial rules with RuleEvaluator')
    parser.add_argument('--run-dir', required=True, help='Aerial run directory containing converted_rules.json')
    parser.add_argument('--no-hop-distance', action='store_true', help='skip the hop distance computation')
    args = parser.parse_args()

    averages = evaluate_run(Config(), args.run_dir, compute_hop_distance=False if args.no_hop_distance else None)
    for k, v in averages.items():
        print(f"{k:<24} {v}")
