"""Standalone FP-growth baseline: mine rules and evaluate them with the project's RuleEvaluator.

Runs in the project environment. Usage, from the project root:
    python -m baselines.fp_growth.run_fp_growth [--out-dir results/<dataset>/fp_growth/run_<timestamp>] [--no-hop-distance]

Unlike the Aerial/RuleNaps baselines, FP-growth needs no export/train/convert steps and has no tunable
hyperparameters to search (its thresholds mirror the GAE's own, fixed everywhere -- see docs/experiment_plan.md),
so there is no optimize.py and no separate experiment.py: extraction and evaluation happen directly here, in the
one script meant to be run on its own, the same role train_aerial.py/run.py play for the other baselines.
run_experiment() still returns (rules, averages, summary) -- the same shape as Aerial/RuleNaps's run_experiment() --
so a future script could call every baseline uniformly.

Evaluates on the train split only (split='train'), matching every other method, and evaluates with evaluate(rules)
(no min_support/min_confidence override) like every other baseline: FP-growth's mined rules already satisfy
min_support_fp/min_confidence_fp exactly by construction, so passing those back into evaluate() as the filter
threshold would make config.filter_rules a no-op for this method while still filtering GAE/Aerial/RuleNaps against
config.min_support/config.min_confidence.

Writes to --out-dir:
    rules_fp_growth.csv   one row per rule with its metrics, same columns as rules_gae.csv / rules_aerial.csv / rules_rulenaps.csv
    run_fp_growth.json    timings, averages, rule counts and the top 10 rules
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from baselines.fp_growth.fp_extractor import FPExtractor
from src.rules.evaluator import RuleEvaluator, annotate_hop_distances
from src.utils.config import Config
from src.utils.dataset_factory import build_dataset

RULE_METRIC_COLS = ('support', 'support_ant', 'confidence', 'lift', 'zhang', 'confidence_zhang', 'hop_distance_min', 'hop_distance_max')


def run_experiment(config: Config, out_dir: str = None, dataset=None, device: torch.device = None,
                    compute_hop_distance: bool = None, save: bool = True) -> tuple[list[dict], dict, dict]:
    """Mine and evaluate FP-growth rules.

    dataset and device can be passed in so the dataset is built only once when many runs are evaluated.

    Returns:
        rules:    mined rules, pre-evaluation [{antecedent, consequent}, ...]
        averages: metric averages from RuleEvaluator (confidence, zhang, coverage, ...)
        summary:  mining params and timings (runtime_mining_s, runtime_rule_generation_s, runtime_filtering_s,
                  runtime_total_s)
    """
    if device is None:
        device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if dataset is None:
        dataset = build_dataset(config)
    if compute_hop_distance is None:
        compute_hop_distance = config.compute_hop_distance
    if out_dir is None:
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        out_dir = Path(config.results_dir) / 'fp_growth' / f'run_{timestamp}'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fp_growth = FPExtractor(dataset, config)
    rules, timings = fp_growth.extract()
    evaluator = RuleEvaluator(dataset, config, device, split='train')
    evaluated_rules, averages = evaluator.evaluate(rules)
    if compute_hop_distance and evaluated_rules:
        hop_avgs = annotate_hop_distances(evaluated_rules, dataset.gb)
        if hop_avgs:
            averages['avg_hop_distance_min'] = hop_avgs['avg_min']
            averages['avg_hop_distance_max'] = hop_avgs['avg_max']

    summary = {
        'min_support_fp': config.min_support_fp,
        'min_confidence_fp': config.min_confidence_fp,
        'max_antecedent_size': config.max_antecedent_size,
        'n_rules': len(rules),
        **timings,
    }

    if save:
        rows = [{'antecedent': str(r['antecedent']), 'consequent': str(r['consequent']),
                 **{k: r[k] for k in RULE_METRIC_COLS if k in r}} for r in evaluated_rules]
        pd.DataFrame(rows).to_csv(out_dir / 'rules_fp_growth.csv', index=False)

        results = {
            'tag': 'fp_growth',
            **summary,
            'filter_rules': config.filter_rules,
            'n_rules_after_filter': len(evaluated_rules),
            'averages': averages,
            'top_rules': evaluated_rules[:10],
        }
        with open(out_dir / 'run_fp_growth.json', 'w') as f:
            json.dump(results, f, indent=2)

    return rules, averages, summary


if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='Mine and evaluate FP-growth rules with RuleEvaluator')
    parser.add_argument('--min-support', type=float, default=config.min_support_fp,
                        help=f'default: config.antecedent_threshold ({config.antecedent_threshold}), the same '
                             'antecedent-probability threshold GAE/Aerial use (see config.min_support_fp)')
    parser.add_argument('--min-confidence', type=float, default=config.min_confidence_fp,
                        help=f'default: config.consequent_threshold ({config.consequent_threshold}), the same '
                             'consequent-probability threshold GAE/Aerial use (see config.min_confidence_fp)')
    parser.add_argument('--out-dir', default=None, help='default: <results_dir>/fp_growth/run_<timestamp>')
    parser.add_argument('--no-hop-distance', action='store_true', help='skip the hop distance computation')
    args = parser.parse_args()

    config.min_support_fp = args.min_support
    config.min_confidence_fp = args.min_confidence

    _, averages, summary = run_experiment(config, out_dir=args.out_dir,
                                          compute_hop_distance=False if args.no_hop_distance else None)
    for k, v in summary.items():
        print(f"{k:<24} {v}")
    for k, v in averages.items():
        print(f"{k:<24} {v}")
