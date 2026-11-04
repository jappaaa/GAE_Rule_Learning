"""Step 2 of the Aerial baseline: train the Autoencoder and extract rules from it.

Runs in the project environment: unlike RuleNaps, Aerial (PyAerial) is a plain pip dependency with no conflicting
Python-version/package requirements, so there is no separate environment or subprocess here, and training and rule
extraction are combined into this single step instead of run.py + train_rulenaps.py.

Usage, from the project root (after export.py):
    python -m baselines.aerial.train_aerial [--split train] [--train-params '{"epochs": 10}']
                                             [--rule-params '{"max_antecedents": 3}']

Writes to --out-dir:
    rules.json         raw output of aerial.rule_extraction.generate_rules ('rules' and 'statistics')
    run_summary.json   params used and run metadata; read by convert.py
"""
import argparse
import json
import os
from datetime import datetime
from timeit import default_timer as timer

import pandas as pd
import torch
from aerial import model, rule_extraction

from baselines.aerial.export import get_output_dir
from src.utils.config import Config

# Aerial's own training defaults (aerial.model.train), used as a starting point until Aerial is tuned on our
# data (there is no authors' reference config to start from, unlike RuleNaps)
DEFAULT_TRAIN_PARAMS = {}


def _default_rule_params(config: Config) -> dict:
    """Default aerial.rule_extraction.generate_rules kwargs, pinned to the GAE pipeline's analogous settings
    for comparability: min_rule_frequency/min_rule_strength are Aerial's names for the antecedent/consequent
    probability thresholds the GAE extractor also thresholds on, and max_antecedents is the same antecedent-size
    cap every other baseline is held to. All three stay directly overridable via rule_params."""
    return {
        'min_rule_frequency': config.antecedent_threshold,
        'min_rule_strength': config.consequent_threshold,
        'max_antecedents': config.max_antecedent_size,
    }


def train(config: Config, split: str = 'train', train_params: dict = None, rule_params: dict = None,
          out_dir: str = None) -> dict:
    """Train an Autoencoder on the exported transactions and extract rules from it, writing both to out_dir."""
    transactions_path = os.path.join(get_output_dir(config), f'transactions_{split}.parquet')
    if not os.path.exists(transactions_path):
        raise FileNotFoundError(f"{transactions_path} not found; run "
                                f"`python -m baselines.aerial.export --split {split}` first")
    transactions = pd.read_parquet(transactions_path)
    # Aerial treats a numeric column as categorical only if it has <= 10 unique values (config.n_bins is 10 by
    # default, so this holds today but would silently break - falling through to Aerial's own re-discretization
    # of our already-discretized bins - if n_bins were ever raised above 10); casting to str makes every column
    # unambiguously categorical regardless of how many bins it has
    transactions = transactions.astype(str)

    # ensure deterministic behaviour
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    merged_train_params = {**DEFAULT_TRAIN_PARAMS, **(train_params or {})}
    merged_rule_params = {**_default_rule_params(config), **(rule_params or {})}

    start = timer()
    autoencoder = model.train(transactions, **merged_train_params)
    training_time = timer() - start
    if autoencoder is None:
        raise RuntimeError("Aerial training failed; see log output above for details")

    result = rule_extraction.generate_rules(autoencoder, **merged_rule_params)
    total_time = timer() - start

    if out_dir is None:
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        out_dir = os.path.join(config.results_dir, 'aerial', f'run_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'rules.json'), 'w') as f:
        json.dump(result, f, indent=2)

    summary = {
        'split': split,
        'train_params': merged_train_params,
        'rule_params': merged_rule_params,
        'seed': config.random_seed,
        'n_rules': len(result['rules']),
        'statistics': result['statistics'],
        'runtime_training_s': round(training_time, 1),
        'runtime_total_s': round(total_time, 1),
    }
    with open(os.path.join(out_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='Train Aerial and extract rules')
    parser.add_argument('--split', default='train', choices=['train', 'val', 'all'],
                        help='split to train on (default: train)')
    parser.add_argument('--train-params', default='{}', help='JSON dict overriding aerial.model.train kwargs')
    parser.add_argument('--rule-params', default='{}',
                        help='JSON dict overriding aerial.rule_extraction.generate_rules kwargs')
    parser.add_argument('--out-dir', default=None, help='default: <results_dir>/aerial/run_<timestamp>')
    args = parser.parse_args()

    summary = train(config, split=args.split, train_params=json.loads(args.train_params),
                    rule_params=json.loads(args.rule_params), out_dir=args.out_dir)
    for k, v in summary.items():
        print(f"{k:<20} {v}")
