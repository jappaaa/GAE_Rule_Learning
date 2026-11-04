"""Step 2 of the RuleNaps baseline (project side): launch RuleNaps training in its own environment.

Runs in the project environment. RuleNaps needs Python 3.9 and shares package names (data, model, utils) with the
project, so it is never imported here; train_rulenaps.py is executed with the RuleNaps interpreter instead.

Usage, from the project root (after running export.py):
    python -m baselines.rulenaps_adapter.run [--split all] [--params '{"epochs": 2}']
"""
import argparse
import json
import math
import os
import subprocess
from datetime import datetime
from pathlib import Path

from baselines.rulenaps_adapter.export import get_output_dir
from src.utils.config import Config

ADAPTER_DIR = Path(__file__).resolve().parent
RULENAPS_DIR = ADAPTER_DIR.parent / 'rulenaps'
RULENAPS_PYTHON = RULENAPS_DIR / '.venv' / 'Scripts' / 'python.exe'  # Windows venv layout
TRAIN_SCRIPT = ADAPTER_DIR / 'train_rulenaps.py'

# Hyperparameters applied on top of RuleNaps' BASE_PARAMETERS. These are the authors' tuned values for synthetic
# experiment 1 (confidence 0.8), used as a starting point until RuleNaps is tuned on our data.
DEFAULT_PARAMS = {
    'horizontal_weight_decay_head': 1.51,
    'vertical_weight_decay_head': 10.37,
    'learning_rate': 0.00044,
    'train_batch_size': 16,
    'epochs': 138,
}


def run(config: Config, split: str = 'all', params: dict = None, out_dir: str = None, threads: int = 16) -> dict:
    """Train RuleNaps on the exported transactions and return the run summary written by train_rulenaps.py.

    With split='all', the .dat holds all train rows followed by all val rows (LeakDBDataset assigns the split by
    position after shuffling), and RuleNaps cuts it positionally at train_set_size. Setting train_set_size to
    config.train_ratio therefore makes RuleNaps train on our train split and compute its validation loss on our
    val split (up to one row: RuleNaps cuts at ceil(ratio * n), LeakDBDataset at int(ratio * n)).
    """
    dat_path = Path(get_output_dir(config)) / f'transactions_{split}.dat'
    if not dat_path.exists():
        raise FileNotFoundError(f"{dat_path} not found; run `python -m baselines.rulenaps_adapter.export --split {split}` first")
    if not RULENAPS_PYTHON.exists():
        raise FileNotFoundError(f"RuleNaps environment not found at {RULENAPS_PYTHON}; see baselines/rulenaps/SOURCE.md for instructions to create the environment.")

    split_params = {}
    if split == 'all':
        # config.train_ratio can be changed without reprocessing the data, so check it against the stored split
        with open(Path(get_output_dir(config)) / 'split.json') as f:
            stored = json.load(f)
        rulenaps_cut = math.ceil(config.train_ratio * stored['n_transactions'])
        if abs(rulenaps_cut - stored['n_train']) > 1:
            raise ValueError(f"config.train_ratio={config.train_ratio} cuts at row {rulenaps_cut}, but the exported data "
                             f"has {stored['n_train']} train rows; reprocess the dataset and rerun export.py")
        split_params = {'train_set_size': config.train_ratio, 'validation_set_size': round(1 - config.train_ratio, 10)}
    merged_params = {**DEFAULT_PARAMS, **split_params, **(params or {})}
    if out_dir is None:
        timestamp = datetime.now().isoformat(timespec='seconds').replace(':', '-')
        out_dir = Path(config.results_dir) / 'rulenaps' / f'run_{timestamp}'
    out_dir = Path(out_dir).resolve()

    cmd = [
        str(RULENAPS_PYTHON), str(TRAIN_SCRIPT),
        '--data', str(dat_path.resolve()),
        '--out-dir', str(out_dir),
        '--params', json.dumps(merged_params),
        '--seed', str(config.random_seed),
        '--threads', str(threads),  # torch intra-op CPU threads; 16 matches the authors' setting
    ]
    # cwd is the RuleNaps src dir because some RuleNaps modules use paths relative to it
    subprocess.run(cmd, cwd=RULENAPS_DIR / 'src', check=True)

    with open(out_dir / 'run_summary.json') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RuleNaps on the exported transactions')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'all'],
                        help="default: all, so RuleNaps' train/validation cut matches config.train_ratio")
    parser.add_argument('--params', default='{}', help='JSON dict overriding DEFAULT_PARAMS, e.g. \'{"epochs": 2}\'')
    parser.add_argument('--out-dir', default=None, help='default: <results_dir>/rulenaps/run_<timestamp>')
    parser.add_argument('--threads', type=int, default=16, help='torch CPU threads (default: 16, as the authors)')
    args = parser.parse_args()

    run(Config(), split=args.split, params=json.loads(args.params), out_dir=args.out_dir, threads=args.threads)
