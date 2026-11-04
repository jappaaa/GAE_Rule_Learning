"""Step 1 of the RuleNaps baseline: export the transaction data to RuleNaps' .dat input format.

Runs in the project environment (not the RuleNaps one). Usage, from the project root:
    python -m baselines.rulenaps_adapter.export [--split all|train]

Writes to <processed_data_dir>/rulenaps/:
    transactions_<split>.dat  one line per transaction, space-separated indices of the active items
    items.json                index -> item mapping, needed to translate RuleNaps rules back to (st, sname, bin)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from src.utils.config import Config

NON_ITEM_COLS = ('scenario', 'split', 'label')


def get_output_dir(config: Config) -> str:
    # lives inside the processed dir on purpose: the export is derived from dataset.parquet and becomes stale when it is reprocessed
    return os.path.join(config.processed_data_dir, 'rulenaps')


def load_transactions(config: Config) -> tuple[pd.DataFrame, list[tuple[str, str, int]]]:
    """Load the one-hot transaction DataFrame and the item list, in the same column order as LeakDBDataset.items."""
    df = pd.read_parquet(os.path.join(config.processed_data_dir, 'dataset.parquet'))
    item_cols = [c for c in df.columns if c not in NON_ITEM_COLS]
    items = [(st, sname, int(b)) for st, sname, b in (c.split(';') for c in item_cols)]
    return df, items


def write_dat(item_matrix: np.ndarray, path: str):
    """Write a boolean (n_transactions, n_items) matrix in .dat format.

    Item index i in the file corresponds to column i of the matrix, i.e. to items[i]. Every line, including the
    last, must end with a newline: RuleNaps' reader strips the final character of each line unconditionally.
    RuleNaps' reader also hangs on empty lines, so empty transactions are rejected here.
    """
    active_counts = item_matrix.sum(axis=1)
    if (active_counts == 0).any():
        raise ValueError(f"{int((active_counts == 0).sum())} transactions have no active items; RuleNaps cannot read empty lines")

    with open(path, 'w') as f:
        for row in item_matrix:
            f.write(' '.join(map(str, np.flatnonzero(row))) + '\n')


def export(config: Config, split: str = 'all') -> dict:
    df, items = load_transactions(config)
    if split == 'all':
        # RuleNaps splits the .dat positionally, so run.py relies on all train rows preceding all val rows
        is_train = (df['split'] == 'train').to_numpy()
        n_train = int(is_train.sum())
        if not is_train[:n_train].all():
            raise ValueError("dataset.parquet rows are not ordered train-then-val; RuleNaps' positional split would mix them")
    else:
        df = df[df['split'] == split]
    item_cols = [c for c in df.columns if c not in NON_ITEM_COLS]
    item_matrix = df[item_cols].to_numpy(dtype=bool)

    out_dir = get_output_dir(config)
    os.makedirs(out_dir, exist_ok=True)
    dat_path = os.path.join(out_dir, f'transactions_{split}.dat')
    items_path = os.path.join(out_dir, 'items.json')

    write_dat(item_matrix, dat_path)
    with open(items_path, 'w') as f:
        json.dump(items, f)
    if split == 'all':
        # read by run.py to check that config.train_ratio still matches the split stored in dataset.parquet
        with open(os.path.join(out_dir, 'split.json'), 'w') as f:
            json.dump({'n_transactions': int(item_matrix.shape[0]), 'n_train': n_train}, f)

    # RuleNaps infers the number of columns as (highest index present + 1), so items that never occur at the end of the
    # item list are invisible to it. This is harmless because indices are preserved, but it is reported for transparency.
    active_items = np.flatnonzero(item_matrix.any(axis=0))
    summary = {
        'split': split,
        'n_transactions': int(item_matrix.shape[0]),
        'n_items': len(items),
        'n_items_occurring': int(len(active_items)),
        'n_columns_seen_by_rulenaps': int(active_items.max() + 1),
        'items_per_transaction': (int(item_matrix.sum(axis=1).min()), int(item_matrix.sum(axis=1).max())),
        'dat_path': dat_path,
        'items_path': items_path,
    }
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export transactions to RuleNaps .dat format')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'all'],
                        help="which rows to export (default: all; run.py then cuts train/validation at config.train_ratio)")
    args = parser.parse_args()

    summary = export(Config(), split=args.split)
    for k, v in summary.items():
        print(f"{k:<28} {v}")
