"""Step 1 of the Aerial baseline: slice the categorical sensor data into a stable file for Aerial to train on.

Runs in the project environment. Usage, from the project root:
    python -m baselines.aerial.export [--split all|train]

Unlike RuleNaps, Aerial runs in-process (no separate environment/subprocess), so this step does not need to
serialize to a custom text format or hand out an index->item mapping: the categorical columns are already named
"sensor_type;sensor_name" and valued with the discretized bin index, which is exactly what Aerial expects (it
one-hot encodes internally). This step only decouples the baseline from dataset_categorical.parquet, so later
steps don't break when the dataset is reprocessed.

Writes to <processed_data_dir>/aerial/:
    transactions_<split>.parquet   one column per sensor, values are discretized bin indices (0..n_bins-1)
"""
import argparse
import os

import pandas as pd

from src.utils.config import Config

def get_output_dir(config: Config) -> str:
    # lives inside the processed dir on purpose: the export is derived from dataset_categorical.parquet and
    # becomes stale when it is reprocessed
    return os.path.join(config.processed_data_dir, 'aerial')


def load_transactions(config: Config, split: str = 'all') -> pd.DataFrame:
    """Load the categorical transaction DataFrame (one column per sensor, values = discretized bin index)."""
    df = pd.read_parquet(os.path.join(config.processed_data_dir, 'dataset_categorical.parquet'))
    if split != 'all':
        df = df[df['split'] == split]
    # Item columns always have a "signal;raw_id" shape (contain a semicolon); metadata
    # columns (scenario/scenario_id/time_id/split/label, which differ per dataset) never
    # do -- this identifies item columns without hardcoding which metadata columns a
    # given dataset happens to have.
    item_cols = [c for c in df.columns if ';' in c]
    return df[item_cols]


def export(config: Config, split: str = 'all') -> dict:
    df = load_transactions(config, split)

    out_dir = get_output_dir(config)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'transactions_{split}.parquet')
    df.to_parquet(out_path)

    bins_per_sensor = df.nunique()
    summary = {
        'split': split,
        'n_transactions': len(df),
        'n_sensors': df.shape[1],
        'bins_per_sensor': (int(bins_per_sensor.min()), int(bins_per_sensor.max())),
        'out_path': out_path,
    }
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export categorical transactions for Aerial')
    parser.add_argument('--split', default='all', choices=['train', 'val', 'all'],
                        help='which rows to export (default: all)')
    args = parser.parse_args()

    summary = export(Config(), split=args.split)
    for k, v in summary.items():
        print(f"{k:<16} {v}")
