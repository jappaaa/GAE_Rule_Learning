"""Data-quality exploration for the LBNL Building 59 dataset: duplicate-
timestamp checks, missing-value checks, and (later) outlier checks, all
parameterized over an arbitrary candidate time window so different windows
can be compared to find the cleanest one. Meant to be mirrored for the other
datasets' loaders later.

Order matters: duplicates are checked BEFORE missing-value rates are computed,
not after. resample_grid() doesn't drop duplicate timestamps -- it bins by
calendar time rather than exact timestamp matching, so it absorbs same-
timestamp duplicates (and off-grid jitter) into the average on its own,
weighting a duplicated moment by how many raw rows it actually had. That's
only correct if duplicate copies are genuinely separate readings rather than
a copied/corrupted repeat -- which is why duplicate_report() below checks
whether duplicate copies of the same timestamp actually agree in value, and
why remove_backward_jump() explicitly drops the one confirmed exact-copy case
(uft_hw_valve.csv's appended 2020) before any averaging happens (see
docs/experiment_plan.md).
"""
import argparse
import os

import numpy as np
import pandas as pd

from src.data.lbnl_b59.loader import LBNLLoader, WINDOW_START, WINDOW_END, RESAMPLE_FREQ
from src.data.lbnl_b59.preprocessing import clip_window, remove_backward_jump, resample_grid, pad_to_window

# Default span for every report in this module: the whole dataset, not the
# committed WINDOW_START/WINDOW_END -- that window is one candidate among
# several this module exists to compare, so defaulting to it silently would
# just reproduce the "only looks clean/dirty for the window we already
# picked" confusion this module is meant to avoid. Pass start/end explicitly
# to scope a report to one particular window instead.
FULL_SPAN_START = pd.Timestamp('2018-01-01')
FULL_SPAN_END = pd.Timestamp('2020-12-31')


def _all_csv_files(loader: LBNLLoader) -> list:
    """Every raw CSV in csv_dir, deliberately NOT filtered by EXCLUDED_CSV_FILES.

    EXCLUDED_CSV_FILES records which files didn't overlap the *already-chosen*
    committed window (docs/experiment_plan.md) -- using it here would be
    circular, since the point of this module is to help choose a window in
    the first place (including re-considering files like zone_co2.csv/
    ashp_*.csv that are only excluded because of that earlier choice).
    """
    return sorted(f for f in os.listdir(loader.csv_dir) if f.endswith('.csv'))


def cols_by_file_all(loader: LBNLLoader) -> dict:
    """{file: [value columns]} for every raw CSV, for exploration across the
    whole dataset. See LBNLLoader.cols_by_file_from_attachments() for the
    alternative (just the files/columns actually attached to the KG for the
    committed window) once a window has been decided and this is no longer
    needed.
    """
    cols_by_file = {}
    for fname in _all_csv_files(loader):
        cols = pd.read_csv(os.path.join(loader.csv_dir, fname), nrows=0).columns.tolist()
        date_col = cols[0]
        cols_by_file[fname] = [c for c in cols if c != date_col and 'date' not in c.lower()
                                and not c.startswith('Unnamed')]
    return cols_by_file


def duplicate_report(raw_by_file: dict, start: pd.Timestamp = FULL_SPAN_START,
                      end: pd.Timestamp = FULL_SPAN_END) -> pd.DataFrame:
    """One row per file in raw_by_file (see LBNLLoader.load_all_raw()):
    duplicate-timestamp stats within [start, end], on the RAW (pre-dedup)
    data -- i.e. exactly what preprocessing.resample_grid() has to absorb
    for that file/window.

    Also checks, for timestamps with more than one row, whether the copies
    actually agree: sorts by date (so same-timestamp rows become adjacent)
    and compares each row to the previous one when they share a timestamp --
    vectorized, not a per-group loop, since the latter is too slow at this
    scale (confirmed: >120s on a single file's full 3-year span).
    """
    rows = []
    for fname, raw in raw_by_file.items():
        df = raw[(raw.index >= start) & (raw.index <= end)]    # do not use the clipping function from preprocessing.py because it needs sorted data, sorting here would break n_backward_jumps logic
        value_cols = df.columns.tolist()

        # pd.Series(df.index), not df.index.to_series() -- the latter would reuse
        # the (possibly duplicate-valued) DatetimeIndex as this Series's own index,
        # breaking the positional boolean-mask alignment used below
        t = pd.Series(df.index)
        n_backward_jumps = int((t.diff() < pd.Timedelta(0)).sum())
        n_dup_rows = int(t.duplicated().sum())  # count duplicate timestamps, 2 occurences of a timestamp are counted as 1 duplicate, 3 occurences as 2, etc.
        n_unique_timestamps = int(t.nunique())
        dup_mask = t.duplicated(keep=False)
        n_dup_timestamps = int(t[dup_mask].nunique())

        n_pairs, n_mismatched = 0, 0
        if n_dup_timestamps:
            # Restrict to the duplicated rows before sorting/comparing. dup_mask.to_numpy()
            # forces pure positional indexing, since df's own (DatetimeIndex) labels can
            # themselves contain duplicates, making label-based alignment ambiguous.
            sorted_df = df[dup_mask.to_numpy()].sort_index(kind='stable')
            vals = sorted_df[value_cols].to_numpy(dtype=float)
            dates = sorted_df.index.to_numpy()
            # vectorized to improve time complexity
            same_as_prev = dates[1:] == dates[:-1]
            row_equal = np.all((vals[1:] == vals[:-1]) | (np.isnan(vals[1:]) & np.isnan(vals[:-1])), axis=1)
            # number of duplicate timestamp pairs and number of these pairs where values differ
            n_pairs = int(same_as_prev.sum())
            n_mismatched = int((same_as_prev & ~row_equal).sum())

        rows.append({
            "file": fname,
            "n_rows": len(df),
            "n_unique_timestamps": n_unique_timestamps,
            "n_backward_jumps": n_backward_jumps,
            "n_dup_rows": n_dup_rows,
            "n_dup_timestamps": n_dup_timestamps,
            # fraction of this file's DISTINCT timestamps that have >1 row --
            # not n_dup_timestamps / n_rows, which divides by a count that
            # already includes the duplicate copies and so can never reach 1.0
            # even when every single timestamp is duplicated.
            "dup_timestamp_rate": n_dup_timestamps / n_unique_timestamps if n_unique_timestamps else 0.0,
            "n_dup_pairs_checked": n_pairs,
            "n_dup_pairs_mismatched": n_mismatched,
            "dup_mismatch_rate": n_mismatched / n_pairs if n_pairs else 0.0,
        })

    return pd.DataFrame(rows).sort_values("dup_timestamp_rate", ascending=False)


def missing_value_report(raw_by_file: dict, cols_by_file: dict, start: pd.Timestamp = FULL_SPAN_START,
                          end: pd.Timestamp = FULL_SPAN_END) -> pd.DataFrame:
    """One row per (file, column): missing rate on the raw logged rows within
    [start, end] (defaulting to the whole dataset) -- i.e. of the rows that
    were actually logged, how often this column's cell is blank -- missing
    rate after resampling to the common RESAMPLE_FREQ grid (mean), and the
    longest contiguous gap (in days) that survives resampling.

    cols_by_file: {file: [columns]} -- build with cols_by_file_all() to scan
    every raw CSV, or LBNLLoader.cols_by_file_from_attachments() to restrict
    to the committed/kept set.

    native_missing_rate and resampled_missing_rate_unpadded are computed over
    the SAME scope (clipped to [start, end], never padded) -- comparing them
    isolates what resampling itself fixes, without the scope mismatch a
    padded-vs-unpadded comparison would introduce (a file that doesn't cover
    [start, end] at all would otherwise look like resampling helped a lot,
    when really the apparent improvement is just non-coverage becoming
    visible as missing on one side and not the other).

    resampled_missing_rate IS padded to the full [start, end] grid (like
    longest_gap_days's underlying series) -- this is the one period_summary()
    and the monthly heatmap should keep using, since a file not covering the
    window at all should count as fully missing there, not be silently
    excluded from the average.
    """
    rows = []
    for fname, cols in sorted(cols_by_file.items()):
        clipped_raw = clip_window(remove_backward_jump(raw_by_file[fname]), start, end)
        resampled_unpadded = resample_grid(clipped_raw)
        resampled_padded = pad_to_window(resampled_unpadded, start, end)

        for col in cols:
            native_missing = clipped_raw[col].isna().mean()
            resampled_unpadded_missing = resampled_unpadded[col].isna().mean()  # this can directly be compared to the native missing rate to observe how much resampling resolves missing values
            padded_series = resampled_padded[col]
            resampled_missing = padded_series.isna().mean()

            isnan = padded_series.isna()
            blocks = (isnan != isnan.shift()).cumsum()  # the isnan != isnan.shift() checks for changes between i and i + 1 in the missing value series, cumsum only increments when a change happens, i.e., isnan != isnan.shift() == True, to assign block ids
            gap_lengths = isnan.groupby(blocks).sum()   # number of timestamps per block
            gap_lengths = gap_lengths[gap_lengths > 0]  # only include blocks that contained nan values
            longest_gap_days = gap_lengths.max() * pd.Timedelta(RESAMPLE_FREQ) / pd.Timedelta(days=1) if len(gap_lengths) else 0.0

            rows.append({
                "file": fname,
                "column": col,
                "native_missing_rate": native_missing,
                "resampled_missing_rate_unpadded": resampled_unpadded_missing,
                "resampled_missing_rate": resampled_missing,
                "longest_gap_days": longest_gap_days,
            })

    return pd.DataFrame(rows).sort_values("resampled_missing_rate", ascending=False)    # return the missing values report as a pandas dataframe, sorted on descending resampled missing rate


def monthly_missing_value_report(raw_by_file: dict, cols_by_file: dict, start: pd.Timestamp = FULL_SPAN_START,
                                  end: pd.Timestamp = FULL_SPAN_END) -> pd.DataFrame:
    """One row per (file, column, month): same native/resampled missing rates
    as missing_value_report() (including the unpadded/padded resampled split
    -- see its docstring), but broken out by calendar month over a wide span,
    so a window can be chosen by eyeballing which months are clean across
    most files rather than testing a handful of hand-picked candidates.

    cols_by_file: see missing_value_report().

    Each file is loaded once over the full [start, end] span (not once per
    month -- that would re-read the CSV 36 times for no reason) and then
    grouped by month.
    """
    rows = []
    for fname, cols in sorted(cols_by_file.items()):
        clipped_raw = clip_window(remove_backward_jump(raw_by_file[fname]), start, end)
        resampled_unpadded = resample_grid(clipped_raw)
        resampled_padded = pad_to_window(resampled_unpadded, start, end)

        native_missing_by_month = clipped_raw.isna().groupby(clipped_raw.index.to_period('M')).mean()
        resampled_unpadded_missing_by_month = resampled_unpadded.isna().groupby(resampled_unpadded.index.to_period('M')).mean()
        resampled_missing_by_month = resampled_padded.isna().groupby(resampled_padded.index.to_period('M')).mean()
        # Neither native_missing_by_month nor resampled_unpadded_missing_by_month is
        # padded, so a month where this file has zero rows at all (e.g. ashp_cw.csv
        # before mid-2020) simply has no entry -- reindex both onto
        # resampled_missing_by_month's month index (always complete, since padding
        # guarantees a row for every grid timestamp) so the lookup below doesn't
        # KeyError, with NaN correctly meaning "no rows logged, rate undefined"
        # rather than 0% or 100%. This is needed for the monthly missing heatmap.
        native_missing_by_month = native_missing_by_month.reindex(resampled_missing_by_month.index)
        resampled_unpadded_missing_by_month = resampled_unpadded_missing_by_month.reindex(resampled_missing_by_month.index)

        for col in cols:
            for month in resampled_missing_by_month.index:
                rows.append({
                    "file": fname,
                    "column": col,
                    "month": str(month),
                    "native_missing_rate": native_missing_by_month.loc[month, col],
                    "resampled_missing_rate_unpadded": resampled_unpadded_missing_by_month.loc[month, col],
                    "resampled_missing_rate": resampled_missing_by_month.loc[month, col],
                })

    return pd.DataFrame(rows)


def monthly_duplicate_report(raw_by_file: dict, start: pd.Timestamp = FULL_SPAN_START,
                              end: pd.Timestamp = FULL_SPAN_END) -> pd.DataFrame:
    """One row per (file, month): duplicate-timestamp rate within that month,
    on the raw (pre-dedup) data -- same idea as monthly_missing_value_report,
    for spotting e.g. that uft_hw_valve.csv's duplication is confined to 2020
    at month-level granularity.

    Uses preprocessing.clip_window() (sorts, then slices) -- unlike
    duplicate_report(), nothing here is order-sensitive (value_counts() per
    month doesn't care about row order), so there's no reason to keep the
    original row order here.
    """
    rows = []
    for fname, raw in raw_by_file.items():
        # pd.Series(df.index), not df.index.to_series() -- see duplicate_report()
        t = pd.Series(clip_window(raw, start, end).index)

        month = t.dt.to_period('M')
        for m, month_dates in t.groupby(month):
            counts = month_dates.value_counts()
            rows.append({
                "file": fname,
                "month": str(m),
                "n_rows": len(month_dates),
                "dup_timestamp_rate": (counts > 1).mean(),
            })

    return pd.DataFrame(rows)


# TODO: outlier_report() -- not scoped yet (bounds/rate-of-change checks a la
# the Trust-Aware Virtual Sensing clean_data.py reviewed in docs/experiment_plan.md).
# Add once the missing-value/duplicate picture across candidate windows is settled.


def period_summary(raw_by_file: dict, cols_by_file: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Single-row cleanliness summary for one candidate window (duplicates
    computed first, missing values on top), meant for comparing windows
    against each other -- see compare_periods().
    """
    dup_df = duplicate_report(raw_by_file, start=start, end=end)
    miss_df = missing_value_report(raw_by_file, cols_by_file, start=start, end=end)
    return {
        "start": start.date(),
        "end": end.date(),
        "n_days": (end - start).days,
        "any_backward_jump": bool((dup_df["n_backward_jumps"] > 0).any()),
        "mean_dup_timestamp_rate": dup_df["dup_timestamp_rate"].mean(),
        "max_dup_timestamp_rate": dup_df["dup_timestamp_rate"].max(),
        "mean_resampled_missing_rate": miss_df["resampled_missing_rate"].mean(),
        "max_resampled_missing_rate": miss_df["resampled_missing_rate"].max(),
    }


def compare_periods(raw_by_file: dict, cols_by_file: dict, windows: dict) -> pd.DataFrame:
    """windows: {label: (start, end)}. One row per label, for eyeballing which
    candidate window is cleanest before committing to one.
    """
    rows = []
    for label, (start, end) in windows.items():
        summary = period_summary(raw_by_file, cols_by_file, pd.Timestamp(start), pd.Timestamp(end))
        rows.append({"label": label, **summary})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=pd.Timestamp, default=FULL_SPAN_START,
                         help='window start, defaults to the whole dataset (2018-01-01)')
    parser.add_argument('--end', type=pd.Timestamp, default=FULL_SPAN_END,
                         help='window end, defaults to the whole dataset (2020-12-31)')
    args = parser.parse_args()

    loader = LBNLLoader(os.path.join('data', 'lbnl_b59', 'raw'))
    cols_by_file = cols_by_file_all(loader)
    raw_by_file = loader.load_all_raw(cols_by_file)

    print(f"scanning all {len(cols_by_file)} raw CSV files")

    out_dir = 'results_scratch'
    os.makedirs(out_dir, exist_ok=True)
    suffix = '' if (args.start, args.end) == (FULL_SPAN_START, FULL_SPAN_END) else f"_{args.start.date()}_{args.end.date()}"

    dup_df = duplicate_report(raw_by_file, start=args.start, end=args.end)
    dup_path = os.path.join(out_dir, f'lbnl_b59_duplicate_report{suffix}.csv')
    dup_df.to_csv(dup_path, index=False)
    print(f"wrote {len(dup_df)} rows to {dup_path}")
    print("\n--- duplicate report (all raw files) ---")
    print(dup_df.to_string(index=False))

    miss_df = missing_value_report(raw_by_file, cols_by_file, start=args.start, end=args.end)
    miss_path = os.path.join(out_dir, f'lbnl_b59_missing_value_report{suffix}.csv')
    miss_df.to_csv(miss_path, index=False)
    print(f"\nwrote {len(miss_df)} rows to {miss_path}")
    print("\n--- worst 20 columns by resampled missing rate ---")
    print(miss_df.head(20).to_string(index=False))
    print("\n--- per-file summary (mean resampled missing rate) ---")
    print(miss_df.groupby('file')['resampled_missing_rate'].mean().sort_values(ascending=False).to_string())

    print("\n--- candidate window comparison ---")
    candidate_windows = {
        "full 3-year span": (FULL_SPAN_START, FULL_SPAN_END),
        "committed (2020-05-01 to 2020-12-31)": (WINDOW_START, WINDOW_END),
        "2019-02-20 to 2020-04-01": ('2019-02-20', '2020-04-01'),
        "2019-02-20 to 2019-12-31": ('2019-02-20', '2019-12-31'),
    }
    summary_df = compare_periods(raw_by_file, cols_by_file, candidate_windows)
    summary_path = os.path.join(out_dir, 'lbnl_b59_period_comparison.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"wrote {len(summary_df)} rows to {summary_path}")
    print(summary_df.to_string(index=False))

    print("\n--- monthly missing-value report (file x month, mean resampled missing rate) ---")
    monthly_miss_df = monthly_missing_value_report(raw_by_file, cols_by_file)
    monthly_miss_path = os.path.join(out_dir, 'lbnl_b59_monthly_missing_value_report.csv')
    monthly_miss_df.to_csv(monthly_miss_path, index=False)
    print(f"wrote {len(monthly_miss_df)} rows to {monthly_miss_path}")
    miss_pivot = monthly_miss_df.pivot_table(index='file', columns='month', values='resampled_missing_rate')
    miss_pivot_path = os.path.join(out_dir, 'lbnl_b59_monthly_missing_value_pivot.csv')
    miss_pivot.to_csv(miss_pivot_path)
    print(f"wrote pivot to {miss_pivot_path}")
    print((miss_pivot * 100).round(1).to_string())

    print("\n--- monthly duplicate report (file x month, duplicate-timestamp rate) ---")
    monthly_dup_df = monthly_duplicate_report(raw_by_file)
    monthly_dup_path = os.path.join(out_dir, 'lbnl_b59_monthly_duplicate_report.csv')
    monthly_dup_df.to_csv(monthly_dup_path, index=False)
    print(f"wrote {len(monthly_dup_df)} rows to {monthly_dup_path}")
    dup_pivot = monthly_dup_df.pivot_table(index='file', columns='month', values='dup_timestamp_rate')
    dup_pivot_path = os.path.join(out_dir, 'lbnl_b59_monthly_duplicate_pivot.csv')
    dup_pivot.to_csv(dup_pivot_path)
    print(f"wrote pivot to {dup_pivot_path}")
    print((dup_pivot * 100).round(2).to_string())
