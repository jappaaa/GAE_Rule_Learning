"""Processing stage functions for LBNL Building 59 per-file timeseries,
composed from load_all_raw()'s raw per-file data. Each function does exactly
one thing, so callers can combine them differently -- e.g. skip
pad_to_window() on both sides to compare native vs. resampled missingness
over the same, unpadded scope.
"""
import numpy as np
import pandas as pd

from src.data.lbnl_b59.loader import WINDOW_START, WINDOW_END, RESAMPLE_FREQ, EXCLUDED_CSV_FILES


def remove_backward_jump(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every row from the first backward jump in df's index (by original
    row position, not label) onward.

    Confirmed for uft_hw_valve.csv, whose entire 2020 is appended a second
    time after 2020-12-31 -- not a second genuine reading, a literal copy
    (every one of 2020's 527,040 unique timestamps matches exactly, across
    all 44 columns, between the two passes). Keeping both passes would
    double-count that single real measurement in any resampled average, so
    rows from the first backward jump onward are dropped before anything
    else. Must run on df as returned by load_all_raw() (original on-disk row
    order, not yet sorted) -- it's a no-op for every other file (0 backward
    jumps), so it's always safe to apply generically rather than special-
    casing this file by name.
    """
    is_backward_jump = pd.Series(df.index).diff() < pd.Timedelta(0)
    if is_backward_jump.any():
        df = df.iloc[:np.flatnonzero(is_backward_jump)[0]]
    return df


def clip_window(df: pd.DataFrame, start: pd.Timestamp = WINDOW_START,
                 end: pd.Timestamp = WINDOW_END) -> pd.DataFrame:
    """Sort df's index and restrict to [start, end]. Must run after
    remove_backward_jump() (which needs the original, pre-sort row order) --
    sorting first would let the corrupted/appended rows that fixes end up
    anywhere relative to the genuine ones, corrupting the logic of 
    remove_backward_jump().
    """
    return df.sort_index().loc[start:end]


def resample_grid(df: pd.DataFrame, freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """Resample df onto the common freq grid (mean).

    Bins by calendar-time range-containment, not exact index matching, so
    this absorbs same-timestamp duplicates and off-grid/jittered raw
    timestamps into the average on its own -- weighting a duplicated moment
    by how many raw rows it actually had, which is correct (not a bias) when
    duplicate copies are genuine distinct measurements rather than a
    copied/corrupted repeat (see docs/experiment_plan.md's dup_mismatch_rate
    check). Only bins across df's own min/max timestamp, not any externally
    desired window -- pad_to_window() handles surfacing a file's
    non-coverage of a wider window as missing.
    """
    return df.resample(freq).mean()


def pad_to_window(df: pd.DataFrame, start: pd.Timestamp = WINDOW_START, end: pd.Timestamp = WINDOW_END,
                   freq: str = RESAMPLE_FREQ) -> pd.DataFrame:
    """Reindex df onto the full [start, end] grid at freq.

    resample_grid() only bins across the data's own min/max timestamp, not
    any externally desired [start, end] -- for a file with no coverage at
    all near either edge of a window (e.g. ashp_cw.csv, whose data only
    starts mid-2020), that would silently shrink the result to just its
    covered sub-range instead of surfacing the uncovered part as missing.
    Only safe to call on data already on the freq grid (e.g. resample_grid()'s
    output), not on raw/native data at its own per-file rate -- reindexing
    that directly would read as missing everywhere it's merely off-grid.
    """
    full_index = pd.date_range(start, end, freq=freq)
    return df.reindex(full_index)


def build_combined_frame(raw_by_file: dict, cols_by_file: dict, start: pd.Timestamp = WINDOW_START,
                          end: pd.Timestamp = WINDOW_END, excluded_files: set = EXCLUDED_CSV_FILES) -> pd.DataFrame:
    """Every file in cols_by_file not in excluded_files, run through the full
    remove_backward_jump -> clip_window -> resample_grid -> pad_to_window
    pipeline and concatenated column-wise into one wide frame: one row per
    RESAMPLE_FREQ timestamp over [start, end], one column per kept sensor
    across every kept file.

    A plain axis=1 concat is safe here specifically because pad_to_window()
    puts every file on the identical [start, end] grid first -- there's no
    index to align/outer-join, so this can't introduce any missingness beyond
    what each file already has on its own. This is the actual input shape
    Aerial/RuleNaps need (one row per transaction/timestep, one column per
    attribute), not just a QA-check convenience -- the per-file dataset
    loaders (e.g. an LBNL equivalent of LeakDBDataset) build on this rather
    than re-deriving it.
    """
    processed = {
        fname: pad_to_window(
            resample_grid(clip_window(remove_backward_jump(raw_by_file[fname][cols_by_file[fname]]), start, end)),
            start, end,
        )
        for fname in sorted(cols_by_file)
        if fname not in excluded_files
    }
    return pd.concat(processed.values(), axis=1)
