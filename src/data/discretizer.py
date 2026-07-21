import pandas as pd


class Discretizer:
    """Bins continuous sensor readings into discrete categories.

    Strategy matches Aerial: configurable equal-width or equal-frequency binning.
    Fit on training data only; transform applied to all splits.
    """

    def __init__(self, n_bins: int = 3, strategy: str = "equal-width"):
        self.n_bins = n_bins
        self.strategy = strategy
        self._bin_edges: dict = {}  # sensor_id → bin edges fitted on train set

    def fit(self, df: pd.DataFrame) -> "Discretizer":
        """Compute bin edges from training data."""
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map continuous readings to bin labels (e.g. 0, 1, 2)."""
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def bin_label(self, sensor_id: str, bin_idx: int) -> str:
        """Return a human-readable label for a (sensor, bin) pair."""
        pass
