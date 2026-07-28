import numpy as np
import pandas as pd


class Discretizer:
    """Bins continuous values into discrete categories using equal-frequency binning.

    Two modes:
    - pooled: fit bin edges on all values across all columns combined (used for sensor data,
      where all sensors of the same type share one set of bin edges)
    - per_column: fit bin edges independently per column (used for attribute data,
      where each attribute type has its own distribution)
    """

    def __init__(self, n_bins: int = 3, pooled: bool = False):
        self.n_bins = n_bins
        self.pooled = pooled
        self._bin_edges: dict = {}
        self._actual_n_bins: dict = {}

    def fit(self, df: pd.DataFrame) -> "Discretizer":
        """Compute bin edges from data."""
        if self.pooled:
            # Flatten all sensor readings of a sensor type into single np array, and only consider actual readings by not considering missing values.
            all_values = df.values.flatten()
            all_values = all_values[~np.isnan(all_values)]

            if len(np.unique(all_values)) == 1:
                print(f"Warning: pooled data has only one unique value — assigning all to bin 0.")
                self._bin_edges["__pooled__"] = np.array([-np.inf, np.inf])
                self._actual_n_bins["__pooled__"] = 1
                return self

            # Use the pandas qcut function to perform equal frequency discretization using quantiles to define the bin edges. If duplicate bin edges arise, these are dropped
            _, edges = pd.qcut(all_values, q=self.n_bins, retbins=True, duplicates="drop")

            # first and last edges need to be open such that every value falls within a bin
            edges[0] = -np.inf
            edges[-1] = np.inf

            # Set the bin edges and report the actual number of bins found (could be affected by duplicate bin edges)
            self._bin_edges["__pooled__"] = edges
            self._actual_n_bins["__pooled__"] = len(edges) - 1

            if self._actual_n_bins["__pooled__"] != self.n_bins:
                print(f"Warning: requested {self.n_bins} bins but got {self._actual_n_bins['__pooled__']} due to duplicate bin edges.")
                
        else:
            for col in df.columns:

                # exclude missing values
                values = df[col].dropna()

                if values.nunique() == 1:
                    print(f"Warning: column '{col}' has only one unique value — assigning all to bin 0.")
                    self._bin_edges[col] = np.array([-np.inf, np.inf])
                    self._actual_n_bins[col] = 1
                    continue

                # Use the pandas qcut function to perform equal frequency discretization using quantiles to define the bin edges. If duplicate bin edges arise, these are dropped
                _, edges = pd.qcut(values, q=self.n_bins, retbins=True, duplicates="drop")

                # first and last edges need to be open such that every value falls within a bin
                edges[0] = -np.inf
                edges[-1] = np.inf

                # Set the bin edges and report the actual number of bins found (could be affected by duplicate bin edges)
                self._bin_edges[col] = edges
                self._actual_n_bins[col] = len(edges) - 1

                if self._actual_n_bins[col] != self.n_bins:
                    print(f"Warning: column '{col}' requested {self.n_bins} bins but got {self._actual_n_bins[col]} due to duplicate bin edges.")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map continuous values to integer bin indices (0 to n_bins-1)."""
        # create new dataframe to store binned values
        disc_df = pd.DataFrame(index=df.index)

        # for each column in the dataframe discretize the values, using either the pooled bins, or column specific bins. NaN values are not assigned to a bin.
        for col in df.columns:
            edges = self._bin_edges["__pooled__"] if self.pooled else self._bin_edges[col]
            disc_df[col] = pd.cut(df[col], bins=edges, labels=False) # use labels=False to return bin indices 

        return disc_df


    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        "compute the bin edges from the data and subsequently apply these to the data"
        return self.fit(df).transform(df)

    def get_n_bins(self, col: str = None) -> int:
        """Return the actual number of bins for a column (or pooled bins if col is None)."""
        if not self.pooled and col is None:
            raise ValueError("Discretizer is not in pooled mode — provide a column name.")
        key = "__pooled__" if self.pooled else col
        return self._actual_n_bins[key]

    def bin_label(self, col: str, bin_idx: int) -> str:
        """Return a human-readable label for a (column, bin_index) pair."""
        edges = self._bin_edges["__pooled__"] if self.pooled else self._bin_edges[col]
        low = edges[bin_idx]
        high = edges[bin_idx + 1]
        low_str = f"{low:.3f}" if not np.isinf(low) else "-inf"
        high_str = f"{high:.3f}" if not np.isinf(high) else "inf"
        return f"{col}_{bin_idx}: [{low_str},{high_str})"
