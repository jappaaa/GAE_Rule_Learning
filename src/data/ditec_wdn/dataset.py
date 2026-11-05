"""PyG Dataset for DiTEC-WDN water network graphs. Mirrors LeakDBDataset's role and on-disk
layout (one graph_{idx}.pt per (scenario, timestamp), metadata.json, base_{scenario}.pt),
reading from Hugging Face parquet shards via DiTECWDNLoader instead of local EPANET files.

Run from the project root to check it end-to-end in isolation:
    python -m src.data.ditec_wdn.dataset
"""
import json
import os
import random

import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Dataset

from src.data.discretizer import Discretizer
from src.data.ditec_wdn.loader import DiTECWDNLoader
from src.data.graph_builder import GraphBuilder

# Which of DiTEC-WDN's 7 dynamic_output signals are measured per-node vs per-link, confirmed
# from the dynamic_output file names themselves (node_pressure/node_demand/node_head vs
# link_flowrate/link_velocity/link_headloss/link_friction_factor) -- see docs/experiment_plan.md.
NODE_SIGNALS = {'pressure', 'demand', 'head'}


class DiTECWDNDataset(Dataset):
    """One graph_{idx}.pt per (scenario, timestamp) pair; metadata.json stores the
    train/val split indices and n_bins_per_type.

    Processing runs once. Delete the processed directory to reprocess with a different
    config (e.g. different scenarios or n_bins). process() runs automatically when the
    files processed_file_names lists aren't present.
    """

    def __init__(self, config, transform=None):
        self.config = config
        self.loader = DiTECWDNLoader(network=config.ditec_network)
        self.topology = self.loader.load_topology()

        super().__init__(root=config.data_dir, transform=transform)

        with open(os.path.join(self.processed_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.n_bins_per_type = meta['n_bins_per_type']
        self.train_indices = meta['train_indices']
        self.val_indices = meta['val_indices']

        self.base_graphs = {
            s: torch.load(os.path.join(self.processed_dir, f'base_{s}.pt'), weights_only=False)
            for s in self.config.scenarios
        }

        self.df = pd.read_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

        item_cols = [c for c in self.df.columns if c not in ('scenario_id', 'time_id', 'split')]
        self.items = [(st, sname, int(b)) for st, sname, b in (c.split(';') for c in item_cols)]
        self.tensor = torch.tensor(self.df[item_cols].values, dtype=torch.bool)

        signal_types = sorted(self.n_bins_per_type.keys())
        node_sensor_types = [s for s in signal_types if s in NODE_SIGNALS]
        self.gb = GraphBuilder(
            self.topology,
            self.n_bins_per_type,
            bidirectional=config.bidirectional_has_measure,
            virtual_node_mode=config.virtual_node_mode,
            onehot_bin_features=config.onehot_bin_features,
            sensor_types=signal_types,
            node_sensor_types=node_sensor_types,
        )

    @property
    def raw_file_names(self):
        return []  # data is fetched from Hugging Face directly, no local download step

    @property
    def processed_file_names(self):
        return ['dataset.parquet', 'dataset_categorical.parquet', 'metadata.json'] + [f'base_{s}.pt' for s in self.config.scenarios]

    def process(self):
        # --- Step 1: build tabular sensor DataFrame (one row per (scenario, timestamp), one column per signal;instance) ---
        df = self.loader.load_sensor_df(self.config.scenarios, self.config.data_fraction)

        # --- Step 2: shuffle and split train/val ---
        n = len(df)
        rng = random.Random(self.config.data_seed)  # kept separate from random_seed so varying the model seed across runs never resamples the split
        shuffled_idx = list(range(n))
        rng.shuffle(shuffled_idx)
        df = df.iloc[shuffled_idx].reset_index()  # (scenario_id, time_id) index -> plain columns; still needed later to build one base graph per scenario

        n_train = int(n * self.config.train_ratio)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))
        df['split'] = ['train' if i < n_train else 'val' for i in range(n)]

        # --- Step 3: fit discretizers on training instances only ---
        # one pooled discretizer per signal (not per LeakDBDataset's 3 sensor types -- DiTEC-WDN
        # has 7: pressure/demand/head/flowrate/velocity/headloss/friction_factor), looped since
        # writing out 7 near-identical blocks by hand would just be repeated code
        train_df = df[df['split'] == 'train']
        signals = sorted({c.split(';')[0] for c in df.columns if ';' in c})

        disc_dfs = {}
        n_bins_per_type = {}
        for signal in signals:
            cols = [c for c in df.columns if c.startswith(f'{signal};')]
            disc = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[cols])
            disc_dfs[signal] = disc.transform(df[cols])
            n_bins_per_type[signal] = disc.get_n_bins()

        # --- Step 4: load and normalize static attributes ---
        # {component_type: df}, one df per component type present in this network (not per
        # LeakDBDataset's 2 fixed types, junctions/pipes -- DiTEC-WDN's set varies by network),
        # looped for the same reason Step 3 loops over signals. No leakage risk since attributes
        # are static and every scenario is represented in the (implicitly whole-dataset) fit.
        attrs_dfs = self.loader.load_attributes_as_df(self.config.scenarios)

        if self.config.normalize_attributes:
            for component_type, attr_df in attrs_dfs.items():
                if len(attr_df) == 0:  # e.g. a network with no reservoirs has an empty reservoir df
                    continue
                scaler = RobustScaler()
                attrs_dfs[component_type] = pd.DataFrame(
                    scaler.fit_transform(attr_df),
                    index=attr_df.index, columns=attr_df.columns,
                )

        # --- Step 5: build GraphBuilder + one base graph per scenario + one graph per row ---
        signal_types = sorted(disc_dfs.keys())
        node_sensor_types = [s for s in signal_types if s in NODE_SIGNALS]

        gb = GraphBuilder(
            self.topology,
            n_bins_per_type,
            bidirectional=self.config.bidirectional_has_measure,
            virtual_node_mode=self.config.virtual_node_mode,
            onehot_bin_features=self.config.onehot_bin_features,
            sensor_types=signal_types,
            node_sensor_types=node_sensor_types,
        )

        for scenario in self.config.scenarios:
            scenario_attrs = {
                ct: attr_df.xs(scenario, level='scenario_id')
                for ct, attr_df in attrs_dfs.items()
                if len(attr_df) > 0
            }
            base = gb.build_base(scenario_attrs)
            torch.save(base, os.path.join(self.processed_dir, f'base_{scenario}.pt'))

            sc_df = df[df['scenario_id'] == scenario]
            for graph_idx, row in sc_df.iterrows():
                # "signal;raw_id" -> "raw_id" per row (non-mutating -- disc_dfs itself stays
                # untouched for Step 6 below, same as LeakDBDataset.process() does)
                sensor_rows = {
                    signal: disc_dfs[signal].loc[graph_idx].rename(index=lambda c: c.split(';', 1)[1])
                    for signal in signal_types
                }
                graph = gb.build(base, sensor_rows)
                graph.validate(raise_on_error=True)
                torch.save(graph, os.path.join(self.processed_dir, f'graph_{graph_idx}.pt'))

        # --- Step 6: build and save one-hot transaction DataFrame (all rows), along with the discretized categorical version ---
        hot_cols = {}
        for signal in signal_types:
            for c in disc_dfs[signal].columns:  # "signal;raw_id"
                for bin_idx in range(n_bins_per_type[signal]):
                    hot_cols[f'{c};{bin_idx}'] = (disc_dfs[signal][c].values == bin_idx)

        # one hot encoded df where column names are of format "signal;raw_id;bin_id"
        hot_df = pd.DataFrame(hot_cols, index=df.index)

        # categorical (non-one-hot) counterpart: one column per sensor, named "signal;raw_id", valued with
        # the discretized bin index -- hot_df above is derived from exactly these frames, so this is the
        # natural source format for baselines that expect discretized-but-not-one-hot input (e.g. Aerial)
        cat_df = pd.concat([disc_dfs[signal] for signal in signal_types], axis=1)

        # --- Step 7: save dataset DataFrames and metadata ---
        dataset_df = pd.concat([df[['scenario_id', 'time_id', 'split']], hot_df], axis=1)
        dataset_df.to_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

        dataset_cat_df = pd.concat([df[['scenario_id', 'time_id', 'split']], cat_df], axis=1)
        dataset_cat_df.to_parquet(os.path.join(self.processed_dir, 'dataset_categorical.parquet'))

        with open(os.path.join(self.processed_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'n_bins_per_type': n_bins_per_type,
                'train_indices': train_indices,
                'val_indices': val_indices,
            }, f)

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'graph_{idx}.pt'),
            weights_only=False,
        )


if __name__ == '__main__':
    from src.utils.config import Config

    config = Config(dataset_name='ditec_wdn', scenarios=[0, 1, 2])
    ds = DiTECWDNDataset(config)
    print(f"network: {ds.config.ditec_network}")
    print(f"topology: {len(ds.topology['nodes'])} nodes, {len(ds.topology['links'])} links")
    print(f"\ndataset ready -- {len(ds)} graphs | {len(ds.train_indices)} train / {len(ds.val_indices)} val")
    print(f"n_bins_per_type: {ds.n_bins_per_type}")
    print(f"n items: {len(ds.items)}  example items: {ds.items[:3]}")
    print(f"tensor shape: {tuple(ds.tensor.shape)}")

    print(f"\nbase graph (scenario {config.scenarios[0]}): {ds.base_graphs[config.scenarios[0]]}")
    print(f"\nsample graph (idx 0): {ds.get(0)}")
