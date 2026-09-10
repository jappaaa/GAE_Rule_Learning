import json
import os
import random

import pandas as pd
import torch
from torch_geometric.data import Dataset

from src.data.loader import LeakDBLoader
from src.data.discretizer import Discretizer
from src.data.graph_builder import GraphBuilder


class LeakDBDataset(Dataset):
    """PyG Dataset for LeakDB water network graphs.

    One graph_{idx}.pt per (scenario, timestamp) pair; metadata.pt stores the
    train/val/test split indices, fitted discretizers, and n_bins_per_type.

    Processing runs once. Delete the processed directory to reprocess with
    different config (e.g. different scenarios or n_bins).
    """

    def __init__(self, config, transform=None):
        self.config = config
        self.loader = LeakDBLoader(config.raw_data_dir)
        self.topology = self.loader.load_topology()

        super().__init__(root=config.data_dir, transform=transform)

        self.df = pd.read_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

        with open(os.path.join(self.processed_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.n_bins_per_type = meta['n_bins_per_type']
        self.train_indices = meta['train_indices']
        self.val_indices   = meta['val_indices']

        self.base_graphs = {
            s: torch.load(os.path.join(self.processed_dir, f'base_{s}.pt'), weights_only=False)
            for s in self.config.scenarios
        }

        item_cols = [c for c in self.df.columns if c not in ('scenario', 'split', 'label')]
        self.items = [(st, sname, int(b)) for st, sname, b in (c.split(';') for c in item_cols)]
        self.tensor = torch.tensor(self.df[item_cols].values, dtype=torch.bool)

        self.gb = GraphBuilder(
            self.topology,
            self.n_bins_per_type,
            bidirectional=config.bidirectional_has_measure,
            virtual_node_mode=config.virtual_node_mode,
        )

    @property
    def raw_file_names(self):
        return []  # data is already local, no download needed

    @property
    def processed_file_names(self):
        return ['dataset.parquet', 'metadata.json'] + [f'base_{s}.pt' for s in self.config.scenarios]

    def process(self):
        # --- Step 1: build raw wide DataFrame (one row per timestamp, one column per sensor) ---
        scenario_dfs = []
        for scenario in self.config.scenarios:
            sd = self.loader.load_sensor_data(scenario)
            n_t = int(len(sd['pressures']) * self.config.data_fraction)

            pressures = sd['pressures'].iloc[:n_t].rename(columns=lambda c: f'pressure;{c}')
            flows     = sd['flows'].iloc[:n_t].rename(columns=lambda c: f'flow;{c}')
            demands   = sd['demands'].iloc[:n_t].rename(columns=lambda c: f'demand;{c}')
            label_col = sd['labels'].iloc[:n_t, 0].astype(int).rename('label')

            df_s = pd.concat([pressures, flows, demands, label_col], axis=1)
            df_s['scenario'] = scenario
            scenario_dfs.append(df_s)

        df = pd.concat(scenario_dfs, axis=0).reset_index(drop=True)

        # --- Step 2: shuffle and split by ratio ---
        n = len(df)
        rng = random.Random(self.config.random_seed)
        shuffled_idx = list(range(n))
        rng.shuffle(shuffled_idx)
        df = df.iloc[shuffled_idx].reset_index(drop=True)

        n_train = int(n * self.config.train_ratio)
        train_indices = list(range(n_train))
        val_indices   = list(range(n_train, n))
        df['split'] = ['train' if i < n_train else 'val' for i in range(n)]

        # --- Step 3: fit discretizers on training rows only ---
        train_df = df[df['split'] == 'train']
        p_cols = [c for c in df.columns if c.startswith('pressure;')]
        f_cols = [c for c in df.columns if c.startswith('flow;')]
        d_cols = [c for c in df.columns if c.startswith('demand;')]

        disc_p = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[p_cols])
        disc_f = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[f_cols])
        disc_d = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[d_cols])

        n_bins_per_type = {
            'pressure': disc_p.get_n_bins(),
            'flow':     disc_f.get_n_bins(),
            'demand':   disc_d.get_n_bins(),
        }

        gb = GraphBuilder(
            self.topology,
            n_bins_per_type,
            bidirectional=self.config.bidirectional_has_measure,
            virtual_node_mode=self.config.virtual_node_mode,
        )

        disc_p_df = disc_p.transform(df[p_cols])
        disc_f_df = disc_f.transform(df[f_cols])
        disc_d_df = disc_d.transform(df[d_cols])

        # --- Step 4: build and save all graphs (scenario by scenario because each scenario has different base graph) ---
        for scenario in self.config.scenarios:
            attrs = self.loader.load_attributes(scenario)
            base = gb.build_base(attrs)

            torch.save(base, os.path.join(self.processed_dir, f'base_{scenario}.pt'))

            sc_df = df[df['scenario'] == scenario]
            for graph_idx, row in sc_df.iterrows():
                graph = gb.build(
                    base,
                    disc_p_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),
                    disc_f_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),
                    disc_d_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),
                )
                graph.y = torch.tensor([int(row['label'])], dtype=torch.long)
                graph.validate(raise_on_error=True)
                torch.save(
                    graph,
                    os.path.join(self.processed_dir, f'graph_{graph_idx}.pt'),
                )

        # --- Step 5: build and save one-hot transaction DataFrame (all rows) ---
        hot_cols = {}
        for c in p_cols:
            for bin_idx in range(n_bins_per_type['pressure']):
                hot_cols[f'{c};{bin_idx}'] = (disc_p_df[c].values == bin_idx)
        for c in f_cols:
            for bin_idx in range(n_bins_per_type['flow']):
                hot_cols[f'{c};{bin_idx}'] = (disc_f_df[c].values == bin_idx)
        for c in d_cols:
            for bin_idx in range(n_bins_per_type['demand']):
                hot_cols[f'{c};{bin_idx}'] = (disc_d_df[c].values == bin_idx)

        hot_df = pd.DataFrame(hot_cols, index=df.index)

        # --- Step 6: save dataset DataFrame and metadata ---
        dataset_df = pd.concat([df[['scenario', 'split', 'label']], hot_df], axis=1)
        dataset_df.to_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

        with open(os.path.join(self.processed_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'n_bins_per_type': n_bins_per_type,
                'train_indices':   train_indices,
                'val_indices':     val_indices,
            }, f)

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'graph_{idx}.pt'),
            weights_only=False,
        )
