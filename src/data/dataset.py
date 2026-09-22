import json
import os
import random

import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch_geometric.data import Dataset

from src.data.loader import LeakDBLoader
from src.data.discretizer import Discretizer
from src.data.graph_builder import GraphBuilder


class LeakDBDataset(Dataset):
    """PyG Dataset for LeakDB water network graphs.

    One graph_{idx}.pt per (scenario, timestamp) pair; metadata.pt stores the
    train/val split indices, and n_bins_per_type.

    Processing runs once. Delete the processed directory to reprocess with
    different config (e.g. different scenarios or n_bins). The process method runs
    automatically when the files returned by processed_file_names are not present.
    """

    def __init__(self, config, transform=None):
        self.config = config

        # the topology is static across scennarios and is therefore only once at initialization
        self.loader = LeakDBLoader(config.raw_data_dir)
        self.topology = self.loader.load_topology()

        # here the root determines the root of the dataset which will contain a raw and processed folder
        super().__init__(root=config.data_dir, transform=transform)

        with open(os.path.join(self.processed_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.n_bins_per_type = meta['n_bins_per_type']
        self.train_indices = meta['train_indices']
        self.val_indices   = meta['val_indices']

        self.base_graphs = {
            s: torch.load(os.path.join(self.processed_dir, f'base_{s}.pt'), weights_only=False)
            for s in self.config.scenarios
        }

        self.df = pd.read_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

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
        # --- Step 1: build tabular sensor DataFrame (one row per timestamp, one column per sensor) ---
        scenario_dfs = []
        for scenario in self.config.scenarios:
            # load the sensor data dict for this scenario and obtain number of timestamps needed for the specified data fraction
            sd = self.loader.load_sensor_data(scenario)
            n_t = int(len(sd['pressures']) * self.config.data_fraction)

            # add sensor type prefixes to the column names 
            pressures = sd['pressures'].iloc[:n_t].rename(columns=lambda c: f'pressure;{c}')
            flows     = sd['flows'].iloc[:n_t].rename(columns=lambda c: f'flow;{c}')
            demands   = sd['demands'].iloc[:n_t].rename(columns=lambda c: f'demand;{c}')
            label_col = sd['labels'].iloc[:n_t, 0].astype(int).rename('label')

            # concatenat the dfs column wise to have all sensor data from a scenario in one df and add a column specifying the scenario number
            df_s = pd.concat([pressures, flows, demands, label_col], axis=1)
            df_s['scenario'] = scenario
            scenario_dfs.append(df_s)

        # concatenate sensor dfs from all scenarios in to a single one, row-wise.
        df = pd.concat(scenario_dfs, axis=0).reset_index(drop=True)

        # --- Step 2: shuffle and split train/val ---
        # shuffle the df indices and create a new df using that order
        n = len(df)
        rng = random.Random(self.config.random_seed)
        shuffled_idx = list(range(n))
        rng.shuffle(shuffled_idx)
        df = df.iloc[shuffled_idx].reset_index(drop=True)

        # obtain the train/val split and create a column specifying for each timestamp if its a train or val instance
        n_train = int(n * self.config.train_ratio)
        train_indices = list(range(n_train))
        val_indices   = list(range(n_train, n))
        df['split'] = ['train' if i < n_train else 'val' for i in range(n)]

        # --- Step 3: fit discretizers on training instances only ---
        # training df only containing training instances
        train_df = df[df['split'] == 'train']

        # obtain all columns per sensor type
        p_cols = [c for c in df.columns if c.startswith('pressure;')]
        f_cols = [c for c in df.columns if c.startswith('flow;')]
        d_cols = [c for c in df.columns if c.startswith('demand;')]

        # per sensor type fit a discretizer with only sensor columns of that type, then immediately transform all rows
        disc_p = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[p_cols])
        disc_p_df = disc_p.transform(df[p_cols])

        disc_f = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[f_cols])
        disc_f_df = disc_f.transform(df[f_cols])

        disc_d = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[d_cols])
        disc_d_df = disc_d.transform(df[d_cols])

        # obtain the number of actual bins per type (n_bins can differ from what user specified)
        n_bins_per_type = {
            'pressure': disc_p.get_n_bins(),
            'flow':     disc_f.get_n_bins(),
            'demand':   disc_d.get_n_bins(),
        }

        # --- Step 4: load and normalize static node/pipe attributes ---
        # load all included scenario attributes at once and fit a RobustScaler per attribute group
        # over all (node/pipe, scenario) pairs — no leakage risk since attributes are static
        # and every scenario is represented in the training set
        attrs_dfs = self.loader.load_attributes_as_df(self.config.scenarios)
        junc_df = attrs_dfs['junctions']  # index: (node_id, scenario), col: base_demand
        pipe_df = attrs_dfs['pipes']      # index: (pipe_id, scenario), cols: length, diameter, roughness

        if self.config.normalize_attributes:
            junc_scaler = RobustScaler()
            pipe_scaler = RobustScaler()
            junc_df = pd.DataFrame(
                junc_scaler.fit_transform(junc_df),
                index=junc_df.index, columns=junc_df.columns,
            )
            pipe_df = pd.DataFrame(
                pipe_scaler.fit_transform(pipe_df),
                index=pipe_df.index, columns=pipe_df.columns,
            )

        # --- Step 5: build and save all graphs (scenario by scenario because each scenario has a different base graph) ---
        # instantiate a GraphBuilder instance
        gb = GraphBuilder(
            self.topology,
            n_bins_per_type,
            bidirectional=self.config.bidirectional_has_measure,
            virtual_node_mode=self.config.virtual_node_mode,
        )

        for scenario in self.config.scenarios:
            # slice the attributes for this scenario and build the base graph
            junc_s = junc_df.xs(scenario, level='scenario')
            pipe_s = pipe_df.xs(scenario, level='scenario')
            base = gb.build_base(junc_s, pipe_s)
            torch.save(base, os.path.join(self.processed_dir, f'base_{scenario}.pt'))

            # iterate over all rows beloning to this scenario
            sc_df = df[df['scenario'] == scenario]
            for graph_idx, row in sc_df.iterrows():
                # build single graph
                graph = gb.build(
                    base,
                    disc_p_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),   # type prefix is removed as the graph builder expects
                    disc_f_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),   # .loc returns series where column names became the index
                    disc_d_df.loc[graph_idx].rename(index=lambda c: c.split(';')[1]),   # therefore index instead of columns are renamed
                )
                graph.y = torch.tensor([int(row['label'])], dtype=torch.long)

                # validate the graph formatting and save it
                graph.validate(raise_on_error=True)
                torch.save(
                    graph,
                    os.path.join(self.processed_dir, f'graph_{graph_idx}.pt'),
                )

        # --- Step 6: build and save one-hot transaction DataFrame (all rows) ---
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

        # one hot encoded df where column names are of format "sensor_type;node_name;bin_id"
        hot_df = pd.DataFrame(hot_cols, index=df.index)

        # --- Step 7: save dataset DataFrame and metadata ---
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
