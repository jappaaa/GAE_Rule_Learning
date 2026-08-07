import os
import random
from collections import defaultdict

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

        meta = torch.load(
            os.path.join(self.processed_dir, 'metadata.pt'),
            weights_only=False,
        )
        self.pairs = meta['pairs']                  # list of (scenario, t, label)
        self.train_indices = meta['train_indices']
        self.val_indices = meta['val_indices']
        self.test_indices = meta['test_indices']
        self.n_bins_per_type = meta['n_bins_per_type']
        self.base_graphs = {
            s: torch.load(os.path.join(self.processed_dir, f'base_{s}.pt'), weights_only=False)
            for s in self.config.scenarios
        }

    @property
    def raw_file_names(self):
        return []  # data is already local, no download needed

    @property
    def processed_file_names(self):
        return ['metadata.pt'] + [f'base_{s}.pt' for s in self.config.scenarios]

    def process(self):
        # Load each scenario once and cache for use across all steps
        sensor_cache = {s: self.loader.load_sensor_data(s) for s in self.config.scenarios}

        # --- Step 1: collect all (scenario, t, label) pairs ---
        all_pairs = []
        for scenario in self.config.scenarios:
            sd = sensor_cache[scenario]
            n_t = len(sd['pressures'])
            for t in range(n_t):
                label = int(sd['labels'].iloc[t, 0])
                all_pairs.append((scenario, t, label))

        # --- Step 2: shuffle and split by ratio ---
        rng = random.Random(self.config.random_seed)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        test_indices = list(range(n_train + n_val, n))

        # --- Step 3: fit discretizers on training timestamps only ---
        train_ts_by_scenario = defaultdict(list)
        for idx in train_indices:
            scenario, t, _ = all_pairs[idx]
            train_ts_by_scenario[scenario].append(t)

        pressure_list, flow_list, demand_list = [], [], []
        for scenario, timestamps in train_ts_by_scenario.items():
            sd = sensor_cache[scenario]
            pressure_list.append(sd['pressures'].iloc[timestamps])
            flow_list.append(sd['flows'].iloc[timestamps])
            demand_list.append(sd['demands'].iloc[timestamps])

        disc_p = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(pd.concat(pressure_list))
        disc_f = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(pd.concat(flow_list))
        disc_d = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(pd.concat(demand_list))

        n_bins_per_type = {
            'pressure': disc_p.get_n_bins(),
            'flow':     disc_f.get_n_bins(),
            'demand':   disc_d.get_n_bins(),
        }

        gb = GraphBuilder(
            self.topology,
            n_bins_per_type,
            bidirectional=self.config.bidirectional_has_measure,
        )

        # --- Step 4: build and save all graphs (scenario by scenario) ---
        # pair_idx_map: (scenario, t) -> position in shuffled all_pairs (= graph filename index)
        pair_idx_map = {(s, t): i for i, (s, t, _) in enumerate(all_pairs)}

        for scenario in self.config.scenarios:
            sd = sensor_cache[scenario]
            attrs = self.loader.load_attributes(scenario)
            base = gb.build_base(attrs)

            torch.save(base, os.path.join(self.processed_dir, f'base_{scenario}.pt'))

            pressures_disc = disc_p.transform(sd['pressures'])
            flows_disc     = disc_f.transform(sd['flows'])
            demands_disc   = disc_d.transform(sd['demands'])

            for t in range(len(sd['pressures'])):
                graph_idx = pair_idx_map[(scenario, t)]
                graph = gb.build(
                    base,
                    pressures_disc.iloc[t],
                    flows_disc.iloc[t],
                    demands_disc.iloc[t],
                )
                graph.y = torch.tensor(
                    [all_pairs[graph_idx][2]], dtype=torch.long
                )
                graph.validate(raise_on_error=True)
                torch.save(
                    graph,
                    os.path.join(self.processed_dir, f'graph_{graph_idx}.pt'),
                )

        # --- Step 5: save metadata ---
        torch.save(
            {
                'pairs':         all_pairs,
                'train_indices': train_indices,
                'val_indices':   val_indices,
                'test_indices':  test_indices,
                'n_bins_per_type': n_bins_per_type,
                'disc_pressure': disc_p,
                'disc_flow':     disc_f,
                'disc_demand':   disc_d,
            },
            os.path.join(self.processed_dir, 'metadata.pt'),
        )

    def len(self):
        return len(self.pairs)

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'graph_{idx}.pt'),
            weights_only=False,
        )
