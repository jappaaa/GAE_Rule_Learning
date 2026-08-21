import random
from itertools import combinations

import torch
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm

from src.data.dataset import LeakDBDataset
from src.data.graph_builder import GraphBuilder
from src.model.gae import GraphAutoEncoder
from src.utils.config import Config


class RuleExtractor:
    def __init__(self, model: GraphAutoEncoder, graph_builder: GraphBuilder,
                 dataset: LeakDBDataset, config: Config, device: torch.device):
        self.model = model
        self.gb = graph_builder
        self.dataset = dataset
        self.config = config
        self.device = device

        # inverse lookups to map indices back to (sensor_type, sensor_name) and bin_idx
        self.idx_to_sensor = {v: k for k, v in graph_builder.sensor_idx.items()}
        self.idx_to_bin = {v: k for k, v in graph_builder.value_node_idx.items()}

        self._rng = random.Random(config.random_seed)

        # maps (sensor_type, global_sensor_idx) -> row in probs[sensor_type] tensor
        self.sensor_to_row = {
            st: {s_idx.item(): row for row, s_idx in enumerate(graph_builder.sensor_indices_by_type[st])}
            for st in graph_builder.SENSOR_TYPES
        }

    def extract(self) -> list[dict]:
        """Enumerate all valid antecedents, query the model in batches, and return candidate rules.

        Each rule is a dict with 'antecedent' and 'consequent' keys.
        Support and confidence are left to the evaluator.
        """
        antecedents = self._get_all_antecedents()
        rules = []
        bs = self.config.extraction_batch_size
        for start in tqdm(range(0, len(antecedents), bs),
                          desc=f"Querying antecedents (batch={bs}, max_size={self.config.max_antecedent_size})"):
            rules.extend(self._query_model_batch(antecedents[start:start + bs]))
        return rules

    def _get_all_antecedents(self) -> list:
        """Return all valid antecedent combinations up to max_antecedent_size.
        Combinations where the same sensor appears with two different bins are excluded."""
        items = self._get_all_items()
        antecedents = []
        for size in range(1, self.config.max_antecedent_size + 1):
            for combo in combinations(items, size):  
                sensors_seen = set()
                valid = True
                for (st, sname, _) in combo: # check whether there is only one item per sensor (sensors can only be assigned to one bin)
                    if (st, sname) in sensors_seen:
                        valid = False
                        break
                    sensors_seen.add((st, sname))
                if valid:
                    antecedents.append(combo)
        return antecedents

    def _query_model_batch(self, antecedents: list) -> list[dict]:
        """Build a batch of query graphs, run one model forward pass, split output per antecedent."""
        graphs = [self.build_query_graph(ant) for ant in antecedents]
        batch = PyGBatch.from_data_list(graphs).to(self.device)

        self.model.eval()
        with torch.no_grad():
            probs_batch = self.model(batch.x_dict, batch.edge_index_dict)

        # split batched probs back into per-graph slices: [n_graphs, n_sensors_of_type, n_bins]
        n_sensors_by_type = {st: len(self.gb.sensor_indices_by_type[st]) for st in self.gb.SENSOR_TYPES}
        probs_split = {
            st: probs_batch[st].view(len(antecedents), n_sensors_by_type[st], -1)
            for st in self.gb.SENSOR_TYPES
        }

        rules = []
        for i, antecedent in enumerate(antecedents):
            # single-graph probs slice: {st: [n_sensors_of_type, n_bins]}
            probs = {st: probs_split[st][i] for st in self.gb.SENSOR_TYPES}
            for consequent in self._check_thresholds(antecedent, probs):
                rules.append({'antecedent': list(antecedent), 'consequent': consequent})
        return rules

    def _check_thresholds(self, antecedent: tuple, probs: dict) -> list:
        """Apply antecedent and consequent thresholds to single-graph probs.

        Returns an empty list if any antecedent item is not reconstructed above
        antecedent_threshold — the model did not confidently recognise the antecedent.
        """
        antecedent_sensors = {(st, sname) for (st, sname, _) in antecedent}

        for (st, sname, bin_idx) in antecedent:
            s_idx = self.gb.sensor_idx[(st, sname)]
            row = self.sensor_to_row[st][s_idx]
            if probs[st][row, bin_idx].item() < self.config.antecedent_threshold:
                return []

        consequents = []
        for st in self.gb.SENSOR_TYPES:
            type_probs = probs[st]  # [n_sensors_of_type, n_bins]
            for row, s_idx in enumerate(self.gb.sensor_indices_by_type[st].tolist()):
                _, sname = self.idx_to_sensor[s_idx]
                if (st, sname) in antecedent_sensors:
                    continue
                for bin_idx in range(self.gb.n_bins_per_type[st]):
                    if type_probs[row, bin_idx].item() >= self.config.consequent_threshold:
                        consequents.append((st, sname, bin_idx))
        return consequents

    def build_query_graph(self, antecedent: list[tuple]):
        """Clone a random base graph and inject measured_by edges for the antecedent items."""
        scenario = self._rng.choice(list(self.dataset.base_graphs.keys()))
        graph = self.dataset.base_graphs[scenario].clone().to(self.device)

        src, dst = [], []
        for (st, sname, bin_idx) in antecedent:
            src.append(self.gb.value_node_idx[(st, bin_idx)])
            dst.append(self.gb.sensor_idx[(st, sname)])

        ei = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        graph[('value_node', 'measured_by', 'sensor')].edge_index = ei
        if self.gb.bidirectional:
            graph[('sensor', 'has_measure', 'value_node')].edge_index = ei.flip(0)

        return graph

    def _get_all_items(self) -> list:
        """Return all possible (sensor_type, sensor_name, bin_idx) items."""
        items = []
        for (st, sname) in self.gb.sensor_idx:
            for bin_idx in range(self.gb.n_bins_per_type[st]):
                items.append((st, sname, bin_idx))
        return items
