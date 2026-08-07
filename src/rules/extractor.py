import random
from itertools import combinations

import torch

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
        """Enumerate all valid antecedents, query the model for each, and return candidate rules.

        Each rule is a dict with 'antecedent' and 'consequent' keys.
        Support and confidence are left to the evaluator.
        """
        rules = []
        for antecedent in self._get_all_antecedents():
            consequents = self._query_model(antecedent)
            for consequent in consequents:
                rules.append({'antecedent': list(antecedent), 'consequent': consequent})
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
                for (st, sname, _) in combo:
                    if (st, sname) in sensors_seen:
                        valid = False
                        break
                    sensors_seen.add((st, sname))
                if valid:
                    antecedents.append(combo)
        return antecedents

    def _query_model(self, antecedent: list[tuple]) -> list:
        """Inject antecedent into a query graph, run the model, and return consequent items.

        Returns an empty list if any antecedent item is not reconstructed above
        antecedent_threshold — the model did not confidently recognise the antecedent.
        """
        graph = self._build_query_graph(antecedent)

        self.model.eval()
        with torch.no_grad():
            probs = self.model(graph.x_dict, graph.edge_index_dict)

        antecedent_sensors = {(st, sname) for (st, sname, _) in antecedent}

        # verify antecedent reconstruction
        for (st, sname, bin_idx) in antecedent:
            s_idx = self.gb.sensor_idx[(st, sname)]
            row = self.sensor_to_row[st][s_idx]
            if probs[st][row, bin_idx].item() < self.config.antecedent_threshold:
                return []

        # collect consequents from non-antecedent sensors above threshold
        consequents = []
        for st in self.gb.SENSOR_TYPES:
            type_probs = probs[st]  # [n_sensors_of_type, n_bins]
            for row, s_idx in enumerate(self.gb.sensor_indices_by_type[st].tolist()):
                (_, sname) = self.idx_to_sensor[s_idx]
                if (st, sname) in antecedent_sensors:
                    continue
                for bin_idx in range(self.gb.n_bins_per_type[st]):
                    if type_probs[row, bin_idx].item() >= self.config.consequent_threshold:
                        consequents.append((st, sname, bin_idx))

        return consequents

    def _build_query_graph(self, antecedent: list[tuple]):
        """Clone a random base graph and inject has_measure edges for the antecedent items."""
        scenario = self._rng.choice(list(self.dataset.base_graphs.keys()))
        graph = self.dataset.base_graphs[scenario].clone().to(self.device)

        src, dst = [], []
        for (st, sname, bin_idx) in antecedent:
            src.append(self.gb.sensor_idx[(st, sname)])
            dst.append(self.gb.value_node_idx[(st, bin_idx)])

        ei = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        graph[('sensor', 'has_measure', 'value_node')].edge_index = ei
        if self.gb.bidirectional:
            graph[('value_node', 'measured_by', 'sensor')].edge_index = ei.flip(0)

        return graph

    def _get_all_items(self) -> list:
        """Return all possible (sensor_type, sensor_name, bin_idx) items."""
        items = []
        for (st, sname) in self.gb.sensor_idx:
            for bin_idx in range(self.gb.n_bins_per_type[st]):
                items.append((st, sname, bin_idx))
        return items
