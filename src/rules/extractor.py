import math
import random
from itertools import combinations

import torch
from torch_geometric.data import Batch as PyGBatch
from tqdm import tqdm

from src.data.leakdb.dataset import LeakDBDataset
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

        # create the sensor hop map in case consequents need to be filtered based on reachability
        if config.filter_consequents_by_hops:
            self.sensor_hop_map = graph_builder.build_sensor_hop_map(max_hops=config.num_layers - 1)

    def extract(self) -> list[dict]:
        """Enumerate valid antecedents, query the model, and return candidate rules.

        Each rule is a dict with 'antecedent' and 'consequent' keys.
        Support and confidence are left to the evaluator.
        Extraction strategy is controlled by config.extraction_strategy:
          'brute_force' — enumerate all combinations upfront, then filter post-query.
          'apriori'     — prune using pairwise frequency estimates before querying.
        """
        if self.config.extraction_strategy == 'apriori':
            return self._extract_apriori()

        antecedents = self._get_all_antecedents()
        rules = []
        bs = self.config.extraction_batch_size
        for start in tqdm(range(0, len(antecedents), bs),
                          desc=f"Querying antecedents (batch={bs}, max_size={self.config.max_antecedent_size})"):
            rules.extend(self._query_model_batch(antecedents[start:start + bs]))
            
        return rules

    def _extract_apriori(self) -> list[dict]:
        """Apriori-style extraction: prune antecedent candidates using pairwise frequency
        estimates derived from single-item forward passes before querying the model."""
        items = self._get_all_items()
        bs = self.config.extraction_batch_size
        n_sensors_by_type = {st: len(self.gb.sensor_indices_by_type[st]) for st in self.gb.SENSOR_TYPES}

        # --- Phase 1: query each item individually as a single-item antecedent ---
        # Self-frequency = P(item | item is marked): the model's own reconstruction confidence.
        # Items below antecedent_threshold are pruned here; their supersets are never queried.
        # Log-prob outputs are stored for frequent items only — needed for pairwise frequency
        # estimates in Phase 2 without any extra model queries.
        item_probs = {}      # item -> {st: prob tensor [n_sensors, n_bins]} (on CPU, frequent items only); used for consequent threshold checks
        item_log_probs = {}  # item -> {st: log-prob tensor [n_sensors, n_bins]} (on CPU, frequent items only); used for pairwise frequency estimates
        frequent_items = []

        single_antecedents = [(item,) for item in items]
        for start in tqdm(range(0, len(single_antecedents), bs),
                          desc="Apriori phase 1: single-item queries"):
            batch_ants = single_antecedents[start:start + bs]
            graphs = [self.build_query_graph(ant) for ant in batch_ants]
            batch = PyGBatch.from_data_list(graphs).to(self.device)
            self.model.eval()
            with torch.no_grad():
                probs_batch = self.model(batch.x_dict, batch.edge_index_dict)
            probs_split = {
                st: probs_batch[st].view(len(batch_ants), n_sensors_by_type[st], -1)
                for st in self.gb.SENSOR_TYPES
            }
            for i, (item,) in enumerate(batch_ants):
                item_st, sname, bin_idx = item
                row = self.sensor_to_row[item_st][self.gb.sensor_idx[(item_st, sname)]]
                probs = {st: probs_split[st][i] for st in self.gb.SENSOR_TYPES}
                freq = probs[item_st][row, bin_idx].item()
                if freq >= self.config.antecedent_threshold:
                    frequent_items.append(item)
                    item_probs[item] = {st: probs[st].cpu() for st in self.gb.SENSOR_TYPES}
                    item_log_probs[item] = {
                        st: torch.log(probs[st].cpu().clamp(min=1e-12))
                        for st in self.gb.SENSOR_TYPES
                    }
                    

        if not frequent_items:
            return []

        # --- Phase 1 continued: sort, emit 1-item rules, initialise combination list ---
        # Sorting by descending self-frequency defines a canonical order: in Phase 2 each
        # antecedent is only extended with items ranked *after* its last element, so every
        # combination is generated exactly once without deduplication bookkeeping.
        frequent_items.sort(key=lambda item: -self._item_log_prob(item_log_probs[item], item))
        item_rank = {item: pos for pos, item in enumerate(frequent_items)}

        rules = []
        for item in frequent_items:
            for consequent in self._check_thresholds((item,), item_probs[item]):
                rules.append({'antecedent': [item], 'consequent': consequent})

        # each entry: (antecedent_tuple, accumulated_log_state)
        # state starts at 0.0 and accumulates pairwise log-prob terms as antecedents grow
        antecedent_combinations = [((item,), 0.0) for item in frequent_items]

        # --- Phase 2: grow antecedents one item at a time ---
        # For each existing combination, try extending with every frequent item ranked after
        # the last element (canonical order → no duplicates). Before querying the model, the
        # frequency of the candidate antecedent is estimated from the already-stored single-item
        # log-probs as the geometric mean over all ordered pairs (a, b): P(b|a) * P(a).
        # Only candidates that pass antecedent_threshold are actually sent to the model.
        for r in range(2, self.config.max_antecedent_size + 1):
            n_pairs = r * (r - 1)
            extended_combinations = []

            for antecedent, state in antecedent_combinations:
                antecedent_sensors = {(st, sname) for st, sname, _ in antecedent}
                for new_item in frequent_items[item_rank[antecedent[-1]] + 1:]:
                    new_st, new_sname, _ = new_item
                    if (new_st, new_sname) in antecedent_sensors:
                        continue  # same sensor, different bin — invalid combination
                    new_state = state + sum(
                        self._item_log_prob(item_log_probs[existing], new_item)
                        + self._item_log_prob(item_log_probs[existing], existing)
                        + self._item_log_prob(item_log_probs[new_item], existing)
                        + self._item_log_prob(item_log_probs[new_item], new_item)
                        for existing in antecedent
                    )
                    if math.exp(new_state / n_pairs) >= self.config.antecedent_threshold:
                        extended_combinations.append((antecedent + (new_item,), new_state))

            if not extended_combinations:
                break

            # query the model for all surviving candidates and emit rules
            for start in tqdm(range(0, len(extended_combinations), bs),
                              desc=f"Apriori phase 2 (size={r}): {len(extended_combinations)} candidates"):
                batch_slice = extended_combinations[start:start + bs]
                batch_ants = [ant for ant, _ in batch_slice]
                graphs = [self.build_query_graph(ant) for ant in batch_ants]
                batch_data = PyGBatch.from_data_list(graphs).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    probs_batch = self.model(batch_data.x_dict, batch_data.edge_index_dict)
                probs_split = {
                    st: probs_batch[st].view(len(batch_ants), n_sensors_by_type[st], -1)
                    for st in self.gb.SENSOR_TYPES
                }
                for i, antecedent in enumerate(batch_ants):
                    probs = {st: probs_split[st][i] for st in self.gb.SENSOR_TYPES}
                    antecedent_sensors = {(st, sname) for st, sname, _ in antecedent}
                    if self.config.filter_consequents_by_hops:
                        reachable = set()
                        for ant_st, ant_sname, _ in antecedent:
                            reachable |= self.sensor_hop_map[self.gb.sensor_idx[(ant_st, ant_sname)]]
                    for st in self.gb.SENSOR_TYPES:
                        for row, s_idx in enumerate(self.gb.sensor_indices_by_type[st].tolist()):
                            _, sname = self.idx_to_sensor[s_idx]
                            if (st, sname) in antecedent_sensors:
                                continue
                            if self.config.filter_consequents_by_hops and s_idx not in reachable:
                                continue
                            for bin_idx in range(self.gb.n_bins_per_type[st]):
                                if probs[st][row, bin_idx].item() >= self.config.consequent_threshold:
                                    rules.append({'antecedent': list(antecedent), 'consequent': (st, sname, bin_idx)})

            antecedent_combinations = extended_combinations

        return rules

    def _item_log_prob(self, log_probs: dict, target_item: tuple) -> float:
        """Return log P(target_item) from a stored per-antecedent log-prob output dict."""
        st, sname, bin_idx = target_item
        row = self.sensor_to_row[st][self.gb.sensor_idx[(st, sname)]]
        return log_probs[st][row, bin_idx].item()

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

        # check if the probs of the antecedent items pass the antecedent thresholds
        for (st, sname, bin_idx) in antecedent:
            s_idx = self.gb.sensor_idx[(st, sname)]
            row = self.sensor_to_row[st][s_idx]
            if probs[st][row, bin_idx].item() < self.config.antecedent_threshold:
                return []

        # obtain the reachable set of sensors from the antecedent sensors
        if self.config.filter_consequents_by_hops:
            reachable = set()
            for (st, sname, _) in antecedent:
                s_idx = self.gb.sensor_idx[(st, sname)]
                reachable |= self.sensor_hop_map[s_idx]

        # obtain the list of consequents for the given antecedent items
        consequents = []
        for st in self.gb.SENSOR_TYPES:
            type_probs = probs[st]  # [n_sensors_of_type, n_bins]
            for row, s_idx in enumerate(self.gb.sensor_indices_by_type[st].tolist()):
                _, sname = self.idx_to_sensor[s_idx]
                if (st, sname) in antecedent_sensors: # antecedent items cannot form the consequents
                    continue
                if self.config.filter_consequents_by_hops and s_idx not in reachable: # if filter is applied, filter out unreachable consequents
                    continue
                for bin_idx in range(self.gb.n_bins_per_type[st]):
                    if type_probs[row, bin_idx].item() >= self.config.consequent_threshold:
                        consequents.append((st, sname, bin_idx))
        return consequents

    def build_query_graph(self, antecedent: list[tuple]):
        """Clone a random base graph and inject measured_by edges for the antecedent items.

        Non-antecedent sensors are treated according to the training masking strategy so
        the model sees the same input distribution at extraction time as during training:
          'remove'     — no edges (original behaviour)
          'all_bins'   — one edge per bin for each non-antecedent sensor
          'random_bin' — one edge to a random bin for each non-antecedent sensor
        """
        scenario = self._rng.choice(list(self.dataset.base_graphs.keys()))
        graph = self.dataset.base_graphs[scenario].clone().to(self.device)

        antecedent_sensors = {(st, sname) for st, sname, _ in antecedent}

        src, dst = [], []
        for (st, sname, bin_idx) in antecedent:
            src.append(self.gb.value_node_idx[(st, bin_idx)])
            dst.append(self.gb.sensor_idx[(st, sname)])

        # apply training masking strategy to non-antecedent sensors
        if self.config.match_masking_in_extraction and self.config.masking_strategy != 'remove':
            for (st, sname), s_idx in self.gb.sensor_idx.items():
                if (st, sname) in antecedent_sensors:
                    continue
                if self.config.masking_strategy == 'all_bins':
                    for bin_idx in range(self.gb.n_bins_per_type[st]):
                        src.append(self.gb.value_node_idx[(st, bin_idx)])
                        dst.append(s_idx)
                elif self.config.masking_strategy == 'random_bin':
                    bin_idx = self._rng.randint(0, self.gb.n_bins_per_type[st] - 1)
                    src.append(self.gb.value_node_idx[(st, bin_idx)])
                    dst.append(s_idx)

        ei = torch.tensor([src, dst], dtype=torch.long, device=self.device)
        graph[('value_node', 'measured_by', 'sensor')].edge_index = ei
        if self.gb.bidirectional:
            graph[('sensor', 'has_measure', 'value_node')].edge_index = ei.flip(0)

        if self.config.mask_sensor_to_virtual:
            antecedent_sensor_indices = {self.gb.sensor_idx[(st, sname)] for st, sname, _ in antecedent}
            for key in [('sensor', 'to_state', 'type_virtual'), ('sensor', 'to_global_state', 'global_virtual')]:
                if key in graph.edge_types:
                    sv_ei = graph[key].edge_index
                    keep = torch.tensor(
                        [s.item() in antecedent_sensor_indices for s in sv_ei[0]],
                        dtype=torch.bool, device=self.device,
                    )
                    graph[key].edge_index = sv_ei[:, keep]

        return graph

    def _get_all_items(self) -> list:
        """Return all possible (sensor_type, sensor_name, bin_idx) items."""
        items = []
        for (st, sname) in self.gb.sensor_idx:
            for bin_idx in range(self.gb.n_bins_per_type[st]):
                items.append((st, sname, bin_idx))
        return items
