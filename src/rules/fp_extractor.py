import time

import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from tqdm import tqdm

from src.data.dataset import LeakDBDataset
from src.utils.config import Config


class FPExtractor:
    """Extract association rules from the transaction tensor using FP-growth.

    Produces rules in the same format as RuleExtractor so they can be passed
    directly to RuleEvaluator: each rule is a dict with
        'antecedent': list of (sensor_type, sensor_name, bin_idx) tuples
        'consequent': (sensor_type, sensor_name, bin_idx) tuple
    """

    def __init__(self, dataset: LeakDBDataset, config: Config):
        self.dataset = dataset
        self.config = config

    def extract(self) -> list[dict]:
        items = self.dataset.items  # list of (st, sname, bin_idx)

        # mlxtend fpgrowth expects a bool DataFrame; column names are used as item labels
        item_cols = [c for c in self.dataset.df.columns if c not in ('scenario', 'split', 'label')]
        bool_df = self.dataset.df[item_cols].astype(bool)

        print(f"  Mining frequent itemsets (min_support={self.config.min_support_fp}, max_len={self.config.max_antecedent_size + 1})...")
        t0 = time.time()
        freq_sets = fpgrowth(bool_df, min_support=self.config.min_support_fp, use_colnames=True, max_len=self.config.max_antecedent_size + 1)
        print(f"  Found {len(freq_sets)} frequent itemsets in {time.time() - t0:.1f}s")

        if freq_sets.empty:
            return []

        print(f"  Generating association rules (min_confidence={self.config.min_confidence_fp})...")
        t0 = time.time()
        raw_rules = association_rules(freq_sets, metric='confidence', min_threshold=self.config.min_confidence_fp, num_itemsets=len(freq_sets))
        print(f"  Generated {len(raw_rules)} raw rules in {time.time() - t0:.1f}s")

        print("  Filtering and converting rules...")
        rules = []
        for _, row in tqdm(raw_rules.iterrows(), total=len(raw_rules), desc="  Rules"):
            ant_cols = list(row['antecedents'])
            con_cols = list(row['consequents'])

            # require exactly one consequent item
            if len(con_cols) != 1:
                continue

            # enforce max_antecedent_size
            if len(ant_cols) > self.config.max_antecedent_size:
                continue

            # map column names back to item tuples
            col_to_item = {c: item for c, item in zip(item_cols, items)}
            antecedent = [col_to_item[c] for c in ant_cols]
            consequent = col_to_item[con_cols[0]]

            # reject antecedents where the same sensor appears with two different bins
            sensors_seen = set()
            valid = True
            for (st, sname, _) in antecedent:
                if (st, sname) in sensors_seen:
                    valid = False
                    break
                sensors_seen.add((st, sname))
            if not valid:
                continue

            # antecedent sensor must not be the same as consequent sensor
            con_sensor = (consequent[0], consequent[1])
            if con_sensor in sensors_seen:
                continue

            rules.append({'antecedent': antecedent, 'consequent': consequent})

        return rules
