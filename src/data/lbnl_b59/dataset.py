"""PyG Dataset for the LBNL Building 59 graph. Mirrors LeakDBDataset/DiTECWDNDataset's
process() structure, built up step by step -- steps 1-6 (tabular sensor DataFrame,
train/val split, per-sensor-type discretization, structural node attributes, graph
building via LBNLGraphBuilder, one-hot transaction DataFrame) are all implemented.
processed_file_names only lists what process() currently actually writes, extended as
later steps are added. One graph_{idx}.pt per timestamp (not listed in
processed_file_names, same as LeakDBDataset/DiTECWDNDataset's graph_{idx}.pt -- PyG's
own file-exists check only looks at the aggregate files listed there, not every
individual graph file).

Run from the project root to check steps 1-6:
    python -m src.data.lbnl_b59.dataset
"""
import json
import os
import random

import pandas as pd
import torch
from torch_geometric.data import Dataset

from src.data.discretizer import Discretizer
from src.data.lbnl_b59.graph_builder import LBNLGraphBuilder
from src.data.lbnl_b59.loader import LBNLLoader, WINDOW_START, WINDOW_END, EXCLUDED_CSV_FILES
from src.data.lbnl_b59.preprocessing import build_combined_frame


class LBNLDataset(Dataset):
    """Processing runs once. Delete the processed directory to reprocess."""

    def __init__(self, config, transform=None):
        self.config = config
        self.loader = LBNLLoader(config.raw_data_dir)

        # entities/relations are static (one building, no scenario dimension) --
        # computed fresh every __init__ (cheap: a single ttl parse), same as
        # LeakDBDataset/DiTECWDNDataset computing self.topology fresh every __init__
        # rather than persisting it.
        entities_relations = self.loader.load_entities_and_relations()
        self.entities, self.relations = entities_relations['entities'], entities_relations['relations']

        super().__init__(root=config.data_dir, transform=transform)

        with open(os.path.join(self.processed_dir, 'metadata.json')) as f:
            meta = json.load(f)
        self.n_bins_per_type = meta['n_bins_per_type']
        self.train_indices = meta['train_indices']
        self.val_indices = meta['val_indices']

        with open(os.path.join(self.processed_dir, 'raw_ids_by_type.json')) as f:
            self.raw_ids_by_type = json.load(f)
        with open(os.path.join(self.processed_dir, 'raw_id_owner.json')) as f:
            raw_id_owner = json.load(f)

        self.gb = LBNLGraphBuilder(
            self.entities, self.relations, self.raw_ids_by_type, self.n_bins_per_type, raw_id_owner,
            bidirectional=config.bidirectional_has_measure,
            virtual_node_mode=config.virtual_node_mode,
            onehot_bin_features=config.onehot_bin_features,
        )

        # LBNL has no scenario dimension (one building, one window) -- base_graphs keeps
        # the same {scenario: base} shape RuleExtractor/trainer expect from LeakDB/
        # DiTEC-WDN (see e.g. src/rules/extractor.py's self.dataset.base_graphs[scenario]),
        # just with a single placeholder key.
        self.base_graphs = {0: torch.load(os.path.join(self.processed_dir, 'base.pt'), weights_only=False)}

        # self.df is the one-hot dataset.parquet (not dataset_categorical.parquet), same as
        # LeakDBDataset/DiTECWDNDataset -- self.items/self.tensor are read directly by
        # src/rules/evaluator.py's RuleEvaluator, not just exported for the baselines.
        self.df = pd.read_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))
        item_cols = [c for c in self.df.columns if ';' in c]
        self.items = [(st, sname, int(b)) for st, sname, b in (c.split(';') for c in item_cols)]
        self.tensor = torch.tensor(self.df[item_cols].values, dtype=torch.bool)

    @property
    def raw_file_names(self):
        return []  # data is already local, no download needed

    @property
    def processed_file_names(self):
        return [
            'sensor_df.parquet', 'dataset_categorical.parquet', 'dataset.parquet', 'raw_ids_by_type.json',
            'raw_id_owner.json', 'metadata.json', 'attrs_Zone.parquet', 'attrs_VAV.parquet', 'base.pt',
        ]

    def process(self):
        # --- Step 1: build tabular sensor DataFrame (one row per timestamp, one column per raw_id) ---
        entities_relations = self.loader.load_entities_and_relations()
        entities, relations = entities_relations['entities'], entities_relations['relations']
        attachments = self.loader.load_csv_column_attachments(entities, relations)['attachments']

        # cols_by_file restricted to kept/attached columns only (load_csv_column_attachments()
        # already filters out EXCLUDED_CSV_FILES and unmatched columns)
        cols_by_file = self.loader.cols_by_file_from_attachments(attachments)
        raw_by_file = self.loader.load_all_raw(cols_by_file)

        combined = build_combined_frame(raw_by_file, cols_by_file, WINDOW_START, WINDOW_END, EXCLUDED_CSV_FILES)

        # build_combined_frame()'s columns are the raw CSV column names, not necessarily the
        # canonical point entity id -- 12 of 310 differ (the 3 CSV_TO_TTL_RENAME_PATTERNS files,
        # e.g. rtu_oa_fr.csv's "rtu_001_oa_flow_tn" -> point id "rtu_001_oa_fr"), so this rename
        # is still needed even though raw_ids are already globally unique on their own (verified
        # directly) and so need no sensor-type prefix to disambiguate.
        rename = {col: (a['point_id'] or col) for (fname, col), a in attachments.items()}
        df = combined.rename(columns=rename)

        # only use the specified data fraction
        num_transactions = int(len(df) * self.config.data_fraction)
        df = df.iloc[:num_transactions]

        # raw_ids_by_type: {sensor_type: [raw_id, ...]} -- the grouping Step 3's per-type
        # discretization pooling (and later LBNLGraphBuilder's sensor_idx) slice df by, derived
        # from cols_by_sensor_type() rather than re-parsed from column names.
        cols_by_type = self.loader.cols_by_sensor_type(attachments, entities)
        raw_ids_by_type = {
            sensor_type: [attachments[(fname, col)]['point_id'] or col for fname, col in pairs]
            for sensor_type, pairs in cols_by_type.items()
        }
        with open(os.path.join(self.processed_dir, 'raw_ids_by_type.json'), 'w') as f:
            json.dump(raw_ids_by_type, f)

        # raw_id_owner: {raw_id: owning structural entity id} -- a straight passthrough of
        # attachments['owner'], which load_csv_column_attachments() already resolved via (1)
        # the hasPoint/isPointOf relations (owner_of), for every point matched either exactly
        # or through a CSV_TO_TTL_RENAME_PATTERNS rename, and (2) the uft_hw_valve.csv
        # zone-prefix fallback for the 44 columns with no Brick point entity at all -- not
        # re-derived here, just reshaped from (file, column) keys to raw_id keys.
        raw_id_owner = {(a['point_id'] or col): a['owner'] for (fname, col), a in attachments.items()}
        with open(os.path.join(self.processed_dir, 'raw_id_owner.json'), 'w') as f:
            json.dump(raw_id_owner, f)

        # --- Step 2: shuffle and split train/val ---
        # No scenario dimension to track here (unlike LeakDBDataset/DiTECWDNDataset) -- there's
        # only ever this one committed window/building, so every row is implicitly the same
        # "scenario." df.index (the pad_to_window grid) is kept as a plain "timestamp" column
        # (not dropped, like DiTECWDNDataset's time_id) so a given graph_idx can still be traced
        # back to the actual timestamp it came from.
        df.index.name = 'timestamp'
        n = len(df)
        rng = random.Random(self.config.data_seed)  # kept separate from random_seed so varying the model seed across runs never resamples the split
        shuffled_idx = list(range(n))
        rng.shuffle(shuffled_idx)
        df = df.iloc[shuffled_idx].reset_index()

        n_train = int(n * self.config.train_ratio)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))
        df['split'] = ['train' if i < n_train else 'val' for i in range(n)]

        df.to_parquet(os.path.join(self.processed_dir, 'sensor_df.parquet'))

        # --- Step 3: fit per-sensor-type discretizers on training instances only ---
        # One pooled discretizer per sensor type (not LeakDBDataset's 3 hardcoded types or
        # DiTECWDNDataset's 7 signals) -- sliced via raw_ids_by_type rather than parsed from
        # column-name prefixes, since LBNL's raw_ids never needed a type prefix to begin with.
        train_df = df[df['split'] == 'train']

        disc_dfs = {}
        n_bins_per_type = {}
        for sensor_type, raw_ids in raw_ids_by_type.items():
            disc = Discretizer(n_bins=self.config.n_bins, pooled=True).fit(train_df[raw_ids])
            disc_dfs[sensor_type] = disc.transform(df[raw_ids])
            n_bins_per_type[sensor_type] = disc.get_n_bins()

        # categorical (non-one-hot) dataset: one column per raw_id, valued with its discretized
        # bin index -- the natural source format for baselines that expect discretized-but-not-
        # one-hot input (e.g. Aerial); the one-hot version (Step 6) is derived from exactly this.
        cat_df = pd.concat([df[['timestamp', 'split']], *disc_dfs.values()], axis=1)

        # Only the persisted file gets "{sensor_type};{raw_id}" column names, not cat_df itself
        # (Step 5 below keeps reading cat_df's bare raw_id columns via raw_ids_by_type) -- every
        # baseline export script (baselines/aerial/export.py, baselines/rulenaps_adapter/export.py)
        # identifies item columns generically by "contains a ';'", to tell them apart from
        # metadata columns (timestamp/split here, scenario/split/label for LeakDB/DiTEC-WDN)
        # without hardcoding per-dataset metadata column names. LBNL's raw_ids don't need the
        # type prefix for their own identity (already globally unique, see raw_ids_by_type/
        # sensor_idx), but this exported file still needs it to satisfy that cross-dataset
        # contract -- two different concerns, not in tension.
        cat_export_rename = {rid: f'{st};{rid}' for st, rids in raw_ids_by_type.items() for rid in rids}
        cat_df.rename(columns=cat_export_rename).to_parquet(os.path.join(self.processed_dir, 'dataset_categorical.parquet'))

        with open(os.path.join(self.processed_dir, 'metadata.json'), 'w') as f:
            json.dump({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'n_bins_per_type': n_bins_per_type,
            }, f)

        # --- Step 4: derive structural node attributes ---
        # {component_type: df} for the types with a derivable feature (Zone, VAV) --
        # RTU/HVAC_Zone/Floor/Building are left out, relying on
        # GraphBuilder.build_base()'s existing zero-placeholder fallback for any type
        # missing from this dict. The derived attributes are one hot encoded categorical
        # features and therefore normalizing/scaling is not applicable here like 
        # in the other datasets.
        structural_attrs = self.loader.derive_structural_attributes(entities, relations)
        for component_type, attrs_df in structural_attrs.items():
            attrs_df.to_parquet(os.path.join(self.processed_dir, f'attrs_{component_type}.parquet'))

        # --- Step 5: build and save the base graph, then one graph per timestamp ---
        # One base graph total (no per-scenario loop, unlike LeakDBDataset/DiTECWDNDataset --
        # see docs/experiment_plan.md), reused unchanged across every timestamp's graph.
        gb = LBNLGraphBuilder(
            entities, relations, raw_ids_by_type, n_bins_per_type, raw_id_owner,
            bidirectional=self.config.bidirectional_has_measure,
            virtual_node_mode=self.config.virtual_node_mode,
            onehot_bin_features=self.config.onehot_bin_features,
        )
        base = gb.build_base(structural_attrs)
        torch.save(base, os.path.join(self.processed_dir, 'base.pt'))

        for graph_idx, row in cat_df.iterrows():
            sensor_rows = {st: row[raw_ids] for st, raw_ids in raw_ids_by_type.items()}
            graph = gb.build(base, sensor_rows)
            graph.validate(raise_on_error=True)
            torch.save(
                graph,
                os.path.join(self.processed_dir, f'graph_{graph_idx}.pt'),
            )

        # --- Step 6: build and save one-hot transaction DataFrame (all rows) ---
        # Column names are "sensor_type;raw_id;bin_idx", matching LeakDBDataset/
        # DiTECWDNDataset's convention exactly (raw_id never itself contains ';', so this
        # is unambiguous) -- this is what RuleEvaluator/baselines/rulenaps_adapter's
        # export.py both read back into (sensor_type, raw_id, bin_idx) item tuples.
        hot_cols = {
            f'{st};{raw_id};{bin_idx}': (disc_dfs[st][raw_id].values == bin_idx)
            for st, raw_ids in raw_ids_by_type.items()
            for raw_id in raw_ids
            for bin_idx in range(n_bins_per_type[st])
        }
        hot_df = pd.DataFrame(hot_cols, index=df.index)

        dataset_df = pd.concat([df[['timestamp', 'split']], hot_df], axis=1)
        dataset_df.to_parquet(os.path.join(self.processed_dir, 'dataset.parquet'))

    def len(self):
        return len(self.df)

    def get(self, idx):
        return torch.load(
            os.path.join(self.processed_dir, f'graph_{idx}.pt'),
            weights_only=False,
        )


if __name__ == '__main__':
    from src.utils.config import Config

    config = Config(dataset_name='lbnl_b59')
    ds = LBNLDataset(config)

    df = pd.read_parquet(os.path.join(ds.processed_dir, 'sensor_df.parquet'))
    print(f"sensor df shape: {df.shape}")
    print(f"columns beyond raw_ids: {[c for c in df.columns if c in ('timestamp', 'split')]}")

    sensor_cols = [c for c in df.columns if c not in ('timestamp', 'split')]
    print(f"total NaN cells (sensor columns only): {df[sensor_cols].isna().sum().sum()}")

    with open(os.path.join(ds.processed_dir, 'raw_ids_by_type.json')) as f:
        raw_ids_by_type = json.load(f)
    all_raw_ids = [rid for ids in raw_ids_by_type.values() for rid in ids]
    print(f"\n{len(raw_ids_by_type)} sensor types, {len(all_raw_ids)} raw_ids "
          f"(matches df sensor columns: {sorted(all_raw_ids) == sorted(sensor_cols)})")

    with open(os.path.join(ds.processed_dir, 'metadata.json')) as f:
        meta = json.load(f)
    train_indices, val_indices = meta['train_indices'], meta['val_indices']
    print(f"\n{len(train_indices)} train / {len(val_indices)} val "
          f"({len(train_indices) / len(df):.1%} / {len(val_indices) / len(df):.1%})")
    print(f"df['split'] value_counts:\n{df['split'].value_counts()}")
    print(f"\nfirst/last of df['timestamp']: {df['timestamp'].min()} .. {df['timestamp'].max()} "
          f"(shuffled, so not a contiguous range within the df itself)")
    print(f"df['timestamp'] has {df['timestamp'].nunique()} unique values (should be {len(df)})")

    n_bins_per_type = meta['n_bins_per_type']
    cat_df = pd.read_parquet(os.path.join(ds.processed_dir, 'dataset_categorical.parquet'))
    print(f"\ndataset_categorical shape: {cat_df.shape}")
    cat_sensor_cols = [c for c in cat_df.columns if c not in ('timestamp', 'split')]
    print(f"NaN cells in categorical sensor columns: {cat_df[cat_sensor_cols].isna().sum().sum()}")
    print(f"\nn_bins_per_type (requested {config.n_bins}):")
    for st, ids in sorted(raw_ids_by_type.items(), key=lambda x: -len(x[1])):
        actual_bins = sorted(cat_df[f'{st};{ids[0]}'].dropna().unique())
        print(f"  {st:45s} n_bins={n_bins_per_type[st]:2d}  e.g. {ids[0]} bin range {min(actual_bins):.0f}..{max(actual_bins):.0f}")

    with open(os.path.join(ds.processed_dir, 'raw_id_owner.json')) as f:
        raw_id_owner = json.load(f)
    print(f"\nraw_id_owner: {len(raw_id_owner)} entries "
          f"(matches df sensor columns: {sorted(raw_id_owner) == sorted(sensor_cols)})")
    entities = ds.loader.load_entities_and_relations()['entities']
    print(f"all owners exist as entities: {all(owner in entities for owner in raw_id_owner.values())}")

    print("\n--- structural attributes ---")
    for component_type in ['Zone', 'VAV']:
        attrs_df = pd.read_parquet(os.path.join(ds.processed_dir, f'attrs_{component_type}.parquet'))
        print(f"  {component_type}: shape={attrs_df.shape}  columns={list(attrs_df.columns)}")
        for prefix in ('floor_', 'rtu_group_'):
            group_cols = [c for c in attrs_df.columns if c.startswith(prefix)]
            if group_cols:
                print(f"    exactly one '{prefix}*' column set per row: {(attrs_df[group_cols].sum(axis=1) == 1).all()}")

    print("\n--- graphs (step 5) ---")
    print(f"ds.len(): {ds.len()}  (matches cat_df rows: {ds.len() == len(cat_df)})")
    print(f"ds.base_graphs[0]: {ds.base_graphs[0]}")
    g0 = ds.get(0)
    g0.validate(raise_on_error=True)
    print(f"\nds.get(0): {g0}")
    n_measured_by_0 = g0[('value_node', 'measured_by', 'sensor')].edge_index.shape[1]
    print(f"measured_by edges: {n_measured_by_0} (expect n_sensors={ds.gb.n_sensors}, since dataset_categorical has 0 NaN)")
    g_last = ds.get(len(ds) - 1)
    g_last.validate(raise_on_error=True)
    print(f"ds.get({len(ds) - 1}) also loads and validates fine: {g_last[('value_node', 'measured_by', 'sensor')].edge_index.shape[1]} measured_by edges")

    print("\n--- one-hot dataset (step 6) ---")
    print(f"ds.df (dataset.parquet) shape: {ds.df.shape}")
    expected_n_items = sum(len(raw_ids_by_type[st]) * n_bins_per_type[st] for st in raw_ids_by_type)
    print(f"n items: {len(ds.items)}  (expect {expected_n_items}): {len(ds.items) == expected_n_items}")
    print(f"ds.tensor shape: {tuple(ds.tensor.shape)}  (expect ({len(ds.df)}, {expected_n_items}))")
    print(f"example items: {ds.items[:3]}")

    # each transaction has exactly 1 active bin per sensor (one-hot), so exactly n_sensors
    # active items per row
    active_per_row = ds.tensor.sum(dim=1)
    print(f"active items per transaction all == n_sensors ({ds.gb.n_sensors}): {(active_per_row == ds.gb.n_sensors).all().item()}")

    # cross-check: ds.items/ds.tensor agree with dataset_categorical.parquet's bin indices
    item_to_col = {item: col for col, item in enumerate(ds.items)}
    st0, sname0 = ds.items[0][0], ds.items[0][1]
    cat_bin = cat_df.loc[0, f'{st0};{sname0}']
    hot_bin = next(b for (st, sn, b) in ds.items if st == st0 and sn == sname0 and ds.tensor[0, item_to_col[(st, sn, b)]])
    print(f"row 0, sensor ({st0}, {sname0}): categorical bin={int(cat_bin)}  one-hot active bin={hot_bin}  match={int(cat_bin) == hot_bin}")
