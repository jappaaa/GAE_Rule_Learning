import os
import re

import pandas as pd
import rdflib
from rdflib.namespace import split_uri

# CSVs excluded because their coverage doesn't overlap the chosen analysis
# window (2020-05-01 -- 2020-12-31) -- see docs/experiment_plan.md. ashp_cw.csv/
# ashp_meter.csv/ashp_hw.csv do have data inside the window (from 2020-08-01
# onward) but are excluded as a subsystem for consistency; hp_hws_temp.csv's
# native data ends 2020-05-02 (barely into the window) and occ.csv's ends
# 2019-02-21 (entirely before it) -- both have effectively/exactly zero overlap.
EXCLUDED_CSV_FILES = {
    'ashp_cw.csv', 'ashp_meter.csv', 'ashp_hw.csv', 'hp_hws_temp.csv', 'occ.csv',
}

# Typos/artifacts on the ttl side (confirmed by inspecting the raw entity ids).
# Checked against the full URI string, not the raw ttl file itself (data/
# stays an untouched copy of what was downloaded), and before splitting off
# the local name, not after, since 'lig_S_+' contains a '+', which isn't a
# valid XML NCName character and makes rdflib's split_uri() raise before a
# local name is even produced. Each `good` here is the *entire* correct local
# name, not a fragment (unlike CSV_TO_TTL_RENAME_PATTERNS below) -- verified
# by checking `bad` is the full suffix after '#' in every URI it matches, so
# _local_name can return `good` directly without ever calling split_uri().
TTL_TYPO_FIXES = [
    ('lig_S_+', 'lig_S'),                              # stray footnote marker baked into the ttl id
    ('occ_forth_south', 'occ_fourth_south'),           # misspelling in the ttl
    ('rtu_021_econ_stpt_tn', 'rtu_002_econ_stpt_tn'),  # RTU "021" doesn't exist (only 001-004), all points of RTU02 start with rtu_002 except rtu_021_econ_stpt_tn -> likely typo
]

# CSV column substrings -> ttl point-id substrings, for points the ttl names
# differently from the raw BAS tag the CSV uses. Confirmed by inspecting every
# rtu_001_* point id directly, then checking the same pattern holds for RTU 002-004.
CSV_TO_TTL_RENAME_PATTERNS = [
    ('oadmpr_pct', 'oa_damper'),
    ('fltrd_gnd_lvl_plenum_press_tn', 'fltrd_gnd_plenum_press_tn'),
    ('oa_flow_tn', 'oa_fr'),  # rtu_oa_fr.csv's rtu_00X_oa_flow_tn -> ttl's rtu_00X_oa_fr
]

# uft_hw_valve.csv's zone_<n>_hw_valve columns have no ttl point entity modeled
# by the authors. However, uft_fan_spd.csv, already establishes the convention 
# of attaching UFT points directly to the owning Zone entity (zone_016 hasPoint zone_016_fan_spd), 
# so we apply that same convention here rather than inventing a new one. This is
# deliberately a one-off, not a generic prefix-fallback mechanism -- the other
# currently-unmatched files (rtu_oa_t/sa_fr/sa_t_sp.csv, hp_hws_temp.csv,
# site_weather.csv's dew point) are left unmatched/unattached on purpose, since
# the dataset authors chose not to model them either.
ZONE_PREFIX_FALLBACK_FILE = {'uft_hw_valve.csv'}
ZONE_PREFIX_RE = re.compile(r'^(zone_\d+)_')

# Analysis window -- see docs/experiment_plan.md. Every kept file is
# monotonic with zero duplicate timestamps inside this range (verified directly;
# the duplicates/one out-of-order jump some files have near the 2018/2021 edges
# of their full 3-year span fall entirely outside it), so clipping to this
# window first needs no separate dedup/sort step.
WINDOW_START = pd.Timestamp('2020-05-01')
WINDOW_END = pd.Timestamp('2020-12-31')

# Common grid every file gets resampled onto -- coarser than any file's own
# native rate (1/10/15 min depending on file), so one shared constant instead
# of repeating the literal wherever a file gets resampled.
RESAMPLE_FREQ = '15min'


class LBNLLoader:
    """Loads the Brick model and sensor CSVs for the LBNL Building 59 dataset."""

    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        self.ttl_path = os.path.join(raw_data_dir, "Building_59", "Bldg59_w_occ Brick model.ttl")
        self.csv_dir = os.path.join(raw_data_dir, "Building_59", "Bldg59_clean data")

    @staticmethod
    def _local_name(uri: rdflib.URIRef) -> str:
        s = str(uri)
        for bad, good in TTL_TYPO_FIXES:      # check for known ttl-side typos before splitting, not after (see comment above TTL_TYPO_FIXES)
            if bad in s:
                return good
        return split_uri(s)[1]

    def load_entities_and_relations(self) -> dict:
        """Parse the Brick ttl into a flat entity registry and a typed relation list.

        Returns:
            {"entities": {id: {"type": brick_class_local_name}},
             "relations": [(subject_id, predicate, object_id), ...]}
        Every rdf:type'd subject becomes an entity (equipment, location, and point
        instances alike — e.g. Zone, VAV, Rooftop_Unit as well as individual sensor/
        setpoint points); which of those end up as graph nodes vs. CSV-column
        attachment points is decided later, not during this parse.
        """
        g = rdflib.Graph()
        g.parse(self.ttl_path, format='turtle')

        entities = {}
        for s, _, o in g.triples((None, rdflib.RDF.type, None)):    # only loop over the triples that specify the class of an entity
            eid = self._local_name(s)
            etype = self._local_name(o)
            entities.setdefault(eid, {"type": []})["type"].append(etype)    # accounts for the possiblity that entities have multiple types

        relations = [
            (self._local_name(s), self._local_name(p), self._local_name(o))    
            for s, p, o in g
            if p != rdflib.RDF.type
        ]

        return {"entities": entities, "relations": relations}

    @staticmethod
    def classify_entities(entities: dict, relations: list) -> tuple:
        """Split entity ids into (structural_ids, point_ids).

        A "point" is a leaf measurement/setpoint (becomes a CSV-column attachment
        target, not its own graph node): identified as any entity that is the
        object of a hasPoint relation or the subject of an isPointOf relation —
        the two directions Brick uses interchangeably in this ttl (equipment
        hasPoint point vs. point isPointOf equipment). Everything else (Zone,
        VAV, Rooftop_Unit, Floor, Building, HVAC_Zone, Wifi_AP) is "structural"
        and becomes a graph node.
        """
        point_ids = (
            {o for _, p, o in relations if p == 'hasPoint'} |
            {s for s, p, _ in relations if p == 'isPointOf'}
        )
        structural_ids = set(entities.keys()) - point_ids
        return structural_ids, point_ids

    @staticmethod
    def _owner_of(relations: list) -> dict:
        """{point_id: owning_structural_entity_id}, inverting hasPoint/isPointOf."""
        owner_of = {}
        for s, p, o in relations:
            if p == 'hasPoint':
                owner_of[o] = s
            elif p == 'isPointOf':
                owner_of[s] = o
        return owner_of

    def load_csv_column_attachments(self, entities: dict, relations: list) -> dict:
        """Attach every kept CSV column to the structural entity it measures.

        Tries, in order: (1) exact match against the point id (ttl-side typos
        already fixed in _local_name, so owner_of's ids are clean); (2) the
        known CSV-tag -> ttl-name rename patterns; (3) for uft_hw_valve.csv
        only, attach directly to the zone_<n> entity named in the column,
        matching the convention its sibling file (uft_fan_spd.csv) already
        uses in the ttl. Columns matching none of these are returned as
        "unmatched" -- left unattached on purpose, since the dataset authors
        chose not to model them.

        Returns {"attachments": {(file, column): {"owner": structural_entity_id,
        "point_id": point_entity_id}}, "unmatched": [(file, column), ...]}.
        point_id is None only for the uft_hw_valve.csv zone-prefix-fallback case,
        where no Brick point entity exists at all -- everywhere else it's the
        actual matched point id (not necessarily == column, since a rename
        pattern may have been applied), so callers can look up e.g. the point's
        Brick type via entities[point_id]['type'] without redoing the matching.
        """
        owner_of = self._owner_of(relations)

        kept_files = sorted(
            f for f in os.listdir(self.csv_dir) if f.endswith('.csv') and f not in EXCLUDED_CSV_FILES
        )

        attachments, unmatched = {}, []
        for fname in kept_files:
            cols = pd.read_csv(os.path.join(self.csv_dir, fname), nrows=0).columns.tolist()
            cols = [c for c in cols if 'date' not in c.lower() and not c.startswith('Unnamed')]
            for col in cols:
                if col in owner_of:
                    attachments[(fname, col)] = {"owner": owner_of[col], "point_id": col}
                    continue
                renamed = col
                # if there is no match with the original col and ttl, try renamed patterns
                for csv_sub, ttl_sub in CSV_TO_TTL_RENAME_PATTERNS:
                    renamed = renamed.replace(csv_sub, ttl_sub)
                if renamed in owner_of:
                    attachments[(fname, col)] = {"owner": owner_of[renamed], "point_id": renamed}
                    continue
                # apply the attachment of the uft_hw_valve.csv columns to their corresponding zones, not modeled by the authors
                zone_match = ZONE_PREFIX_RE.match(col) if fname in ZONE_PREFIX_FALLBACK_FILE else None
                if zone_match and zone_match.group(1) in entities:
                    attachments[(fname, col)] = {"owner": zone_match.group(1), "point_id": None}
                else:
                    unmatched.append((fname, col))

        return {"attachments": attachments, "unmatched": unmatched}

    @staticmethod
    def find_unused_points(point_ids: set, attachments: dict) -> set:
        """Point entities (from classify_entities) with no CSV column attached at
        all -- e.g. because their whole source file is in EXCLUDED_CSV_FILES.
        These must never become sensor nodes, since no data exists for them."""
        matched_point_ids = {a["point_id"] for a in attachments.values() if a["point_id"] is not None}
        return point_ids - matched_point_ids

    @staticmethod
    def cols_by_file_from_attachments(attachments: dict) -> dict:
        """{file: [value columns]} restricted to the columns load_csv_column_attachments()
        actually matched to a Brick entity, for the committed/kept file set only -- the
        shape load_all_raw()/build_combined_frame() need. See cols_by_sensor_type() for
        the alternative grouping (by Brick point type, not by file)."""
        cols_by_file = {}
        for fname, col in attachments:
            cols_by_file.setdefault(fname, []).append(col)
        return cols_by_file

    @staticmethod
    def cols_by_sensor_type(attachments: dict, entities: dict) -> dict:
        """{sensor_type: [(file, column), ...]}, grouping load_csv_column_attachments()'s
        attachments by the Brick type of their point entity -- the actual physical sensor
        type (e.g. 'Cooling_Temperature_Setpoint', 'CO2_Sensor'), not which CSV file a
        column happens to live in. This is the discretization-pooling key and the graph's
        sensor/value-node-type key -- confirmed CSV-file grouping doesn't match it exactly
        (rtu_fan_spd.csv mixes 2 types, Return_Air_Fan_Speed and Supply_Air_Fan_Speed;
        Zone_Air_Temperature_Sensor spans zone_temp_exterior.csv AND
        zone_temp_interior.csv). Every attached point entity has exactly 1 Brick type
        (checked directly), so entities[point_id]['type'][0] is used as a plain string key.

        Columns with no point_id at all (uft_hw_valve.csv's zone-prefix-fallback case --
        no Brick point entity exists for them) fall back to their own file name as the
        group key, since there's no type to group them by.
        """
        groups = {}
        for (fname, col), a in attachments.items():
            pid = a["point_id"]
            key = fname if pid is None else entities[pid]["type"][0]
            groups.setdefault(key, []).append((fname, col))
        return groups

    @staticmethod
    def derive_structural_attributes(entities: dict, relations: list) -> dict:
        """{component_type: df}, one one-hot-encoded node-feature DataFrame per
        structural entity type with a derivable feature -- built only from direct
        Brick relations (isPartOf, hasLocation, isFedBy), never transitive chains
        (deeper structure is left for the GNN's own message passing to pick up).
        RTU/HVAC_Zone/Floor/Building have no derivable feature and are deliberately
        left out of the returned dict entirely -- GraphBuilder.build_base() already
        assigns any type missing from its attrs dict a torch.zeros(n, 1) placeholder
        (see docs/experiment_plan.md for a possible ones-vector alternative).

        Returns:
            'Zone': indexed by zone id, one-hot 'floor_<Floor>' (via isPartOf, every
                zone has exactly one -- verified) + one-hot 'rtu_group_<HVAC_Zone>'
                (via hasLocation, with an explicit 'rtu_group_none' category for the
                16 interior zones that have no hasLocation relation at all).
            'VAV': indexed by VAV id, one-hot 'rtu_group_<Rooftop_Unit>' (via
                isFedBy, every VAV has exactly one -- verified).
        Zone's and VAV's rtu_group categories live in different Brick id spaces
        (HVAC_Zone ids like 'RTU01_Zone' vs. Rooftop_Unit ids like 'RTU01') and are
        never compared across types, so this is harmless.
        """

        # here we can use 'in' for the check because e['type'] returns a list to potentially support multiple types 
        # if stored as a string this would return point entity types such as Zone_Air_Temperature_Sensor wrongly
        zone_ids = sorted(eid for eid, e in entities.items() if 'Zone' in e['type'])
        vav_ids = sorted(eid for eid, e in entities.items() if 'VAV' in e['type'])

        
        zone_floor = {s: o for s, p, o in relations if p == 'isPartOf' and s in zone_ids}
        zone_rtu_group = {s: o for s, p, o in relations if p == 'hasLocation' and s in zone_ids}
        vav_rtu_group = {s: o for s, p, o in relations if p == 'isFedBy' and s in vav_ids}

        zone_df = pd.get_dummies(pd.DataFrame({
            'floor': [zone_floor[zid] for zid in zone_ids],
            'rtu_group': [zone_rtu_group.get(zid, 'none') for zid in zone_ids],
        }, index=zone_ids), prefix=['floor', 'rtu_group'])

        vav_df = pd.get_dummies(pd.DataFrame({
            'rtu_group': [vav_rtu_group[vid] for vid in vav_ids],
        }, index=vav_ids), prefix=['rtu_group'])

        return {'Zone': zone_df, 'VAV': vav_df}

    def load_all_raw(self, columns_by_file: dict) -> dict:
        """{filename: df}, each exactly as logged for that file's
        columns_by_file[fname] columns -- date column parsed and set as the
        index, nothing else: no sort, no window clip, no dedup, no
        backward-jump removal. The single entry point every other loading/
        processing step (see the stage functions below) starts from.

        columns_by_file: {file: [value columns]} -- e.g. cols_by_file_all()
        or cols_by_file_from_attachments() from exploration.py.
        """
        raw = {}
        for fname, columns in columns_by_file.items():
            path = os.path.join(self.csv_dir, fname)
            date_col = pd.read_csv(path, nrows=0).columns[0]
            df = pd.read_csv(path, usecols=[date_col] + columns)
            df[date_col] = pd.to_datetime(df[date_col])
            raw[fname] = df.set_index(date_col)
        return raw

if __name__ == '__main__':
    from collections import Counter

    loader = LBNLLoader(os.path.join('data', 'lbnl_b59', 'raw'))
    result = loader.load_entities_and_relations()
    entities, relations = result['entities'], result['relations']

    print(f"n entities: {len(entities)}")
    print(f"n relations: {len(relations)}")

    print("\n--- relation predicate counts ---")
    for pred, count in Counter(p for _, p, _ in relations).most_common():
        print(f"  {count:4d}  {pred}")

    structural_ids, point_ids = LBNLLoader.classify_entities(entities, relations)
    print(f"\nstructural entities: {len(structural_ids)}  |  point entities: {len(point_ids)}  |  overlap: {len(structural_ids & point_ids)}")

    print("\n--- structural entity types ---")
    for etype, count in Counter(entities[eid]['type'][0] for eid in structural_ids).most_common():
        print(f"  {count:4d}  {etype}")

    print("\n--- point entity types ---")
    for etype, count in Counter(entities[eid]['type'][0] for eid in point_ids).most_common():
        print(f"  {count:4d}  {etype}")

    csv_result = loader.load_csv_column_attachments(entities, relations)
    attachments, unmatched = csv_result['attachments'], csv_result['unmatched']
    print(f"\nCSV columns attached: {len(attachments)}  |  unmatched: {len(unmatched)}")

    # These have no corresponding point/sensor nodes in the KG and remain unused
    print("\n--- unmatched columns by file ---")
    by_file = {}
    for fname, col in unmatched:
        by_file.setdefault(fname, []).append(col)
    for fname, cols in by_file.items():
        print(f"  {fname} ({len(cols)}): {cols[:3]}{'...' if len(cols) > 3 else ''}")

    # Confirmed (by scanning every csv file, including excluded ones, for columns
    # matching these point ids) that all 8 unused points trace back to
    # ashp_cw.csv/ashp_hw.csv's heat-pump water-side points and occ.csv, both
    # already in EXCLUDED_CSV_FILES -- see docs/experiment_plan.md.
    unused_points = loader.find_unused_points(point_ids, attachments)
    print(f"\npoint entities with no linked CSV column at all: {len(unused_points)}")
    for etype, count in Counter(entities[eid]['type'][0] for eid in unused_points).most_common():
        print(f"  {count:4d}  {etype}")
