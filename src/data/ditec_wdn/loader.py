"""Loads topology, static attributes, and sensor data for one DiTEC-WDN network from the
Hugging Face dataset `rugds/ditec-wdn`. Mirrors the role of LeakDBLoader (src/data/leakdb/loader.py)
but the data source is HF parquet shards rather than local EPANET .inp/.csv files.

Run from the project root to check step 1 (file discovery) in isolation:
    python -m src.data.ditec_wdn.loader
"""
import json
import re
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPO_ID = 'rugds/ditec-wdn'

# matches e.g. "CTOWN_1GB_24H/node_pressure-0-dynamic_output.parquet"
FILENAME_RE = re.compile(r'^(?P<network>[^/]+)/(?P<component_parameter>.+)-(?P<shard>\d+)-(?P<type>\w+)\.parquet$')

# node/link component types as they appear as keys in the `onames` metadata dict -- verified
# empirically across all 36 networks' footers; a given network only has some of these (e.g. not
# every network has pumps), which is fine since we only look up the keys that exist. psv is kept
# even though it wasn't observed in any of the 36 networks, since it's documented on the HF
# dataset card as a possible component type.
NODE_TYPES = ('junction', 'reservoir', 'tank')
LINK_TYPES = ('pipe', 'headpump', 'powerpump', 'prv', 'psv', 'fcv', 'tcv')
# headpump/powerpump are both EPANET pump variants (head-curve-driven vs power-driven) that play
# the same structural role in the graph (a link adding head between two nodes), so they collapse
# to one "pump" link type here. Their parameters are NOT interchangeable though -- e.g. CTOWN's
# headpump okeys are ['initial_status', 'energy_price', 'base_speed', 'pump_curve_name'] while
# ky10's powerpump okeys are ['initial_status', 'base_speed', 'power'] -- so load_attributes_as_df()
# must still look up okeys/odims by the real onames key (headpump/powerpump), never by "pump".
LINK_TYPE_ALIASES = {'headpump': 'pump', 'powerpump': 'pump'}


class DiTECWDNLoader:
    """Loads one DiTEC-WDN network's topology and static/dynamic parquet shards from HF."""

    def __init__(self, network: str, repo_id: str = REPO_ID):
        self.network = network
        self.repo_id = repo_id
        self.shard_groups = self._discover_shards()
        self._metadata_cache = None
        self._skip_sets_cache = None

    def _discover_shards(self) -> dict[tuple[str, str], list[str]]:
        """Return {(component_parameter, type): [file_paths, one per shard]} for this network.

        One list_repo_files call lists the whole repo (all 36 networks) in one shot --
        there's no per-network-scoped listing endpoint, but it's just filenames/sizes, so
        this is cheap. Filtering down to self.network happens client-side.
        """
        api = HfApi()
        all_files = api.list_repo_files(repo_id=self.repo_id, repo_type='dataset')

        groups = defaultdict(list)
        for path in all_files:
            m = FILENAME_RE.match(path)
            if m is None or m.group('network') != self.network:
                continue
            key = (m.group('component_parameter'), m.group('type'))
            groups[key].append(path)

        return dict(groups)

    def _read_metadata(self) -> dict:
        """Read the parquet footer's `attrs` key-value metadata (JSON blob with
        onames/adj_list/skip_names/...), read footer only without downloading whole file.

        This blob is duplicated identically across every shard of a network, so any one file
        works -- we just pick whichever one _discover_shards happened to find first. Cached
        since load_topology()/load_attributes_as_df()/load_sensor_data() all need it.
        """
        if self._metadata_cache is None:
            any_path = next(iter(self.shard_groups.values()))[0]
            fs = HfFileSystem()
            with fs.open(f'hf://datasets/{self.repo_id}/{any_path}', 'rb') as f:
                metadata = pq.ParquetFile(f).metadata.metadata
            self._metadata_cache = json.loads(metadata[b'attrs'])
        return self._metadata_cache

    def _skip_sets(self) -> tuple[set, set]:
        """Return (skip_node_names, skip_link_ids): the skip_names nodes and every link
        touching one of them (derived from adj_list). Cached, since load_topology() and
        load_attributes_as_df()/load_sensor_data() all call this and the result never
        changes for a given network. Shared so skip_names removal is applied consistently
        everywhere, matching the official DiTEC PyG loader's own behaviour
        (ditec_wdn_dataset/hf/dataset.py: load_roots() rebuilds adj_list and renumbers
        edge_index with skipped nodes and their incident edges removed).
        """
        if self._skip_sets_cache is None:
            metadata = self._read_metadata()
            skip_names = set(metadata['skip_names'])
            skip_link_ids = {
                link_id for start_node, end_node, link_id in metadata['adj_list']
                if start_node in skip_names or end_node in skip_names
            }
            self._skip_sets_cache = (skip_names, skip_link_ids)
        return self._skip_sets_cache

    def load_topology(self) -> dict:
        """Return {"nodes": {id: {"type": ...}}, "links": {id: {"type": ..., "start_node": ..., "end_node": ...}}}.

        Matches the generalized topology contract (LeakDBLoader.load_topology() will be
        reshaped to the same format once GraphBuilder is generalized): node types are
        junction/reservoir/tank, link types are pipe/pump/prv/psv/fcv/tcv.

        skip_names components are dropped, along with every link touching one of them.
        """
        metadata = self._read_metadata()
        onames = metadata['onames']
        skip_names, skip_link_ids = self._skip_sets()

        nodes = {}
        for type_ in NODE_TYPES:
            for name in onames.get(type_, []):
                if name not in skip_names:
                    nodes[name] = {'type': type_}

        link_type_of = {}
        for type_ in LINK_TYPES:
            for name in onames.get(type_, []):
                link_type_of[name] = LINK_TYPE_ALIASES.get(type_, type_)

        links = {}
        for start_node, end_node, link_id in metadata['adj_list']:
            if link_id in skip_link_ids:
                continue
            links[link_id] = {
                'type': link_type_of[link_id],
                'start_node': start_node,
                'end_node': end_node,
            }

        return {'nodes': nodes, 'links': links}

    def load_attributes_as_df(self, scenarios: list) -> dict:
        """Return {component_type: df}, one DataFrame per component type present in this
        network (using the real onames key, e.g. "junction", "headpump", "powerpump" --
        never the "pump" alias, since headpump/powerpump have different attribute columns).

        Each df is indexed by (instance_id, scenario_id) with one column per static attribute
        of that type (e.g. df["junction"] has columns "elevation", "base_demand"). Only
        static_input groups are read (curve_input stays deferred, see docs/experiment_plan.md).
        skip_names components (and links touching them) are dropped, matching load_topology().
        """
        skip_names, skip_link_ids = self._skip_sets()
        scenario_ids = [float(s) for s in scenarios]
        fs = HfFileSystem()

        per_type_frames = defaultdict(list)
        for (component_parameter, file_type), shard_paths in self.shard_groups.items():
            if file_type != 'static_input':
                continue
            component_type, attribute = component_parameter.split('_', 1)  # e.g. "headpump_pump_curve_name" -> ("headpump", "pump_curve_name")

            paths = [f'hf://datasets/{self.repo_id}/{p}' for p in shard_paths]  # the paths of all shard belongin to this component_type X attribute combination
            wide_df = pq.read_table(paths, filesystem=fs, filters=[('scenario_id', 'in', scenario_ids)]).to_pandas()    # using the filter allows us to only access the shards and row groups necessary for the specified scenarios
            
            skip_set = skip_names if component_type in NODE_TYPES else skip_link_ids
            instance_cols = [c for c in wide_df.columns if c != 'scenario_id' and c not in skip_set]

            long_df = wide_df.melt(id_vars='scenario_id', value_vars=instance_cols, var_name='id', value_name=attribute)    # transform from wide format, into one where the columns become rows exept scenario_id
            long_df = long_df.set_index(['id', 'scenario_id'])
            per_type_frames[component_type].append(long_df)

        df_per_component_type = {component_type: pd.concat(frames, axis=1) for component_type, frames in per_type_frames.items()}

        return df_per_component_type

    def load_sensor_data(self, scenarios: list) -> dict:
        """Return {signal: df} for each of the 7 dynamic_output signals (pressure,
        demand, head, flowrate, velocity, headloss, friction_factor). Each df is wide --
        indexed by (scenario_id, time_id), one column per node/link instance name -- matching
        LeakDBLoader.load_sensor_data()'s own shape (sd['pressures']/sd['flows']/...), just
        generalized to multiple scenarios via the extra scenario_id index level (LeakDB loads
        one scenario at a time, so never needed that level at all).

        dynamic_output files are named generically "node_<signal>"/"link_<signal>" (never per
        sub-type like static_input's files), so a plain split('_', 1) is enough here, same as
        load_attributes_as_df() uses for static_input's per-subtype names.

        skip_names components (and links touching them) are dropped, matching load_topology().
        """
        skip_names, skip_link_ids = self._skip_sets()
        scenario_ids = [float(s) for s in scenarios]
        fs = HfFileSystem()

        dfs = {}
        for (component_parameter, file_type), shard_paths in self.shard_groups.items():
            if file_type != 'dynamic_output':
                continue
            group, signal = component_parameter.split('_', 1)  # e.g. "node_pressure" -> ("node", "pressure")

            paths = [f'hf://datasets/{self.repo_id}/{p}' for p in shard_paths]
            wide_df = pq.read_table(paths, filesystem=fs, filters=[('scenario_id', 'in', scenario_ids)]).to_pandas()

            skip_set = skip_names if group == 'node' else skip_link_ids
            instance_cols = [c for c in wide_df.columns if c not in ('scenario_id', 'time_id') and c not in skip_set]

            dfs[signal] = wide_df.set_index(['scenario_id', 'time_id'])[instance_cols]

        return dfs

    def load_sensor_df(self, scenarios: list, data_fraction: float = 1.0) -> pd.DataFrame:
        """Build one combined sensor DataFrame covering every requested scenario, with columns
        named "{signal};{instance}" (e.g. "pressure;J511", "flowrate;P100") -- the shape
        LeakDBDataset.process()'s Step 1 currently builds by hand from load_sensor_data().

        Only the column-wise concat (renaming + combining the different signals) is needed
        here, no row-wise concat across scenarios: load_sensor_data() already returns every
        requested scenario in one call. Rows are indexed by (scenario_id, time_id).

        data_fraction trims to the first N timesteps of that one scenario's own timeline --
        only meaningful with a single scenario (Config enforces this: data_fraction < 1.0 with
        multiple scenarios raises there), since with multiple scenarios, reducing data volume
        should be done via the scenarios list itself, not by independently truncating each
        one's timesteps.
        """
        sensor_dfs = self.load_sensor_data(scenarios)
        renamed = [df.rename(columns=lambda c: f'{signal};{c}') for signal, df in sensor_dfs.items()]
        # concat(axis=1) here leaves every column as its own internal "block" instead of merging
        # same-dtype columns into one -- copy() rebuilds the block manager, consolidating them, so
        # downstream row ops (.iloc, column assignment, ...) don't have to loop over ~thousands of
        # single-column blocks one by one
        combined = pd.concat(renamed, axis=1).copy()

        if data_fraction < 1.0:
            combined = combined.sort_index(level='time_id')
            n_t = int(len(combined) * data_fraction)
            combined = combined.iloc[:n_t]

        return combined


if __name__ == '__main__':
    loader = DiTECWDNLoader(network='CTOWN_1GB_24H')
    print(f"Found {len(loader.shard_groups)} (component_parameter, type) groups for {loader.network}:\n")
    for (component_parameter, type_), paths in sorted(loader.shard_groups.items()):
        print(f"  {component_parameter:40s} [{type_:15s}] -> {len(paths)} shard(s)")
        for p in sorted(paths):
            print(f"      {p}")

    print("\nLoading topology...")
    topology = loader.load_topology()
    print(f"  {len(topology['nodes'])} nodes, {len(topology['links'])} links")

    node_type_counts = defaultdict(int)
    for info in topology['nodes'].values():
        node_type_counts[info['type']] += 1
    print(f"  node types: {dict(node_type_counts)}")

    link_type_counts = defaultdict(int)
    for info in topology['links'].values():
        link_type_counts[info['type']] += 1
    print(f"  link types: {dict(link_type_counts)}")

    print("\nLoading attributes for scenarios [0, 1, 2]...")
    attrs_dfs = loader.load_attributes_as_df([0, 1, 2])
    for component_type, df in sorted(attrs_dfs.items()):
        print(f"  {component_type:10s} shape={df.shape} columns={list(df.columns)}")

    print("\nLoading sensor data for scenarios [0, 1, 2]...")
    sensor_dfs = loader.load_sensor_data([0, 1, 2])
    for signal, df in sorted(sensor_dfs.items()):
        print(f"  {signal:16s} shape={df.shape}  columns[:3]={list(df.columns[:3])}")

    print("\nLoading combined sensor df for scenarios [0, 1, 2]...")
    sensor_df = loader.load_sensor_df([0, 1, 2])
    print(f"  shape={sensor_df.shape}  columns[:3]={list(sensor_df.columns[:3])}  columns[-3:]={list(sensor_df.columns[-3:])}")
