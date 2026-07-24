import os
from pathlib import Path

import pandas as pd
import wntr


class LeakDBLoader:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir
        base_inp_path = os.path.join(raw_data_dir, "LeakDB", "Hanoi_CMH", "Hanoi_CMH", "Hanoi_CMH.inp")
        self.base_wn = wntr.network.WaterNetworkModel(base_inp_path)


    def load_topology(self) -> dict:
        """Parse node list and pipe-to-junction connections from the base .inp file.
        Called once — topology is identical across all scenarios."""
        nodes = (
            {n: {"type": "junction"} for n in self.base_wn.junction_name_list} |
            {n: {"type": "reservoir"} for n in self.base_wn.reservoir_name_list}
        )

        pipes = {
            name: {
                "start_node": self.base_wn.get_link(name).start_node_name,
                "end_node": self.base_wn.get_link(name).end_node_name,
            }
            for name in self.base_wn.pipe_name_list
        }

        return {"nodes": nodes, "pipes": pipes}
    

    def load_attributes(self, scenario: int) -> dict:
        """Parse static node and pipe attributes from the scenario-specific .inp file.
        Returns junction attributes (elevation, base demand) and pipe attributes
        (length, diameter, roughness)."""
        scenario_inp_path = os.path.join(self.raw_data_dir, "LeakDB", "Hanoi_CMH", "Hanoi_CMH", f"Scenario-{scenario}", f"Hanoi_CMH_Scenario-{scenario}.inp")
        scenario_wn = wntr.network.WaterNetworkModel(scenario_inp_path)

        junctions = {
            n: {
                "elevation": scenario_wn.get_node(n).elevation,
                "base_demand": scenario_wn.get_node(n).base_demand,
            }
            for n in scenario_wn.junction_name_list
        }

        pipes = {
            name: {
                "length": scenario_wn.get_link(name).length,
                "diameter": scenario_wn.get_link(name).diameter,
                "roughness": scenario_wn.get_link(name).roughness,
            }
            for name in scenario_wn.pipe_name_list
        }

        return {"junctions": junctions, "pipes": pipes}


    def load_attributes_as_df(self, scenarios: list[int]) -> dict[str, pd.DataFrame]:
        """Load static attributes for multiple scenarios in long format for discretization.
        Returns two DataFrames (junctions and pipes), each with one column per attribute type
        and one row per (node/pipe, scenario) combination."""
        junction_records = []
        pipe_records = []

        # iterate over the scenarios
        for scenario in scenarios:

            # load the attributes from the EPANET file
            attrs = self.load_attributes(scenario)

            # for each junction node extract the attributes and combine into dict and add to the junction list
            for node_id, node_attrs in attrs["junctions"].items():
                junction_records.append({
                    "node_id": node_id,
                    "scenario": scenario,
                    "elevation": node_attrs["elevation"],
                    "base_demand": node_attrs["base_demand"],
                })

            # for each pipe extract the attributes and combine into dict and add to the pipe list
            for pipe_id, pipe_attrs in attrs["pipes"].items():
                pipe_records.append({
                    "pipe_id": pipe_id,
                    "scenario": scenario,
                    "length": pipe_attrs["length"],
                    "diameter": pipe_attrs["diameter"],
                    "roughness": pipe_attrs["roughness"],
                })

        # create the pipe an junction dataframe from list of dicts, where the columns are the attributes, indexed by ID and scenario
        junctions_df = pd.DataFrame(junction_records).set_index(["node_id", "scenario"])
        pipes_df = pd.DataFrame(pipe_records).set_index(["pipe_id", "scenario"])

        return {"junctions": junctions_df, "pipes": pipes_df}   
    

    def load_sensor_data(self, scenario: int) -> dict[str, pd.DataFrame]:
        """Read pressure, flow, and demand CSVs for a given scenario.
        Returns a dict with keys 'pressures', 'flows', 'demands', each a DataFrame
        with timestamps as index and node/pipe IDs as columns."""

        sensor_types = ["Pressures", "Flows", "Demands"]
        scenario_dir = Path(self.raw_data_dir, "LeakDB", "Hanoi_CMH", "Hanoi_CMH", f"Scenario-{scenario}")

        sensor_dfs_dict = {}

        for sensor_type in sensor_types:
            sensor_type_dir = scenario_dir / sensor_type
            dfs = []

            for path in sorted(sensor_type_dir.iterdir()):
                df = pd.read_csv(path)
                sensor_place = path.stem.lower()
                df.rename(columns={"Value": sensor_place}, inplace=True)
                df.set_index("Timestamp", inplace=True)
                dfs.append(df)

            sensor_dfs_dict[sensor_type.lower()] = pd.concat(dfs, axis=1)

        labels = pd.read_csv(scenario_dir / "Labels.csv", index_col="Timestamp")
        sensor_dfs_dict["labels"] = labels

        return sensor_dfs_dict