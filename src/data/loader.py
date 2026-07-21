import pandas as pd


class LeakDBLoader:
    def __init__(self, raw_data_dir: str):
        self.raw_data_dir = raw_data_dir

    def load_sensor_data(self) -> pd.DataFrame:
        pass

    def load_knowledge_graph(self):
        pass

    def get_graph_structure(self) -> tuple:
        """Return (node_list, edge_index) derived from the KG topology."""
        pass
