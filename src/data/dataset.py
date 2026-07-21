from torch_geometric.data import InMemoryDataset


class LeakDBDataset(InMemoryDataset):
    def __init__(self, config, transform=None):
        self.config = config
        super().__init__(config.processed_data_dir, transform)

    @property
    def raw_file_names(self) -> list[str]:
        pass

    @property
    def processed_file_names(self) -> list[str]:
        pass

    def download(self):
        pass

    def process(self):
        pass
