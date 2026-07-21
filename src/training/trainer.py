import torch
from torch_geometric.utils import negative_sampling

from src.model.gae import GraphAutoEncoder
from src.utils.config import Config


class Trainer:
    def __init__(self, model: GraphAutoEncoder, dataset, config: Config):
        self.model = model
        self.dataset = dataset
        self.config = config

    def train(self):
        pass

    def _train_epoch(self, loader) -> float:
        pass

    def _validate(self, loader) -> float:
        pass

    def _sample_negatives(self, data) -> torch.Tensor:
        """Sample negative edges (non-existing pairs) for BCE loss."""
        pass

    def _save_checkpoint(self, epoch: int, val_loss: float):
        pass

    def load_checkpoint(self, path: str):
        pass
