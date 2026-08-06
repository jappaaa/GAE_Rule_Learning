import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data paths — raw_data_dir and processed_data_dir are derived from data_dir
    data_dir: str = "data"
    raw_data_dir: str = field(init=False)
    processed_data_dir: str = field(init=False)

    def __post_init__(self):
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")

    # Reproducibility
    random_seed: int = 42

    # Dataset
    scenarios: list = field(default_factory=lambda: list(range(1, 11)))
    n_bins: int = 3
    train_ratio: float = 0.7
    val_ratio: float = 0.15  # test_ratio = 1 - train_ratio - val_ratio

    # Graph
    bidirectional_has_measure: bool = False

    # Model
    hidden_channels: int = 64
    latent_channels: int = 32
    encoder_type: str = "sage"  # "sage" or "gat"
    num_layers: int = 2         # use 4 for bidirectional experiments
    aggr: str = "sum"           # how to combine messages from different edge types

    # Training
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    batch_size: int = 32

    # Rule extraction
    reconstruction_threshold: float = 0.5
    min_support: float = 0.1
    min_confidence: float = 0.5
