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

    # Device — set to "cpu" or "cuda" to override auto-detection
    device: str = None

    # Reproducibility
    random_seed: int = 42

    # Dataset
    scenarios: list = field(default_factory=lambda: list(range(1, 3)))
    n_bins: int = 4
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
    train_model: bool = False
    checkpoint_path: str = 'checkpoints/best_model.pt'
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    batch_size: int = 32
    use_masking: bool = False
    mask_ratio: float = 0.3
    masking_strategy: str = 'remove'  # 'remove' or 'all_bins'

    # Rule extraction
    antecedent_threshold: float = 0.9
    consequent_threshold: float = 0.9
    max_antecedent_size: int = 2
    min_support: float = 0.1
    min_confidence: float = 0.5
