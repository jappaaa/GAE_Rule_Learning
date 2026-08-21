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
    scenarios: list = field(default_factory=lambda: list(range(1, 2)))
    n_bins: int = 10
    train_ratio: float = 0.8
    val_ratio: float = 0.1  # test_ratio = 1 - train_ratio - val_ratio

    # Graph
    bidirectional_has_measure: bool = False

    # Model
    hidden_channels: int = 32
    latent_channels: int = 4
    encoder_type: str = "sage"  # "sage" or "gat"
    num_layers: int = 7         # Important as this determines how far message travel through the graph
    aggr: str = "sum"           # how to combine messages from different edge types in heterogeneous GNN, passed to .to_hetero()

    # Training
    train_model: bool = True
    checkpoint_path: str = 'checkpoints/best_model.pt'
    lr: float = 1e-3
    epochs: int = 2
    patience: int = 5
    batch_size: int = 128
    use_masking: bool = True
    mask_ratio: float = 0.75
    masking_strategy: str = 'remove'  # 'remove' or 'all_bins'

    # Rule extraction
    learn_rules: bool = False
    rules_path: str = 'rules/rules.json'
    extraction_batch_size: int = 256
    antecedent_threshold: float = 0.9
    consequent_threshold: float = 0.9
    max_antecedent_size: int = 2
    filter_rules: bool = False
    min_support: float = 0.1
    min_confidence: float = 0.5

    # Visualization (diagnostic only)
    visualize_graphs: bool = False
    viz_output_dir: str = 'visualizations'

    # Results
    results_dir: str = 'results'
