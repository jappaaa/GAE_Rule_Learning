from dataclasses import dataclass


@dataclass
class Config:
    # Data paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"

    # Graph — set after inspecting LeakDB
    node_feature_dim: int = None

    # Model
    hidden_channels: int = 64
    latent_channels: int = 32
    encoder_type: str = "gcn"  # "gcn" or "gat"
    noise_std: float = 0.1

    # Training
    train_ratio: float = 0.8
    lr: float = 1e-3
    epochs: int = 100
    patience: int = 10
    batch_size: int = 32

    # Rule extraction
    reconstruction_threshold: float = 0.5
    min_support: float = 0.1
    min_confidence: float = 0.5
