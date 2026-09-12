import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data paths — raw_data_dir and processed_data_dir are derived from data_dir and therefore defined in __post_init__
    data_dir: str = "data"
    raw_data_dir: str = field(init=False)
    processed_data_dir: str = field(init=False)

    def __post_init__(self):
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")

    # Device — set to "cpu" or "cuda" to override auto-detection of cuda
    device: str = None

    # Reproducibility
    random_seed: int = 42

    # Dataset
    scenarios: list = field(default_factory=lambda: list(range(1, 2))) # Although we do not change its contents, default factory is used to avoid sharing same list amongst different instances
    data_fraction: float = 1.0  # fraction of timestamps per scenario to use (most useful when only one scenario is used); delete processed dir when changing
    n_bins: int = 10
    train_ratio: float = 0.8

    # Graph
    bidirectional_has_measure: bool = True
    virtual_node_mode: str = 'none'  # 'none', 'global', 'hierarchical', 'hierarchical_direct', or 'type_interconnected' if not none, set filter_consequents_by_hops to False

    # Model
    hidden_channels: int = 32
    latent_channels: int = 4    # last dimension of node embeddings before decoding
    encoder_type: str = "sage"  # "sage" or "gat" are implemented
    num_layers: int = 7         # Important as this determines how far message travel through the graph 
    aggr: str = "sum"           # how to combine messages from different edge types in heterogeneous GNN

    # Training
    train_model: bool = True    # if set to True, a new model is trained
    checkpoint_path: str = 'checkpoints/best_model.pt'  # location of model with lowest validation loss
    lr: float = 1e-3
    epochs: int = 30
    patience: int = 15
    batch_size: int = 128
    use_masking: bool = True    
    mask_validation: bool = True      # mask the validation graphs such that val and train tasks are better aligned
    mask_ratio: float = 0.80          # ratio of sensors per graph that are masked
    masking_strategy: str = 'remove'  # 'remove', 'all_bins', or 'random_bin'

    # Rule extraction (GAE)
    learn_rules: bool = True     
    evaluate_rules: bool = True
    filter_consequents_by_hops: bool = True  # if set to true, only sensors within the receptive field of the antecedents are considered as consequents
    rules_path: str = 'rules/rules.json'
    extraction_batch_size: int = 1024
    antecedent_threshold: float = 0.8
    consequent_threshold: float = 0.8
    max_antecedent_size: int = 2
    filter_rules: bool = False               # remove rule that do not pass min_support or min_confidence
    min_support: float = 0.1
    min_confidence: float = 0.5
    compute_hop_distance: bool = True        # if the hop distance between antecedent items and consequent needs to be computed for each rule

    # Rule extraction (FP-growth)
    learn_rules_fp: bool = False             
    evaluate_rules_fp: bool = False
    rules_path_fp: str = 'rules/rules_fp.json'
    min_support_fp: float = 0.05
    min_confidence_fp: float = 0.8

    # Visualization (diagnostic only)
    visualize_graphs: bool = False
    viz_output_dir: str = 'visualizations'

    # Results
    results_dir: str = 'results'
