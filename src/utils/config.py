import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # Data paths — data_dir defaults to a dataset-specific subfolder (data/<dataset_name>) so each dataset's
    # raw/processed files never collide; override data_dir directly if a dataset needs a different layout.
    # raw_data_dir and processed_data_dir are derived from data_dir and therefore defined in __post_init__
    dataset_name: str = "leakdb"
    data_dir: str = None
    raw_data_dir: str = field(init=False)
    processed_data_dir: str = field(init=False)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = os.path.join("data", self.dataset_name)
        self.raw_data_dir = os.path.join(self.data_dir, "raw")
        self.processed_data_dir = os.path.join(self.data_dir, "processed")
        # min_support_fp/min_confidence_fp default to the GAE pipeline's analogous thresholds for comparability
        # (antecedent_threshold/consequent_threshold), unless explicitly overridden
        if self.min_support_fp is None:
            self.min_support_fp = self.antecedent_threshold
        if self.min_confidence_fp is None:
            self.min_confidence_fp = self.consequent_threshold
        # data_fraction trims one scenario's own timeline short; with multiple scenarios, reducing
        # data volume should be done via the scenarios list itself, not by independently truncating
        # each one's timesteps -- a cross-dataset contract, not specific to any one Dataset class
        if self.data_fraction < 1.0 and len(self.scenarios) > 1:
            raise ValueError("data_fraction < 1.0 only makes sense with a single scenario; use fewer scenarios instead to reduce data volume")

    # Device — set to "cpu" or "cuda" to override auto-detection of cuda
    device: str = None

    # Reproducibility — random_seed covers model/extraction stochasticity (torch, extraction-time sampling,
    # RuleNaps' own seed) and is what the seeded main experiment varies; data_seed is kept separately fixed
    # so the train/val split (src/data/leakdb/dataset.py) never resamples across those seeded runs.
    random_seed: int = 42
    data_seed: int = 42

    # Dataset
    ditec_network: str = "CTOWN_1GB_24H"  # which of the 36 DiTEC-WDN networks to use (only read when dataset_name == "ditec_wdn")
    scenarios: list = field(default_factory=lambda: list(range(1, 2))) # Although we do not change its contents, default factory is used to avoid sharing same list amongst different instances
    data_fraction: float = 1.0  # fraction of timestamps per scenario to use (most useful when only one scenario is used); delete processed dir when changing
    n_bins: int = 10
    train_ratio: float = 0.9    # changed to 0.9 to match validation set used by RuleNaps

    # Graph
    normalize_attributes: bool = False   # apply RobustScaler to input junction/pipe attributes
    bidirectional_has_measure: bool = False
    virtual_node_mode: str = 'type_interconnected'  # 'none', 'global', 'hierarchical', 'hierarchical_direct', or 'type_interconnected' if not none, set filter_consequents_by_hops to False
    onehot_bin_features: bool = False     # if True, value_node bin feature is one-hot (size max_bins); if False, normalized scalar (size 1); delete processed dir when changing
    mask_sensor_to_virtual: bool = False  # if True, only unmasked sensors contribute to virtual nodes (sensor→virtual edges filtered); virtual→sensor kept for all sensors

    # Model
    hidden_channels: int = 32
    latent_channels: int = 8    # last dimension of node embeddings before decoding
    encoder_type: str = "sage"  # "sage" or "gat" are implemented
    num_layers: int = 5         # Important as this determines how far message travel through the graph
    aggr: str = "sum"           # how to combine messages from different edge types in heterogeneous GNN
    norm_type: str = 'none'     # normalization between GNN layers: 'none', 'batch', or 'layer'
    gat_for_sensor_virtual: bool = False  # if True, use GAT (instead of encoder_type) for sensor→virtual edges to learn sensor importance weights
    use_input_projection: bool = True         # if True, project all node features to hidden_channels before GNN layers; if False, conv layers handle raw features via lazy init
    value_node_projection_only: bool = False   # if True (and use_input_projection=True), only project value_node; other node types keep raw features for conv to handle via lazy init

    # Training
    train_model: bool = False    # if set to True, a new model is trained
    checkpoint_path: str = 'checkpoints/best_model.pt'  # location of model with lowest validation loss
    lr: float = 1e-2
    epochs: int = 30
    patience: int = 15
    batch_size: int = 256
    use_masking: bool = True    
    mask_validation: bool = True      # mask the validation graphs such that val and train tasks are better aligned
    mask_ratio: float = 0.8          # ratio of sensors per graph that are masked
    masking_strategy: str = 'all_bins'  # 'remove', 'all_bins', or 'random_bin'

    # Rule extraction (GAE)
    learn_rules: bool = False     
    evaluate_rules: bool = False
    filter_consequents_by_hops: bool = False  # if set to true, only sensors within the receptive field of the antecedents are considered as consequents
    rules_path: str = 'rules/rules.json'
    extraction_batch_size: int = 1024
    antecedent_threshold: float = 0.2
    consequent_threshold: float = 0.8
    max_antecedent_size: int = 2
    filter_rules: bool = False               # remove rule that do not pass min_support or min_confidence
    min_support: float = 0.1
    min_confidence: float = 0.5
    compute_hop_distance: bool = True        # if the hop distance between antecedent items and consequent needs to be computed for each rule
    extraction_strategy: str = 'apriori'  # 'brute_force' or 'apriori'; apriori prunes antecedents using pairwise frequency estimates before querying the model
    match_masking_in_extraction: bool = True  # if True, non-antecedent sensors in query graphs use the training masking_strategy; if False, they have no edges (remove behaviour)

    # Rule extraction (FP-growth)
    learn_rules_fp: bool = False             
    evaluate_rules_fp: bool = False
    rules_path_fp: str = 'rules/rules_fp.json'
    min_support_fp: float = None     # defaults to antecedent_threshold if not set; see __post_init__
    min_confidence_fp: float = None  # defaults to consequent_threshold if not set; see __post_init__

    # Visualization 
    visualize_graphs: bool = False
    viz_output_dir: str = 'visualizations'

    # HPO
    hpo_n_trials: int = 2           #TODO: Set back to larger number for actual optimization
    hpo_study_name: str = 'test' # Note: to create new study, always change the name    TODO: change test back to actual name
    hpo_patience: int = 10          # fixed patience for every HPO trial
    hpo_n_seeds: int = 1            # seeds per trial; > 1 adds std penalty (multiplies compute)
    hpo_seed_lambda: float = 1.0    # weight on std(val_loss) penalty when hpo_n_seeds > 1
    hpo_enable_pruning: bool = False # prune bad trials early using intermediate val loss
    hpo_min_coverage: float = None   # if set, trials whose rules cover less than this fraction of transactions are infeasible (Optuna constraint); currently only used by the RuleNaps HPO

    # Results
    results_dir: str = 'results'
