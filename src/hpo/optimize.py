import copy
import dataclasses
import json
from pathlib import Path

import optuna
import torch

from src.model.gae import GraphAutoEncoder
from src.rules.evaluator import RuleEvaluator
from src.rules.extractor import RuleExtractor
from src.training.trainer import Trainer
from src.utils.config import Config


def _sample_params(trial: optuna.Trial, base_config: Config) -> Config:
    """Sample one set of hyperparameters for a trial.

    Graph-level settings (n_bins, virtual_node_mode, etc.) are inherited unchanged
    from base_config because changing them would require rebuilding the dataset.
    """
    config = copy.deepcopy(base_config)

    # -- Model architecture --
    config.hidden_channels = trial.suggest_categorical('hidden_channels', [8, 16, 32, 64, 128])
    config.latent_channels = trial.suggest_categorical('latent_channels', [4, 8, 16])
    config.num_layers = trial.suggest_int('num_layers', 5, 8)
    config.encoder_type = trial.suggest_categorical('encoder_type', ['sage', 'gat'])
    config.aggr = trial.suggest_categorical('aggr', ['sum', 'mean', 'max'])
    #config.norm_type = trial.suggest_categorical('norm_type', ['none', 'batch', 'layer'])
    config.gat_for_sensor_virtual = trial.suggest_categorical('gat_for_sensor_virtual', [True, False])

    # -- Training --
    config.lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)     #sample on a log scale because orders of magnitude matter most
    config.batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    config.epochs = trial.suggest_int('epochs', 2, 4)  #TODO: change back to larger range for actual optimization
    config.masking_strategy = trial.suggest_categorical('masking_strategy', ['remove', 'all_bins'])
    config.mask_ratio = trial.suggest_float('mask_ratio', 0.5, 0.9)

    # Override patience with the HPO-fixed value so each trial uses the same stopping criterion
    config.patience = base_config.hpo_patience

    return config


def _objective(
    trial: optuna.Trial,
    base_config: Config,
    dataset,
    device: torch.device,
    run_dir: Path,
) -> float:
    """Optuna objective: trains a model, extracts rules, returns the average of each rule's confidence × zhang.

    Val loss is stored as a user attribute for reference but is not the optimisation target.
    Returns 0.0 as a fallback when no rules are extracted.
    """
    config = _sample_params(trial, base_config)

    # Each trial gets its own subdirectory for its checkpoint and sampled config
    trial_dir = run_dir / f'trial_{trial.number:04d}'
    trial_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path = str(trial_dir / 'best_model.pt')

    with open(trial_dir / 'config.json', 'w') as f:
        json.dump(dataclasses.asdict(config), f, indent=2)

    torch.manual_seed(base_config.random_seed)
    torch.cuda.manual_seed_all(base_config.random_seed)

    sample = dataset[0]
    model = GraphAutoEncoder(config, sample.metadata(), dataset.gb)
    model.to(device)

    trainer = Trainer(model, dataset, config, device)
    history = trainer.train()

    best_val_loss = min(history['val'])
    best_epoch = history['val'].index(best_val_loss) + 1
    trial.set_user_attr('best_val_loss', best_val_loss)
    trial.set_user_attr('best_epoch', best_epoch)

    # Load the best checkpoint — training ends at the last epoch but the saved
    # checkpoint is from the epoch with the lowest val loss
    checkpoint = torch.load(config.checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Override extraction settings for HPO: evaluate all rules (no filter) and
    # skip hop distance computation to keep each trial fast
    config.filter_rules = False
    config.compute_hop_distance = False

    extractor = RuleExtractor(model, dataset.gb, dataset, config, device)
    rules = extractor.extract()

    if not rules:
        trial.set_user_attr('n_rules', 0)
        trial.set_user_attr('avg_confidence', 0.0)
        trial.set_user_attr('avg_zhang', 0.0)
        trial.set_user_attr('avg_confidence_zhang', 0.0)
        return 0.0

    evaluator = RuleEvaluator(dataset, config, device, split='train')
    _, averages = evaluator.evaluate(rules)  # compute all metrics; confidence_zhang needs support_ant/con anyway

    score = averages.get('confidence_zhang', 0.0)

    trial.set_user_attr('n_rules', len(rules))
    for metric, value in averages.items():
        trial.set_user_attr(f'avg_{metric}', value)

    return score


def run_hpo(
    base_config: Config,
    dataset,
    device: torch.device,
    run_dir: Path,
) -> 'optuna.Study':
    """Create (or resume) an Optuna study and run HPO for hpo_n_trials trials.

    Study metadata (params, objective values) is persisted in a SQLite database
    inside run_dir. Model checkpoints and configs are saved in trial subdirectories.

    Note: SQLite works correctly for sequential execution (n_jobs=1).
    For parallel workers, switch storage to:
        JournalStorage(JournalFileStorage(str(run_dir / 'journal.log')))
    """
    db_path = run_dir / f'{base_config.hpo_study_name}.db'
    storage = f'sqlite:///{db_path}'

    # Suppress Optuna's per-trial log lines; we print our own summary instead
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        if base_config.hpo_enable_pruning
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        study_name=base_config.hpo_study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,  # resumes an existing study with the same name
        pruner=pruner,
    )

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            return
        conf  = trial.user_attrs.get('avg_confidence', 0.0)
        zhang = trial.user_attrs.get('avg_zhang', 0.0)
        print(
            f"  Trial {trial.number:3d} | score={trial.value:.4f} "
            f"(avg conf={conf:.3f}, avg zhang={zhang:.3f}) "
            f"| rules={trial.user_attrs.get('n_rules', '?')} "
            f"| val_loss={trial.user_attrs.get('best_val_loss', float('nan')):.4f} "
            f"| best={study.best_value:.4f} "
            f"| {trial.params}"
        )

    study.optimize(
        lambda trial: _objective(trial, base_config, dataset, device, run_dir),
        n_trials=base_config.hpo_n_trials,
        callbacks=[_log_trial],
    )

    return study
