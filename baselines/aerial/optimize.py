"""Step 5 of the Aerial baseline: tune Aerial's hyperparameters with Optuna.

Runs in the project environment. Usage, from the project root (after export.py):
    python -m baselines.aerial.optimize [--study-name aerial_confidence_zhang] [--n-trials 25]

Builds the processed dataset if it is missing and re-exports the train-split transactions itself, so neither
main.py nor export.py has to be run first.

Single objective, confidence_zhang: the average, over the converted rules, of each rule's confidence x zhang,
evaluated with RuleEvaluator (no filtering, no hop distances) -- the same objective as the GAE HPO and
RuleNaps' rule_metrics HPO. Unlike RuleNaps, Aerial's own training reports no held-out validation loss
(aerial.model.train trains and reports loss on the train split only), so there is no second objective to
choose from here.

min_rule_frequency/min_rule_strength/max_antecedents are NOT part of the search space -- they stay fixed to
the GAE pipeline's antecedent_threshold/consequent_threshold/max_antecedent_size, same as every other
baseline (see docs/experiment_plan.md).

The search space below is calibrated to this dataset's scale (15,768 train transactions, 98 categorical
features, ~625 columns after Aerial's internal one-hot encoding) using the qualitative guidance in PyAerial's
own Parameter Tuning Guide (https://pyaerial.readthedocs.io/en/latest/parameter_guide.html) -- see the
comment above sample_params() and docs/experiment_plan.md for the reasoning behind each range.
"""
import argparse
import copy
import json
from pathlib import Path

import optuna
import torch

from baselines.aerial.experiment import run_experiment
from baselines.aerial.export import export
from src.data.leakdb.dataset import LeakDBDataset
from src.utils.config import Config

# Calibrated to this dataset's scale per PyAerial's Parameter Tuning Guide:
#   - epochs: the guide recommends short training (1-5) for fewer/higher-quality rules, 10+ for more/
#     lower-quality rules, and to increase only on underfitting (no rules). Our own trial runs already
#     produce 900k-1.8M rules at epochs ~12-17, nowhere near underfitting, so the range is biased short
#     rather than symmetric.
#   - batch_size: the guide's own auto-selection picks 64 for datasets >=5000 rows (ours has 15,768 train
#     transactions), so the choices are centered there instead of around a small-dataset default.
#   - layer_dims: the guide's examples ([4, 2] strong / [50, 25] weak compression) assume a far smaller
#     one-hot width than our ~625 columns; the four tiers below scale that same idea to our size, and
#     'large' is capped at 2 hidden layers per "1-2 hidden layers sufficient for most tabular datasets."
#   - min/max_unmasked_features: the guide gives no feature-count-relative formula, so these stay close to
#     Aerial's own defaults (1, 10).
#   - lr: the guide gives only the default (5e-3), no tuning range, so the placeholder log-range already
#     bracketing it is kept as is.
_LAYER_DIMS_CHOICES = {
    'tiny':   [16, 8],      # strongest compression -> fewest, highest-quality rules
    'small':  [32, 16],
    'medium': [64, 32],
    'large':  [128, 64],    # weakest compression -> most, lowest-quality rules
}


def sample_params(trial: optuna.Trial) -> dict:
    """Sample one aerial.model.train parameter set (passed as train_params to run_experiment)."""
    layer_dims_key = trial.suggest_categorical('layer_dims', list(_LAYER_DIMS_CHOICES))
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 1, 15)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    min_unmasked = trial.suggest_int('min_unmasked_features', 1, 3)
    max_unmasked = trial.suggest_int('max_unmasked_features', min_unmasked + 1, 10)
    return {
        'lr': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'layer_dims': _LAYER_DIMS_CHOICES[layer_dims_key],
        'min_unmasked_features': min_unmasked,
        'max_unmasked_features': max_unmasked,
    }


def _objective(trial: optuna.Trial, config: Config, eval_config: Config, dataset,
               device: torch.device, run_dir: Path) -> float:
    """Train Aerial with the sampled parameters, evaluate its rules, and return confidence_zhang (0.0 if no rules)."""
    train_params = sample_params(trial)
    trial_dir = run_dir / f'trial_{trial.number:04d}'

    rules, averages, summary = run_experiment(
        config, train_params, rule_params=None, run_dir=trial_dir, dataset=dataset, device=device,
        eval_config=eval_config,
    )

    trial.set_user_attr('n_rules', len(rules))
    trial.set_user_attr('runtime_training_s', summary['runtime_training_s'])
    for metric, value in averages.items():
        trial.set_user_attr(f'avg_{metric}', value)

    return averages.get('confidence_zhang', 0.0)


def best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Best completed trial, or None if there is none."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        return None
    return max(completed, key=lambda t: t.value)


def run_hpo(config: Config, study_name: str, n_trials: int) -> tuple[optuna.Study, optuna.trial.FrozenTrial | None]:
    """Create (or resume) an Optuna study and run HPO until it holds n_trials completed trials.

    Builds the dataset and re-exports the train-split transactions itself, so neither main.py nor export.py
    has to be run first (same fix as RuleNaps' optimize.py: re-exporting keeps the .parquet Aerial trains on
    in sync with the current dataset.parquet, instead of silently reusing a stale export).

    Writes to <results_dir>/aerial/hpo/study_<study_name>/:
        <study_name>.db   Optuna study (SQLite); rerunning with the same name resumes it
        trial_XXXX/       Aerial run directory of each trial (rules.json, converted_rules.json, run_aerial.json, ...)
        trials.csv        one row per trial: sampled parameters, objective value and all logged metrics
        hpo_best.json     best trial and its parameters
    """
    device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = LeakDBDataset(config)  # (re)builds the processed dir if missing; shared by all trials for rule evaluation
    export(config, split='train')  # always re-export, so the transactions Aerial trains on match the current dataset.parquet

    # same evaluation settings as the GAE/RuleNaps HPO: all rules, no filtering
    eval_config = copy.deepcopy(config)
    eval_config.filter_rules = False

    run_dir = Path(config.results_dir) / 'aerial' / 'hpo' / f'study_{study_name}'
    run_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{run_dir / f"{study_name}.db"}',
        direction='maximize',
        load_if_exists=True,  # resumes an existing study with the same name
    )

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            print(f"  Trial {trial.number:3d} | {trial.state.name}")
            return
        a = trial.user_attrs
        print(
            f"  Trial {trial.number:3d} | confidence_zhang={trial.value:.4f} "
            f"| rules={a.get('n_rules', '?')} "
            f"| conf={a.get('avg_confidence', 0.0):.3f} zhang={a.get('avg_zhang', 0.0):.3f} "
            f"| cov={a.get('avg_coverage', 0.0):.3f} "
            f"| best={study.best_value:.4f} "
            f"| {trial.params}"
        )

    # only completed trials count, so a resumed study still reaches n_trials after failed trials
    n_remaining = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_remaining > 0:
        study.optimize(
            lambda trial: _objective(trial, config, eval_config, dataset, device, run_dir),
            n_trials=n_remaining, callbacks=[_log_trial],
        )

    study.trials_dataframe().to_csv(run_dir / 'trials.csv', index=False)
    best = best_trial(study)
    with open(run_dir / 'hpo_best.json', 'w') as f:
        json.dump({
            'best_trial': best.number if best else None,
            'best_value': best.value if best else None,
            'best_params': best.params if best else None,
            'best_user_attrs': best.user_attrs if best else None,
        }, f, indent=2)

    return study, best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tune Aerial hyperparameters with Optuna')
    parser.add_argument('--study-name', default='aerial_confidence_zhang',
                        help='default: aerial_confidence_zhang; reuse a name to resume')
    parser.add_argument('--n-trials', type=int, default=25, help='total number of trials (default: 25)')
    args = parser.parse_args()

    study, best = run_hpo(Config(), args.study_name, args.n_trials)
    if best is None:
        print("\nNo completed trial found")
    else:
        print(f"\nBest trial #{best.number}: confidence_zhang={best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
