"""Step 5 of the RuleNaps baseline: tune RuleNaps' hyperparameters with Optuna.

Runs in the project environment. Usage, from the project root:
    python -m baselines.rulenaps_adapter.optimize --objective val_loss|rule_metrics [--min-coverage 1.0]

Builds the processed dataset if it is missing and re-exports the transactions itself, so neither main.py nor
export.py has to be run first.

Both objectives use the authors' tuning setup (RuleNaps/src/utils/experiments/tuning.py, tune_predefined_exp): the
same search space, 5 Latin hypercube trials followed by TPE up to 25 trials, and the authors' sampler settings.
    val_loss       the authors' objective: minimize the unregularized reconstruction loss of the final model on the
                   held-out rows (final_validation_loss in run_summary.json). The authors take the median of this loss
                   over several datasets; with a single dataset it is used directly.
    rule_metrics   the GAE HPO objective (src/hpo/optimize.py): maximize the average, over the converted rules,
                   of each rule's confidence × zhang, evaluated with RuleEvaluator. A trial without rules scores 0.

Every trial converts and evaluates its rules with RuleEvaluator (no filtering, no hop distances) and stores the
validation loss and all metric averages as user attributes, whichever objective is optimized, so that the two can be
compared afterwards.

Optional coverage constraint (config.hpo_min_coverage, or --min-coverage): trials whose rules cover less than this
fraction of the transactions are infeasible. It is passed to the TPE sampler as the constraint
min_coverage - coverage <= 0, so infeasible trials still guide the search, and the best trial is chosen among the
feasible ones only.

Writes to <results_dir>/rulenaps/hpo/study_<study-name>/:
    <study-name>.db    Optuna study (SQLite); rerunning with the same name resumes it
    trial_XXXX/        RuleNaps run directory of each trial (see run.py, convert.py, evaluate.py)
    trials.csv         one row per trial: sampled parameters, objective value and all logged metrics
    hpo_best.json      best (feasible) trial and its parameters
"""
import argparse
import copy
import json
import math
import subprocess
from pathlib import Path

import optuna
import torch
from scipy.stats import qmc

from baselines.rulenaps_adapter.experiment import run_experiment
from baselines.rulenaps_adapter.export import export
from src.data.leakdb.dataset import LeakDBDataset
from src.utils.config import Config

# The authors' search space (tune_predefined_exp): (type, low, high, scale) or ('cat', choices)
HP_RANGES = {
    'vertical_weight_decay_head': ('float', 0.3, 20.0, 'linear'),
    'horizontal_weight_decay_head': ('float', 0.5, 20.0, 'linear'),
    'learning_rate': ('float', 0.0001, 0.005, 'log'),
    'train_batch_size': ('cat', [16, 32, 64]),
    'epochs': ('int', 30, 150, 'linear'),
}

OBJECTIVE_DIRECTIONS = {'val_loss': 'minimize', 'rule_metrics': 'maximize'}


def _map_unit_to_param(u: float, spec: tuple):
    """Map a unit-interval sample to a parameter value, as the authors' map_unit_to_param (incl. rounding)."""
    if spec[0] == 'float':
        _, lo, hi, scale = spec
        value = lo * (hi / lo) ** u if scale == 'log' else lo + (hi - lo) * u
        return float(round(value, 4))  # plain float instead of np.float64, so study and json files stay clean
    if spec[0] == 'int':
        _, lo, hi, scale = spec
        value = lo * (hi / lo) ** u if scale == 'log' else lo + (hi - lo) * u
        return int(math.floor(value))
    choices = spec[1]
    return choices[min(int(math.floor(u * len(choices))), len(choices) - 1)]


def lhs_samples(n_samples: int, seed: int) -> list[dict]:
    """Latin hypercube start points over HP_RANGES, as the authors' lhs_samples_from_specs."""
    unit = qmc.LatinHypercube(d=len(HP_RANGES), seed=seed).random(n=n_samples)
    return [{name: _map_unit_to_param(unit[i, j], spec) for j, (name, spec) in enumerate(HP_RANGES.items())}
            for i in range(n_samples)]


def sample_params(trial: optuna.Trial) -> dict:
    params = {}
    for name, spec in HP_RANGES.items():
        if spec[0] == 'float':
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=spec[3] == 'log')
        elif spec[0] == 'int':
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        else:
            params[name] = trial.suggest_categorical(name, spec[1])
    return params


def _coverage_constraint(trial: optuna.trial.FrozenTrial) -> list[float]:
    """Optuna constraint: feasible when <= 0. Trials that did not get this far count as infeasible."""
    return [trial.user_attrs.get('coverage_constraint', 1.0)]


def _objective(trial: optuna.Trial, objective: str, min_coverage: float | None, config: Config, eval_config: Config,
               dataset: LeakDBDataset, device: torch.device, run_dir: Path, threads: int) -> float:
    """Train RuleNaps with the sampled parameters, evaluate its rules, and return the objective value."""
    params = sample_params(trial)
    trial_dir = run_dir / f'trial_{trial.number:04d}'

    rules, averages, summary = run_experiment(config, params, trial_dir, dataset, device,
                                               threads=threads, eval_config=eval_config)

    trial.set_user_attr('final_validation_loss', summary['final_validation_loss'])
    trial.set_user_attr('n_rules', len(rules))
    trial.set_user_attr('n_rule_lines_raw', summary['n_rule_lines'])
    trial.set_user_attr('runtime_training_s', summary['runtime_training_s'])
    for metric, value in averages.items():
        trial.set_user_attr(f'avg_{metric}', value)
    # without rules the evaluator returns no averages; such a trial covers nothing and scores 0
    confidence_zhang, coverage = (averages.get(k, 0.0) for k in ('confidence_zhang', 'coverage'))
    if min_coverage is not None:
        trial.set_user_attr('coverage_constraint', min_coverage - coverage)
        trial.set_user_attr('feasible', coverage >= min_coverage)

    if objective == 'val_loss':
        if summary['final_validation_loss'] is None:
            raise RuntimeError("no final_validation_loss in run_summary.json; perform_validation_runs must be enabled")
        return summary['final_validation_loss']
    return confidence_zhang


def best_trial(study: optuna.Study, min_coverage: float | None) -> optuna.trial.FrozenTrial | None:
    """Best completed trial, restricted to feasible trials when a coverage constraint is used; None if there is none."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if min_coverage is not None:
        completed = [t for t in completed if t.user_attrs.get('feasible', False)]
    if not completed:
        return None
    pick = min if study.direction == optuna.study.StudyDirection.MINIMIZE else max
    return pick(completed, key=lambda t: t.value)


def run_hpo(config: Config, objective: str, study_name: str, n_trials: int, n_lhs: int, threads: int,
            min_coverage: float | None = None) -> tuple[optuna.Study, optuna.trial.FrozenTrial | None]:
    """Create (or resume) the study, run trials until it holds n_trials finished trials, and return the best trial."""
    device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = LeakDBDataset(config)  # (re)builds the processed dir if missing; shared by all trials for rule evaluation
    export(config, split='all')  # always re-export, so the .dat used by RuleNaps matches the current dataset.parquet

    # same evaluation settings as the GAE HPO (src/hpo/optimize.py): all rules, no hop distances
    eval_config = copy.deepcopy(config)
    eval_config.filter_rules = False

    run_dir = Path(config.results_dir) / 'rulenaps' / 'hpo' / f'study_{study_name}'
    run_dir.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=f'sqlite:///{run_dir / f"{study_name}.db"}',
        direction=OBJECTIVE_DIRECTIONS[objective],
        load_if_exists=True,
        # the authors' sampler settings; n_startup_trials=0 because the LHS trials below are the start phase
        sampler=optuna.samplers.TPESampler(
            multivariate=True, group=True, n_startup_trials=0, n_ei_candidates=24, seed=config.random_seed,
            constraints_func=_coverage_constraint if min_coverage is not None else None,
        ),
    )
    # a resumed study must keep its settings, otherwise its trials are not comparable
    settings = {'objective': objective, 'min_coverage': min_coverage}
    if len(study.trials) == 0:
        for key, value in settings.items():
            study.set_user_attr(key, value)
        # This queues the trials using the initial latin hypercube sampled parameters
        for params in lhs_samples(n_lhs, seed=config.random_seed): 
            study.enqueue_trial(params)                                 
    elif any(study.user_attrs.get(k) != v for k, v in settings.items()):
        raise ValueError(f"study '{study_name}' was created with {study.user_attrs}, not {settings}; use another --study-name")

    def _log_trial(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is None:
            print(f"  Trial {trial.number:3d} | {trial.state.name}")
            return
        a = trial.user_attrs
        feasible = '' if min_coverage is None else f" | {'feasible' if a.get('feasible') else 'INFEASIBLE'}"
        print(f"  Trial {trial.number:3d} | {objective}={trial.value:.4f}{feasible} "
              f"| val_loss={a.get('final_validation_loss', float('nan')):.4f} rules={a.get('n_rules')} "
              f"conf={a.get('avg_confidence', 0.0):.3f} zhang={a.get('avg_zhang', 0.0):.3f} "
              f"conf_zhang={a.get('avg_confidence_zhang', 0.0):.3f} "
              f"cov={a.get('avg_coverage', 0.0):.3f} | {trial.params}")

    # only completed trials count, so a resumed study still reaches n_trials after failed trials
    n_remaining = n_trials - len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_remaining > 0:
        # a crashing RuleNaps subprocess marks that trial as failed instead of stopping the study
        study.optimize(
            lambda trial: _objective(trial, objective, min_coverage, config, eval_config, dataset, device, run_dir,
                                     threads),
            n_trials=n_remaining, callbacks=[_log_trial], catch=(subprocess.CalledProcessError,),
        )

    study.trials_dataframe().to_csv(run_dir / 'trials.csv', index=False)
    best = best_trial(study, min_coverage)
    with open(run_dir / 'hpo_best.json', 'w') as f:
        json.dump({**settings,
                   'best_trial': best.number if best else None,
                   'best_value': best.value if best else None,
                   'best_params': best.params if best else None,
                   'best_user_attrs': best.user_attrs if best else None}, f, indent=2)
    return study, best


if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='Tune RuleNaps hyperparameters with Optuna')
    parser.add_argument('--objective', default='val_loss', choices=list(OBJECTIVE_DIRECTIONS),
                        help="val_loss: the authors' objective (minimized); rule_metrics: average of each rule's confidence × zhang (maximized)")
    parser.add_argument('--study-name', default=None, help='default: rulenaps_<objective>; reuse a name to resume')
    parser.add_argument('--n-trials', type=int, default=25, help='total number of trials (default: 25, as the authors)')
    parser.add_argument('--n-lhs', type=int, default=5, help='Latin hypercube start trials (default: 5, as the authors)')
    parser.add_argument('--threads', type=int, default=16, help='torch CPU threads per RuleNaps run (default: 16)')
    parser.add_argument('--min-coverage', type=float, default=config.hpo_min_coverage,
                        help=f'coverage constraint (default: config.hpo_min_coverage = {config.hpo_min_coverage}); -1 disables it')
    args = parser.parse_args()

    min_coverage = None if args.min_coverage is None or args.min_coverage < 0 else args.min_coverage
    study, best = run_hpo(config, args.objective, args.study_name or f'rulenaps_{args.objective}',
                          args.n_trials, args.n_lhs, args.threads, min_coverage)
    if best is None:
        print(f"\nNo feasible trial found (min_coverage={min_coverage})")
    else:
        print(f"\nBest trial #{best.number}: {args.objective}={best.value:.4f}")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
