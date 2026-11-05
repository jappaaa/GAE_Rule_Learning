"""Single-call interface for the Aerial baseline: train → convert → evaluate.

Used by the orchestrating script to run Aerial with a given set of hyperparameters and get
back the evaluated rules and metric averages in the same format as the GAE, FP-growth and
RuleNaps baselines. Also used internally by optimize.py to avoid duplicating the pipeline per trial.

Usage:
    from baselines.aerial.experiment import run_experiment
    rules, averages, summary = run_experiment(config, train_params, rule_params, run_dir, dataset, device)
"""
from pathlib import Path

import torch

from baselines.aerial.convert import convert_run
from baselines.aerial.evaluate import evaluate_run
from baselines.aerial.train_aerial import train
from src.utils.config import Config


def run_experiment(
    config: Config,
    train_params: dict,
    rule_params: dict,
    run_dir: str | Path,
    dataset,
    device: torch.device,
    compute_hop_distance: bool = False,
    eval_config: Config = None,
) -> tuple[list[dict], dict, dict]:
    """Run Aerial with the given hyperparameters and return the evaluated rules and metrics.

    Trains and evaluates on the train split only (see train_aerial.py and evaluate.py), so val stays available
    as a held-out set for any method that wants to track a validation-style signal, without being mixed into
    this comparison.

    Args:
        config:               project config (used for max_antecedent_size, etc.)
        train_params:         aerial.model.train kwargs overriding DEFAULT_TRAIN_PARAMS
        rule_params:          aerial.rule_extraction.generate_rules kwargs overriding DEFAULT_RULE_PARAMS
        run_dir:              directory to write all Aerial and adapter output files to
        dataset:              pre-built dataset, shared across calls to avoid reloading
        device:               torch device for RuleEvaluator
        compute_hop_distance: whether to annotate rules with hop distances (slow; skip for HPO)
        eval_config:          config override for evaluation (e.g. filter_rules=False); defaults to config

    Returns:
        rules:    converted rules in project format [{antecedent, consequent}, ...]
        averages: metric averages from RuleEvaluator (confidence, zhang, coverage, ...)
        summary:  run_summary.json dict from train_aerial.py (includes runtime_training_s)
    """
    if eval_config is None:
        eval_config = config
    run_dir = Path(run_dir)
    summary = train(config, split='train', train_params=train_params, rule_params=rule_params, out_dir=run_dir)
    rules, _ = convert_run(str(run_dir))
    averages = evaluate_run(eval_config, str(run_dir), dataset=dataset, device=device,
                            compute_hop_distance=compute_hop_distance)
    return rules, averages, summary
