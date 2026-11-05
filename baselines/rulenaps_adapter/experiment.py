"""Single-call interface for the RuleNaps baseline: run → convert → evaluate.

Used by the orchestrating script to run RuleNaps with a given set of hyperparameters and get
back the evaluated rules and metric averages in the same format as the GAE and FP-growth baselines.
Also used internally by optimize.py to avoid duplicating the three-step pipeline per trial.

Usage:
    from baselines.rulenaps_adapter.experiment import run_experiment
    rules, averages, summary = run_experiment(config, params, run_dir, dataset, device)
"""
from pathlib import Path

import torch

from baselines.rulenaps_adapter.convert import convert_run
from baselines.rulenaps_adapter.evaluate import evaluate_run
from baselines.rulenaps_adapter.run import run
from src.utils.config import Config


def run_experiment(
    config: Config,
    params: dict,
    run_dir: str | Path,
    dataset,
    device: torch.device,
    compute_hop_distance: bool = False,
    threads: int = 16,
    eval_config: Config = None,
) -> tuple[list[dict], dict, dict]:
    """Run RuleNaps with the given hyperparameters and return the evaluated rules and metrics.

    Args:
        config:               project config (used for split, train_ratio, max_antecedent_size, etc.)
        params:               RuleNaps hyperparameter overrides (on top of DEFAULT_PARAMS in run.py)
        run_dir:              directory to write all RuleNaps and adapter output files to
        dataset:              pre-built dataset, shared across calls to avoid reloading
        device:               torch device for RuleEvaluator
        compute_hop_distance: whether to annotate rules with hop distances (slow; skip for HPO)
        threads:              torch CPU threads for RuleNaps training (default: 16, as the authors)
        eval_config:          config override for evaluation (e.g. filter_rules=False); defaults to config

    Returns:
        rules:    converted rules in project format [{antecedent, consequent}, ...]
        averages: metric averages from RuleEvaluator (confidence, zhang, coverage, ...)
        summary:  run_summary.json dict from train_rulenaps.py (includes runtime_training_s)
    """
    if eval_config is None:
        eval_config = config
    run_dir = Path(run_dir)
    summary = run(config, split='all', params=params, out_dir=run_dir, threads=threads)
    rules, _ = convert_run(config, str(run_dir), config.max_antecedent_size)
    averages = evaluate_run(eval_config, str(run_dir), dataset=dataset, device=device,
                            compute_hop_distance=compute_hop_distance)
    return rules, averages, summary
