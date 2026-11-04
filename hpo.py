import json
from pathlib import Path

import torch

from src.data.leakdb.dataset import LeakDBDataset
from src.hpo.optimize import run_hpo
from src.utils.config import Config


def main():
    # ------------------------------------------------------------------ config
    config = Config()
    # Set graph-level params here (these are fixed across all trials because
    # changing them would require rebuilding the dataset):
    #   config.scenarios         = list(range(1, 6))
    #   config.n_bins            = 9
    #   config.virtual_node_mode = 'type_interconnected'
    #   config.onehot_bin_features = False
    #   ...
    #
    # Set HPO params here:
    #   config.hpo_n_trials      = 50
    #   config.hpo_study_name    = 'gae_hpo'
    #   config.hpo_patience      = 10
    #   config.hpo_enable_pruning = False

    # ------------------------------------------------------------------ device
    device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Global reproducibility settings — applied once for the whole process
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # ----------------------------------------------------------------- dataset
    # Built once here and shared across all trials — no rebuild per trial needed
    # because HPO only varies model/training params, not graph construction params
    print("Building dataset...")
    dataset = LeakDBDataset(config)
    print(f"Dataset ready — {len(dataset)} graphs | {len(dataset.train_indices)} train / {len(dataset.val_indices)} val")

    # ----------------------------------------------------------------- run dir
    run_dir = Path(config.results_dir) / 'hpo' / f"study_{config.hpo_study_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Study: '{config.hpo_study_name}' | {config.hpo_n_trials} trials | patience={config.hpo_patience}")
    print(f"Note: checkpoints/best_model.pt will be overwritten by each trial during HPO\n")

    # --------------------------------------------------------------------- hpo
    study = run_hpo(config, dataset, device, run_dir)

    # ----------------------------------------------------------------- results
    best = study.best_trial
    print(f"\nHPO complete — best trial #{best.number}: score={best.value:.4f} "
          f"(conf={best.user_attrs.get('avg_confidence', 0.0):.3f} "
          f"× zhang={best.user_attrs.get('avg_zhang', 0.0):.3f})")
    print(f"  rules={best.user_attrs.get('n_rules', '?')} "
          f"| val_loss={best.user_attrs.get('best_val_loss', float('nan')):.4f} "
          f"| best_epoch={best.user_attrs.get('best_epoch', '?')}")
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    best_result = {
        'best_trial': best.number,
        'best_score': best.value,
        'avg_confidence': best.user_attrs.get('avg_confidence'),
        'avg_zhang': best.user_attrs.get('avg_zhang'),
        'n_rules': best.user_attrs.get('n_rules'),
        'best_val_loss': best.user_attrs.get('best_val_loss'),
        'best_epoch': best.user_attrs.get('best_epoch'),
        'best_params': best.params,
    }
    out_path = run_dir / 'hpo_best.json'
    with open(out_path, 'w') as f:
        json.dump(best_result, f, indent=2)
    print(f"\nBest params saved to {out_path}")
    print("Copy best_params into Config in main.py to run the full pipeline with the best configuration.")


if __name__ == '__main__':
    main()
