import dataclasses
import json
from datetime import datetime
from pathlib import Path

import torch

from src.data.dataset import LeakDBDataset
from src.model.gae import GraphAutoEncoder
from src.training.trainer import Trainer
from src.rules.extractor import RuleExtractor
from src.rules.evaluator import RuleEvaluator
from src.utils.config import Config


def main():
    config = Config()
    if config.device is not None:
        device = torch.device(config.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = LeakDBDataset(config)
    print(f"Dataset ready — {len(dataset)} graphs | {len(dataset.train_indices)} train / {len(dataset.val_indices)} val / {len(dataset.test_indices)} test")

    print("Initialising model...")
    sample = dataset[0]
    model = GraphAutoEncoder(config, sample.metadata(), dataset.gb)
    model.to(device)
    print("Model ready")

    if config.train_model:
        print("Starting training...")
        trainer = Trainer(model, dataset, config, device)
        trainer.train()
        print("Training complete")
    else:
        print(f"Loading checkpoint from {config.checkpoint_path}...")
        checkpoint = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Checkpoint loaded — epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")

    if config.learn_rules:
        print("Extracting rules...")
        extractor = RuleExtractor(model, dataset.gb, dataset, config, device)
        rules = extractor.extract()
        print(f"Extraction complete — {len(rules)} candidate rules")
        Path(config.rules_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config.rules_path, 'w') as f:
            json.dump(rules, f)
        print(f"Rules saved to {config.rules_path}")
    else:
        rules_path = Path(config.rules_path)
        if not rules_path.exists():
            raise FileNotFoundError(
                f"learn_rules=False but no rules file found at '{config.rules_path}'. "
                "Set learn_rules=True to extract and save rules first."
            )
        print(f"Loading rules from {config.rules_path}...")
        with open(rules_path) as f:
            raw = json.load(f)
        rules = [
            {
                'antecedent': [tuple(item) for item in rule['antecedent']],
                'consequent': tuple(rule['consequent']),
            }
            for rule in raw
        ]
        print(f"Loaded {len(rules)} rules")

    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")

    print("Evaluating rules...")
    evaluator = RuleEvaluator(dataset, config, device)
    evaluated_rules, averages = evaluator.evaluate(rules)
    if config.filter_rules:
        print(f"Evaluation complete — {len(evaluated_rules)} rules pass support>={config.min_support} and confidence>={config.min_confidence}")
    else:
        print(f"Evaluation complete — {len(evaluated_rules)} rules (no filtering applied)")
    print(f"Averages over all {len(rules)} rules: " + " | ".join(f"{k}={v:.4f}" for k, v in averages.items()))
    for rule in evaluated_rules[:5]:
        print(f"  support={rule['support']:.3f} conf={rule['confidence']:.3f} lift={rule['lift']:.3f} zhang={rule['zhang']:.3f} | {rule['antecedent']} -> {rule['consequent']}")
    print(f"Data coverage: {averages.get('coverage', 'n/a')}")

    results = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': dataclasses.asdict(config),
        'n_rules_total': len(rules),
        'n_rules_after_filter': len(evaluated_rules),
        'averages': averages,
        'top_rules': evaluated_rules[:10],
    }
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"run_{results['timestamp'].replace(':', '-')}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
