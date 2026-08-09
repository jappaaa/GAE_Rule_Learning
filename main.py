import torch

from src.data.dataset import LeakDBDataset
from src.data.graph_builder import GraphBuilder
from src.model.gae import GraphAutoEncoder
from src.training.trainer import Trainer
from src.rules.extractor import RuleExtractor
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

    print("Building graph builder...")
    gb = GraphBuilder(
        dataset.topology,
        dataset.n_bins_per_type,
        bidirectional=config.bidirectional_has_measure,
    )

    print("Initialising model...")
    sample = dataset[0]
    model = GraphAutoEncoder(config, sample.metadata(), gb)
    model.to(device)
    print(f"Model ready — {sum(p.numel() for p in model.parameters())} parameters")

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

    print("Extracting rules...")
    extractor = RuleExtractor(model, gb, dataset, config, device)
    rules = extractor.extract()
    print(f"Extraction complete — {len(rules)} candidate rules")
    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
