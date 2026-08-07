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

    dataset = LeakDBDataset(config)

    gb = GraphBuilder(
        dataset.topology,
        dataset.n_bins_per_type,
        bidirectional=config.bidirectional_has_measure,
    )

    # metadata (node types + edge types) required by to_hetero(); derived from any graph
    sample = dataset[0]
    model = GraphAutoEncoder(config, sample.metadata(), gb)
    model.to(device)

    if config.train_model:
        trainer = Trainer(model, dataset, config, device)
        trainer.train()
    else:
        checkpoint = torch.load(config.checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

    extractor = RuleExtractor(model, gb, dataset, config, device)
    rules = extractor.extract()
    print(f"Extracted {len(rules)} candidate rules")
    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")


if __name__ == "__main__":
    main()
