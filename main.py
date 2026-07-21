from src.data.dataset import LeakDBDataset
from src.model.gae import GraphAutoEncoder
from src.training.trainer import Trainer
from src.rules.extractor import RuleExtractor
from src.rules.evaluator import RuleEvaluator
from src.utils.config import Config


def main():
    config = Config()

    dataset = LeakDBDataset(config)
    model = GraphAutoEncoder(config)

    trainer = Trainer(model, dataset, config)
    trainer.train()

    extractor = RuleExtractor(model, config)
    rules = extractor.extract(dataset)

    evaluator = RuleEvaluator()
    metrics = evaluator.evaluate(rules)
    evaluator.print_summary(metrics)


if __name__ == "__main__":
    main()
