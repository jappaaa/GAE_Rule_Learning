import pandas as pd

from src.model.gae import GraphAutoEncoder
from src.utils.config import Config


class RuleExtractor:
    def __init__(self, model: GraphAutoEncoder, config: Config):
        self.model = model
        self.config = config

    def reconstruct_has_measure(self, dataset) -> pd.DataFrame:
        """Run test graphs through the model and return reconstruction scores
        for has_measure edges only (sensor → value_node pairs)."""
        pass

    def binarize(self, scores: pd.DataFrame) -> pd.DataFrame:
        """Threshold reconstruction scores → binary transaction matrix.
        Rows = timestamps, columns = (sensor, bin) items."""
        pass

    def extract(self, dataset) -> list:
        """Full pipeline: reconstruct → binarize → FP-Growth → association rules."""
        pass
