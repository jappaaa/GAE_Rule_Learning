import pandas as pd


class RuleEvaluator:
    def evaluate(self, rules: list) -> pd.DataFrame:
        """Compute support, confidence, lift, and coverage for each rule."""
        pass

    def compare(self, rules_gae: pd.DataFrame, rules_baseline: pd.DataFrame) -> pd.DataFrame:
        """Side-by-side comparison of GAE rules vs Aerial baseline."""
        pass

    def print_summary(self, metrics: pd.DataFrame):
        pass
