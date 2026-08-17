import torch

from src.data.dataset import LeakDBDataset
from src.utils.config import Config

SUPPORTED_METRICS = {'support', 'confidence', 'lift', 'zhang', 'coverage'}


class RuleEvaluator:
    def __init__(self, dataset: LeakDBDataset, config: Config, device: torch.device):
        self.config = config
        self.device = device
        self.item_to_col = {item: col for col, item in enumerate(dataset.test_items)}
        # float for matrix multiply; shape (n_test, n_items)
        self.T = dataset.test_tensor.to(device).float()

    def evaluate(self, rules: list[dict], metrics: list[str] = None) -> tuple[list[dict], dict]:
        if metrics is None:
            metrics = list(SUPPORTED_METRICS)
        unknown = set(metrics) - SUPPORTED_METRICS
        if unknown:
            raise ValueError(f"Unknown metrics: {unknown}. Supported: {SUPPORTED_METRICS}")

        # One Python pass to extract column indices per rule — avoids building three large
        # (n_rules, n_items) mask tensors upfront, which would be gigabytes for millions of rules
        ant_cols  = [[self.item_to_col[tuple(i)] for i in r['antecedent']] for r in rules]
        con_cols  = [[self.item_to_col[tuple(r['consequent'])]] for r in rules]
        full_cols = [a + c for a, c in zip(ant_cols, con_cols)]

        averages = {}
        support_full = support_ant = support_con = confidence = None

        # support_full is an input to every other metric
        if metrics:
            support_full, _ = self._compute_support(full_cols)
            averages['support'] = round(support_full.mean().item(), 4)
            for rule, v in zip(rules, support_full.tolist()):
                rule['support'] = round(v, 4)

        if 'confidence' in metrics or 'lift' in metrics or 'zhang' in metrics or 'coverage' in metrics:
            support_ant, covered = self._compute_support(ant_cols, track_coverage='coverage' in metrics)
            if 'coverage' in metrics:
                averages['coverage'] = round(covered.float().mean().item(), 4)
            if 'confidence' in metrics or 'lift' in metrics or 'zhang' in metrics:
                confidence = self._compute_confidence(support_full, support_ant)
                averages['confidence'] = round(confidence.mean().item(), 4)
                for rule, v in zip(rules, confidence.tolist()):
                    rule['confidence'] = round(v, 4)

        if 'lift' in metrics or 'zhang' in metrics:
            support_con, _ = self._compute_support(con_cols)

        if 'lift' in metrics:
            lift = self._compute_lift(confidence, support_con)
            averages['lift'] = round(lift.mean().item(), 4)
            for rule, v in zip(rules, lift.tolist()):
                rule['lift'] = round(v, 4)

        if 'zhang' in metrics:
            zhang = self._compute_zhang(support_full, support_ant, support_con)
            averages['zhang'] = round(zhang.mean().item(), 4)
            for rule, v in zip(rules, zhang.tolist()):
                rule['zhang'] = round(v, 4)

        if self.config.filter_rules:
            rules = self._filter(rules, metrics)
        rules.sort(key=lambda r: r.get('confidence', r.get('support', 0)), reverse=True)
        return rules, averages

    def _build_masks(self, rule_cols_chunk: list[list[int]]) -> torch.Tensor:
        """Build a (chunk_size, n_items) bool mask for one chunk of rules.

        Index lists are assembled on CPU first, then written to the GPU tensor in a
        single scatter op (m[row_idx, col_idx] = True). This replaces one GPU write
        per rule item with one GPU write per chunk, reducing kernel launches from
        chunk_size × items_per_rule down to 1.
        """
        c = len(rule_cols_chunk)
        n_items = self.T.shape[1]

        row_idx, col_idx = [], []

        # For each rule in the chunk obtain the index combinations per item, i.e. row x col 
        for i, cols in enumerate(rule_cols_chunk):
            row_idx.extend([i] * len(cols))  # creates [i, i , i ....]
            col_idx.extend(cols)

        # prevent small GPU writes by assigning True once for entire chunk.
        m = torch.zeros(c, n_items, dtype=torch.bool, device=self.device)
        m[row_idx, col_idx] = True
        return m

    def _compute_support(self, rule_cols: list[list[int]], chunk_size: int = 512, track_coverage: bool = False):
        """Compute support for each rule via chunked matmul.

        Processing all rules at once would require an O(n_test × n_rules) float tensor —
        infeasible at millions of rules. Chunking keeps peak memory at O(n_test × chunk_size)
        while retaining fully vectorised matmul within each chunk.

        When track_coverage=True, accumulates a (n_test,) bool vector via logical_or_ across
        chunks — O(n_test) memory — indicating which transactions are covered by at least one
        rule. Returns (support, covered); covered is None when track_coverage=False.
        """
        n_test = self.T.shape[0]
        parts = []
        covered = torch.zeros(n_test, dtype=torch.bool, device=self.device) if track_coverage else None

        for start in range(0, len(rule_cols), chunk_size):
            m = self._build_masks(rule_cols[start:start + chunk_size])
            counts    = self.T @ m.float().T     # (n_test, c)
            required  = m.float().sum(dim=1)     # (c,)
            satisfied = counts == required        # (n_test, c) bool
            if track_coverage:
                covered.logical_or_(satisfied.any(dim=1))
            parts.append(satisfied.float().mean(dim=0))  # (c,)

        return torch.cat(parts), covered          # (n_rules,), (n_test,) or None

    def _compute_confidence(self, support_full: torch.Tensor, support_ant: torch.Tensor) -> torch.Tensor:
        # avoid division by zero for antecedents that never appear in test data
        return torch.where(support_ant > 0, support_full / support_ant, torch.zeros_like(support_full))

    def _compute_lift(self, confidence: torch.Tensor, support_con: torch.Tensor) -> torch.Tensor:
        return torch.where(support_con > 0, confidence / support_con, torch.zeros_like(confidence))

    def _compute_zhang(self, support_full: torch.Tensor, support_ant: torch.Tensor, support_con: torch.Tensor) -> torch.Tensor:
        numerator = support_full - support_ant * support_con
        denom = torch.max(
            support_full * (1 - support_ant),
            support_ant * (support_con - support_full),
        )
        return torch.where(denom > 0, numerator / denom, torch.zeros_like(numerator))

    def _filter(self, results: list[dict], metrics: list[str]) -> list[dict]:
        filtered = []
        for r in results:
            if 'support' in metrics and r.get('support', 1) < self.config.min_support:
                continue
            if 'confidence' in metrics and r.get('confidence', 1) < self.config.min_confidence:
                continue
            filtered.append(r)
        return filtered
