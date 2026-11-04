"""Diagnostic: how much RuleNaps' raw rule format changes under each step of convert.py's pipeline.

Dry run on one already-trained RuleNaps run (no retraining): loads its raw .rules file and items.json, classifies
and removes degenerate rules step by step (empty rules, duplicates, structurally impossible/tautological rules),
evaluating the surviving rule set with RuleEvaluator at each checkpoint, then isolates the effect of the
antecedent-size limit and consequent-splitting steps individually and combined.

Being built incrementally, one step at a time. Currently implements: loading the raw rules and items,
classifying/removing empty rules, deduplicating exact (head, tail) duplicates, translating the survivors to
RuleEvaluator's format (unsplit, multi-item consequent), evaluating that full set (the "all valid-format rules,
including impossible/tautological ones" checkpoint), classifying/removing structurally invalid or tautological
rules, and running the 2x2 ablation (clean / antecedent-size limit only / consequent splitting only / both
combined) on the resulting "clean" set.

Usage, from the project root:
    python -m baselines.rulenaps_adapter.rule_format_analysis [--run-dir DIR]
"""
import argparse
import glob
import json
import os

import pandas as pd
import torch

from baselines.rulenaps_adapter.convert import Item, parse_rulenaps_rules
from baselines.rulenaps_adapter.export import get_output_dir
from src.data.leakdb.dataset import LeakDBDataset
from src.rules.evaluator import RuleEvaluator
from src.utils.config import Config

DEFAULT_STUDY_DIR = os.path.join('results', 'rulenaps', 'hpo', 'study_rulenaps_val_loss')
DEFAULT_OUT_DIR = os.path.join('results', 'rulenaps', 'rule_format_analysis')


def classify_empty_rules(raw_rules: list[tuple[list[int], list[int]]]):
    """Split raw rules by whether their head and/or tail is empty, and return the non-empty survivors, i.e., valid rule forms.

    'tail_only_empty' (non-empty head, empty tail) is provably impossible for RuleNaps: binarize_tensor_with_
    dynamic_thresholds only produces a non-empty head when its dynamic split exists, and that same split always
    guarantees a non-empty tail too (see the derivation discussed alongside this script). So this count is
    expected to always be 0 here — it's kept as its own category so the script stays correct on other baselines,
    where nothing guarantees that.
    """
    fully_empty = [(h, t) for h, t in raw_rules if not h and not t]   # empty head & tail rules
    head_only_empty = [(h, t) for h, t in raw_rules if not h and t]   # frequent itemsets: tail with no condition
    tail_only_empty = [(h, t) for h, t in raw_rules if h and not t]   # head with empty tail, expected to be 0 for RuleNaps
    non_empty = [(h, t) for h, t in raw_rules if h and t]
    counts = {
        'raw_rules': len(raw_rules),
        'fully_empty': len(fully_empty),
        'head_only_empty_frequent_itemset': len(head_only_empty),
        'tail_only_empty': len(tail_only_empty),
        'non_empty': len(non_empty),
    }
    return non_empty, counts


def deduplicate_rules(rules: list[tuple[list[int], list[int]]]):
    """Remove exact duplicate (head, tail) rules; head and tail are each treated as unordered sets.

    RuleNaps learns one rule per hidden neuron, and nothing in the architecture guarantees that different neurons
    converge to different rules — duplicates across neurons are expected, and worth counting on their own,
    separately from the structural-validity checks that come next. Sorting is done on the raw indices (not the
    translated items); since the index-to-item mapping is a bijection, grouping is identical either way.
    """
    seen = set()
    unique = []
    for h, t in rules:
        key = (tuple(sorted(h)), tuple(sorted(t)))  # sort the antecedent and consequent to ensure finding all semantically identical rules
        if key not in seen:
            seen.add(key)
            unique.append((h, t))
    counts = {
        'rules_before_dedup': len(rules),
        'duplicate_rules_removed': len(rules) - len(unique),
        'rules_after_dedup': len(unique),
    }
    return unique, counts


def translate_rules(rules: list[tuple[list[int], list[int]]], items: list[Item]) -> list[dict]:
    """Translate (head indices, tail indices) pairs into RuleEvaluator's format, keeping the tail unsplit (a list
    of items, not a single item) — RuleEvaluator.evaluate() accepts this directly, treating it as a conjunction
    like the antecedent. Use compute_hop_distance=False with these rules; annotate_hop_distances assumes a
    single-item consequent.
    """
    return [{'antecedent': [items[i] for i in h], 'consequent': [items[i] for i in t]} for h, t in rules]


def _sensor_bins(itemlist: list[Item]) -> dict[tuple[str, str], set[int]]:
    """Group a rule side's items (antecedent or consequent) by sensor, collecting the distinct bins mentioned."""
    d: dict[tuple[str, str], set[int]] = {}
    for st, sname, b in itemlist:
        d.setdefault((st, sname), set()).add(b)     # if sensor has not been encountered yet, initialize empty set. Adds the observed bin
    return d


def classify_structural_validity(rules: list[dict]):
    """Classify translated rules (consequent possibly multi-item) by sensor conflicts across the whole rule, and
    return the 'clean' survivors (rules with none of the three flags below).

    Three independent, non-mutually-exclusive checks per rule:
      - antecedent_internal_conflict: two distinct bins of one sensor both within the antecedent.
        support_ant == 0 by construction — the antecedent itself is unreachable.
      - tail_involving_conflict: two distinct bins of one sensor, with at least one of them in the consequent
        (tail-internal, or split across antecedent/consequent). support_ant may be > 0, but support_full is
        always 0 — confidence is deterministically 0 rather than undefined.
      - tautological_duplicate: the exact same (sensor, bin) item appears in both antecedent and consequent.
        support_full == support_ant, confidence == 1 — trivially true, adds no information. Provably impossible
        for RuleNaps itself (see classify_empty_rules' docstring), kept here for possible reuse on other baselines.
    """
    flagged = {'antecedent_internal_conflict': [], 'tail_involving_conflict': [], 'tautological_duplicate': []}
    clean = []
    for r in rules:
        ant_bins = _sensor_bins(r['antecedent'])
        con_bins = _sensor_bins(r['consequent'])

        ant_conflict = any(len(bins) > 1 for bins in ant_bins.values())
        tail_conflict = any(len(ant_bins.get(s, set()) | bins) > 1 for s, bins in con_bins.items())     # check if length of observed bins per item over entire rule > 1
        tautology = any(ant_bins.get(s, set()) & bins for s, bins in con_bins.items())      # check for every cons sensor if the same sensor with same bin occurs in ant

        if ant_conflict:
            flagged['antecedent_internal_conflict'].append(r)
        if tail_conflict:
            flagged['tail_involving_conflict'].append(r)
        if tautology:
            flagged['tautological_duplicate'].append(r)
        if not (ant_conflict or tail_conflict or tautology):
            clean.append(r)

    counts = {
        'rules_checked': len(rules),
        'antecedent_internal_conflict': len(flagged['antecedent_internal_conflict']),
        'tail_involving_conflict': len(flagged['tail_involving_conflict']),
        'tautological_duplicate': len(flagged['tautological_duplicate']),
        'clean': len(clean),
    }
    return clean, counts


def limit_antecedent_size(rules: list[dict], max_antecedent_size: int | None) -> list[dict]:
    """Drop rules whose antecedent is larger than max_antecedent_size (None disables the limit)."""
    if max_antecedent_size is None:
        return rules
    return [r for r in rules if len(r['antecedent']) <= max_antecedent_size]


def split_tails(rules: list[dict]) -> list[dict]:
    """Split each rule's (possibly multi-item) tail into one rule per consequent item.

    Deduplicates afterwards: splitting can make previously-distinct multi-item-tail rules collide on the same
    (antecedent, single consequent item) pair.
    """
    single = {(tuple(sorted(r['antecedent'])), c) for r in rules for c in r['consequent']}      # create set to avoid new duplicates duplicat
    return [{'antecedent': list(h), 'consequent': c} for h, c in sorted(single)]


EVAL_METRICS = ('support', 'support_ant', 'confidence', 'lift', 'zhang', 'coverage')


def safe_evaluate(evaluator: RuleEvaluator, rules: list[dict]) -> dict:
    """evaluator.evaluate(rules), but returns NaN averages instead of crashing when rules is empty.

    RuleEvaluator.evaluate() has no empty-input guard (only the evaluate_rules() wrapper in evaluate.py does,
    which this script bypasses by calling the evaluator directly for each ablation stage). An empty rule set is
    expected here — e.g. a trial's 'clean' set, or the antecedent-size limit applied to it, can genuinely leave
    zero rules — and the metrics are undefined then, not zero, hence NaN rather than 0.0.
    """
    if not rules:
        return {m: float('nan') for m in EVAL_METRICS}
    _, averages = evaluator.evaluate(rules)
    return averages


def discover_trial_dirs(study_dir: str) -> list[str]:
    """All trial_XXXX directories of an HPO study, in trial-number order."""
    return sorted(d for d in glob.glob(os.path.join(study_dir, 'trial_*')) if os.path.isdir(d))     # check for directories matching 'trial_.........' in the study directory provided.


def analyze_trial(config: Config, run_dir: str, evaluator: RuleEvaluator, items: list[Item],
                  max_antecedent_size: int | None) -> dict:
    """Run the full step-by-step pipeline (empty removal -> dedup -> translate -> structural validity ->
    2x2 ablation) on one already-trained RuleNaps run, and return every intermediate result as one dict.
    """
    with open(os.path.join(run_dir, 'run_summary.json')) as f:
        summary = json.load(f)
    raw_rules = parse_rulenaps_rules(summary['rules_file'])

    non_empty, empty_counts = classify_empty_rules(raw_rules)
    deduped, dedup_counts = deduplicate_rules(non_empty)
    translated = translate_rules(deduped, items)
    all_rules_averages = safe_evaluate(evaluator, translated)
    clean, validity_counts = classify_structural_validity(translated)

    ablation = {
        'clean': clean,
        'limit_only': limit_antecedent_size(clean, max_antecedent_size),
        'split_only': split_tails(clean),
    }
    ablation['both'] = split_tails(ablation['limit_only'])

    ablation_results = {}
    for stage, rules in ablation.items():
        ablation_results[stage] = {'n_rules': len(rules), **safe_evaluate(evaluator, rules)}

    return {
        'run_dir': run_dir,
        'hyperparameters': summary.get('overrides', {}),
        'max_antecedent_size': max_antecedent_size,
        'empty_rule_counts': empty_counts,
        'dedup_counts': dedup_counts,
        'all_rules_checkpoint': {'n_rules': len(translated), **all_rules_averages},
        'structural_validity_counts': validity_counts,
        'ablation': ablation_results,
    }


def flatten_result(trial_name: str, result: dict) -> dict:
    """Flatten one trial's analyze_trial() result into a single CSV row."""
    row = {'trial': trial_name, **result['hyperparameters'], 'max_antecedent_size': result['max_antecedent_size']}
    for group in ('empty_rule_counts', 'dedup_counts', 'all_rules_checkpoint', 'structural_validity_counts'):
        prefix = group.removesuffix('_counts').removesuffix('_checkpoint')
        for k, v in result[group].items():
            row[f'{prefix}.{k}'] = v
    for stage, metrics in result['ablation'].items():
        for k, v in metrics.items():
            row[f'{stage}.{k}'] = v
    return row


def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Step-by-step diagnostic of RuleNaps' rule conversion pipeline")
    parser.add_argument('--run-dir', default=None,
                        help='analyze a single RuleNaps run directory instead of a whole study (for quick testing)')
    parser.add_argument('--study-dir', default=DEFAULT_STUDY_DIR,
                        help=f'HPO study directory to sweep all trials of (default: {DEFAULT_STUDY_DIR})')
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                        help=f'where to write per-trial JSON files and the combined CSV (default: {DEFAULT_OUT_DIR})')
    parser.add_argument('--max-antecedent-size', type=int, default=config.max_antecedent_size,
                        help=f'default: config.max_antecedent_size ({config.max_antecedent_size}); use -1 for no limit')
    args = parser.parse_args()
    max_antecedent_size = None if args.max_antecedent_size == -1 else args.max_antecedent_size

    trial_dirs = [args.run_dir] if args.run_dir else discover_trial_dirs(args.study_dir)
    if not trial_dirs:
        raise FileNotFoundError(f"No trial_* directories found under {args.study_dir}")
    print(f"Analyzing {len(trial_dirs)} run(s)")

    with open(os.path.join(get_output_dir(config), 'items.json')) as f:
        items = [tuple(item) for item in json.load(f)]
    device = torch.device(config.device if config.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset = LeakDBDataset(config)
    evaluator = RuleEvaluator(dataset, config, device, split='train')

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for run_dir in trial_dirs:
        trial_name = os.path.basename(run_dir.rstrip(os.sep))
        if not os.path.exists(os.path.join(run_dir, 'run_summary.json')):
            print(f"  {trial_name:<14} skipped (no run_summary.json — failed trial)")
            continue
        result = analyze_trial(config, run_dir, evaluator, items, max_antecedent_size)
        with open(os.path.join(args.out_dir, f'{trial_name}.json'), 'w') as f:
            json.dump(result, f, indent=2)
        rows.append(flatten_result(trial_name, result))

        clean_m, both_m = result['ablation']['clean'], result['ablation']['both']
        print(f"  {trial_name:<14} clean n={clean_m['n_rules']:<4} conf={clean_m['confidence']:.3f} zhang={clean_m['zhang']:.3f}"
              f"  |  both n={both_m['n_rules']:<4} conf={both_m['confidence']:.3f} zhang={both_m['zhang']:.3f}")

    csv_path = os.path.join(args.out_dir, 'all_trials_summary.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nWrote {len(rows)} per-trial JSON file(s) and {csv_path}")


if __name__ == '__main__':
    main()
