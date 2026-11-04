"""Step 3 of the RuleNaps baseline: convert RuleNaps rules to the project's rule format.

Runs in the project environment. Usage, from the project root:
    python -m baselines.rulenaps_adapter.convert --run-dir results/rulenaps/<run> [--max-antecedent-size N]

Produces rules in the same format as RuleExtractor, so they can be passed directly to RuleEvaluator:
    {'antecedent': [(sensor_type, sensor_name, bin_idx), ...], 'consequent': (sensor_type, sensor_name, bin_idx)}

RuleNaps learns one rule per hidden neuron, of the form head -> tail with possibly several tail items. The conversion
applies the same constraints as FPExtractor, so the rule sets of all methods are comparable:
    1. drop rules with an empty head or tail (neurons that learned no rules)
    2. drop rules whose head contains the same sensor twice (two bins of one sensor never co-occur in one-hot data)
    3. drop rules whose head is larger than max_antecedent_size (None disables this limit)
    4. split head -> {c1, c2, ...} into head -> c1, head -> c2, ...
    5. drop rules whose consequent sensor also appears in the antecedent
    6. remove duplicate rules
"""
import argparse
import json
import os
import re
from collections import Counter

from baselines.rulenaps_adapter.export import get_output_dir
from src.utils.config import Config

Item = tuple[str, str, int]


def parse_rulenaps_rules(rules_file: str) -> list[tuple[list[int], list[int]]]:
    """Parse a RuleNaps .rules file into (head indices, tail indices) pairs, one per line.

    Lines look like "[272 273]  ->  274" or "[]  ->  []": numpy prints single items without brackets.
    """
    raw_rules = []
    with open(rules_file) as f:
        for line in f:
            head, tail = line.split('->')
            raw_rules.append(([int(i) for i in re.findall(r'\d+', head)],   # matches consecutive decimal digits, i.e., without whitespace
                              [int(i) for i in re.findall(r'\d+', tail)]))  # matches consecutive decimal digits, i.e., without whitespace
    return raw_rules


def convert(raw_rules: list[tuple[list[int], list[int]]], items: list[Item],
            max_antecedent_size: int | None) -> tuple[list[dict], dict]:
    """Convert (head, tail) index pairs to project rules. Returns the rules and the number of rules removed per step."""
    stats = {'raw_rules': len(raw_rules)}

    # 1. empty heads or tails
    rules = [(h, t) for h, t in raw_rules if h and t]
    stats['after_removing_empty'] = len(rules)

    # translate indices to (sensor_type, sensor_name, bin_idx) items
    rules = [([items[i] for i in h], [items[i] for i in t]) for h, t in rules]
    stats['head_size_distribution'] = dict(sorted(Counter(len(h) for h, _ in rules).items()))

    # 2. Remove rules having items from the same sensor in the antecedent
    rules = [(h, t) for h, t in rules if len({(st, sname) for st, sname, _ in h}) == len(h)] # for each rule check whether the set of ant items equals list of ant items to remove rules with duplicates sensor items
    stats['after_removing_duplicate_sensor_heads'] = len(rules)

    # 3. head size limit
    if max_antecedent_size is not None:
        rules = [(h, t) for h, t in rules if len(h) <= max_antecedent_size]
    stats['after_max_antecedent_size'] = len(rules)

    # 4. split multi-item tails into single-consequent rules
    single = [(h, c) for h, t in rules for c in t]
    stats['after_splitting_tails'] = len(single)

    # 5. consequent sensor must not be an antecedent sensor
    single = [(h, c) for h, c in single if (c[0], c[1]) not in {(st, sname) for st, sname, _ in h}]
    stats['after_removing_consequent_sensor_in_antecedent'] = len(single)

    # 6. duplicates; antecedents are sorted so that the same itemset always has the same order, set creation automatically removes duplicates
    unique = {(tuple(sorted(h)), c) for h, c in single}
    converted = [{'antecedent': list(h), 'consequent': c} for h, c in sorted(unique)]
    stats['final_rules'] = len(converted)

    return converted, stats


def convert_run(config: Config, run_dir: str, max_antecedent_size: int | None) -> tuple[list[dict], dict]:
    """Convert the rules of one RuleNaps run (as written by run.py) and save them as rules.json in the run dir."""
    with open(os.path.join(run_dir, 'run_summary.json')) as f:
        summary = json.load(f)
    with open(os.path.join(get_output_dir(config), 'items.json')) as f:
        items = [tuple(item) for item in json.load(f)]

    rules, stats = convert(parse_rulenaps_rules(summary['rules_file']), items, max_antecedent_size)
    stats['max_antecedent_size'] = max_antecedent_size

    with open(os.path.join(run_dir, 'rules.json'), 'w') as f:
        json.dump(rules, f)
    with open(os.path.join(run_dir, 'conversion_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return rules, stats


if __name__ == '__main__':
    config = Config()
    parser = argparse.ArgumentParser(description='Convert RuleNaps rules to the project rule format')
    parser.add_argument('--run-dir', required=True, help='RuleNaps run directory containing run_summary.json')
    parser.add_argument('--max-antecedent-size', type=int, default=config.max_antecedent_size,
                        help=f'default: config.max_antecedent_size ({config.max_antecedent_size}); use -1 for no limit')
    args = parser.parse_args()

    limit = None if args.max_antecedent_size == -1 else args.max_antecedent_size
    rules, stats = convert_run(config, args.run_dir, limit)
    for k, v in stats.items():
        print(f"{k:<48} {v}")
    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")
