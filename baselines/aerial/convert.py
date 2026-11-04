"""Step 3 of the Aerial baseline: convert Aerial rules to the project's rule format.

Runs in the project environment. Usage, from the project root:
    python -m baselines.aerial.convert --run-dir results/aerial/<run>

Produces rules in the same format as RuleExtractor, so they can be passed directly to RuleEvaluator:
    {'antecedent': [(sensor_type, sensor_name, bin_idx), ...], 'consequent': (sensor_type, sensor_name, bin_idx)}

Unlike RuleNaps, Aerial already guarantees the following by construction (see aerial.rule_extraction._emit_rules
and the antecedent-growth loop in _generate_rules_core):
    - every rule has exactly one consequent item, so there is no tail to split
    - antecedent items are drawn from distinct sensors, and the consequent's sensor is never also an antecedent
      sensor, so there is no same-sensor-bin conflict to filter out
    - antecedent/consequent combinations are only ever generated once, so there are no exact duplicate rules
    - the antecedent size never exceeds rule_params['max_antecedents'] (train_aerial.py defaults this to
      config.max_antecedent_size; see _default_rule_params), so there is no antecedent-size limit to re-apply
So this step is just parsing: translating Aerial's {'feature': 'sensor_type;sensor_name', 'value': 'bin_idx'}
dicts back into item tuples.
"""
import argparse
import json
import os

Item = tuple[str, str, int]


def parse_item(d: dict) -> Item:
    """Translate one Aerial {'feature', 'value'} dict into a (sensor_type, sensor_name, bin_idx) item.

    'value' arrives as a string: train_aerial.py casts all columns to str before training, so that Aerial
    treats every discretized bin as categorical rather than silently re-discretizing numeric columns itself.
    """
    sensor_type, sensor_name = d['feature'].split(';')
    return (sensor_type, sensor_name, int(d['value']))


def convert(raw_rules: list[dict]) -> list[dict]:
    """Convert Aerial's rule dicts to project rules."""
    return [{'antecedent': [parse_item(a) for a in r['antecedents']], 'consequent': parse_item(r['consequent'])}
            for r in raw_rules]


def convert_run(run_dir: str) -> tuple[list[dict], dict]:
    """Convert the rules of one Aerial run (as written by train_aerial.py) and save them as converted_rules.json."""
    with open(os.path.join(run_dir, 'rules.json')) as f:
        raw = json.load(f)

    rules = convert(raw['rules'])
    stats = {'raw_rules': len(raw['rules']), 'converted_rules': len(rules)}

    # named differently from RuleNaps' convert.py output (rules.json): here rules.json is already taken by the
    # raw Aerial output written in train_aerial.py
    with open(os.path.join(run_dir, 'converted_rules.json'), 'w') as f:
        json.dump(rules, f)
    with open(os.path.join(run_dir, 'conversion_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    return rules, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Aerial rules to the project rule format')
    parser.add_argument('--run-dir', required=True, help='Aerial run directory containing rules.json')
    args = parser.parse_args()

    rules, stats = convert_run(args.run_dir)
    for k, v in stats.items():
        print(f"{k:<28} {v}")
    for rule in rules[:5]:
        print(f"  {rule['antecedent']} -> {rule['consequent']}")
