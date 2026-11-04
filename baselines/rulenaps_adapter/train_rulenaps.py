"""Step 2 of the RuleNaps baseline (RuleNaps side): train RuleNaps on a .dat file and write the raw rules.

Runs in the RuleNaps environment (Python 3.9, baselines/rulenaps/.venv) with baselines/rulenaps/src on the path.
It is launched by run.py and should not import anything from the project's own src/.

Mirrors the authors' experiment pipeline (perform_grid_search_for_one_parameter_setting): BASE_PARAMETERS from
utils/experiments/experiments.py, updated with the given overrides, trained with learn_destructive_net.
RuleNaps' own main.py is deliberately not used, as its defaults collapse to an empty rule set.

Writes to --out-dir:
    rulenaps_<params>.rulenaps.rules   raw rules, one per hidden neuron ("[head items]  ->  [tail items]")
    rulenaps_<params>_enc_weight.pt    trained encoder weights
    rulenaps_<params>_loss.log         training/validation loss
    rulenaps_parameters.json           full RuleNaps parameter set
    run_summary.json                   file locations, overrides, and runtime; read by the next step
"""
import argparse
import json
import os
import sys
from timeit import default_timer as timer

RULENAPS_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rulenaps', 'src'))
sys.path.insert(0, RULENAPS_SRC)

import torch  # noqa: E402

import utils.seed  # noqa: E402
import utils.io  # noqa: E402
import utils.result_generation  # noqa: E402
from utils.experiments.experiments import BASE_PARAMETERS  # noqa: E402
from model.parameters import get_new_updated_instance_given_parameter_dict  # noqa: E402
from model.network import learn_destructive_net  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description='Train RuleNaps on a .dat file')
    parser.add_argument('--data', required=True, help='path to the .dat file')
    parser.add_argument('--out-dir', required=True, help='directory for all output files')
    parser.add_argument('--params', default='{}', help='JSON dict of NetworkParameters overriding BASE_PARAMETERS')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()

    overrides = json.loads(args.params)
    os.makedirs(args.out_dir, exist_ok=True)
    utils.seed.fix_random_seed(args.seed)
    torch.set_num_threads(args.threads)

    parameters = get_new_updated_instance_given_parameter_dict(BASE_PARAMETERS, overrides)
    # all RuleNaps output file names are derived from base_name (+ a parameter suffix); no ground truth is available
    base_name = os.path.join(os.path.abspath(args.out_dir), 'rulenaps')
    parameters.update_filenames(os.path.abspath(args.data), base_name, None)
    _, loss_file, result_file, _, _ = parameters.get_names_of_result_files()
    parameters.save_as_json()

    start = timer()
    _, weights_head, _, _, train_data = learn_destructive_net(parameters)
    training_time = timer() - start
    utils.result_generation.print_found_rules_for_destructive_net_weights(result_file, train_data, weights_head, verbose=False)
    total_time = timer() - start

    with open(result_file) as f:
        lines = f.read().splitlines()
    n_complete = sum(1 for line in lines if not line.replace(' ', '').startswith('[]') and not line.replace(' ', '').endswith('[]'))

    # The authors' hp tuning objective (utils/experiments/tuning.py, objective_from_config): the unregularized
    # reconstruction loss of the final model on RuleNaps' held-out rows (the last validation_set_size of the .dat),
    # which is the last value in the loss log. The log is only written when perform_validation_runs is enabled.
    final_validation_loss = None
    if parameters.perform_validation_runs:
        final_validation_loss = utils.io.parse_log_file(loss_file, ground_truth_evaluated=False)[-1]

    summary = {
        'data_file': os.path.abspath(args.data),
        'rules_file': result_file,
        'weights_file': parameters.get_weights_file_name() if parameters.save_weights else None,
        'loss_file': loss_file,
        'overrides': overrides,
        'seed': args.seed,
        'threads': args.threads,
        'final_validation_loss': final_validation_loss,
        'runtime_training_s': round(training_time, 1),
        'runtime_total_s': round(total_time, 1),
        'n_rule_lines': len(lines),
        'n_rules_with_head_and_tail': n_complete,
    }
    with open(os.path.join(args.out_dir, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
