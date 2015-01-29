"""
Command-line module to run experiments
"""

import argparse
import imp
from pl2mind import experiments

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment module")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hyperparams", default = None,
                        help=("Comma separated list of "
                              "<key>:<value> pairs"))
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    experiment = imp.load_source("module.name", args.experiment)

    if args.hyperparams is not None:
        parsed = args.hyperparams.split(",")
        hyperparams = dict(tuple(p.split(":")) for p in parsed)
    else:
        hyperparams = {}

    experiments.run_experiment(experiment, **hyperparams)