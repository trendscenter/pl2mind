"""
Command-line module to run experiments
"""

import argparse
import imp
from pl2mind import experiments
import warnings


warnings.filterwarnings("ignore")

def make_argument_parser():
    """
    Arg parser for simple runner.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="Experiment module")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hyperparams", default = None,
                        help=("Comma separated list of "
                              "<key>:<value> pairs"))
    parser.add_argument("-k", "--keep", action="store_true",
                        help="Model checkpoints are deleted after training by "
                        "default. Add this flag to keep after training.")

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

    experiments.run_experiment(experiment, keep=args.keep, **hyperparams)