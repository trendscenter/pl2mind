import argparse
from os import path

# YAML template for the experiment.
yaml_file = path.join(path.abspath(path.dirname(__file__)), "initial.yaml")
# Name for your experiment
name = "First DAE Experiment"

# File parameters to keep them separate from hyperparams.
fileparams = {
    "out_path": "${PYLEARN2_OUTS}"
}

# Default hyperparams.
default_hyperparams = {
    "nvis": 0,
    "nhid": 100,
    "dataset_name": "smri",
    "learning_rate": 0.001,
    "batch_size": 10,
    "act_enc": "tanh",
    "act_dec": "null",
    "corruptor": {
        "__builder__": "pl2mind.datasets.MRI.GaussianMRICorruptor",
        "stdev": 0.2,
        "variance_map": "!pkl: \"%(variance_map_file)s\""
        },
    "termination_criterion": {
        "__builder__": "pylearn2.termination_criteria.EpochCounter",
        "max_epochs": 10
        },
    "data_class": "MRI_Standard",
    }

# An argument parser to customize running.
def make_argument_parser():
    """
    Command-line parser for other scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hyperparams")
    return parser

# List of results that will take priority when using the web interfaces.
results_of_interest = [
    "objective"
]
