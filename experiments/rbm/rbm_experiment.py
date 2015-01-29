import argparse
from jobman.tools import expand
from jobman.tools import flatten

import os
from os import path
import pylab as plt

from pl2mind.tools import mri_analysis

import numpy as np

# YAML template for the experiment.
yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")
name = "RBM Experiment"

fileparams = {
    "out_path": "${PYLEARN2_OUTS}"
}

default_hyperparams = {
    "nvis": input_dim,
    "nhid": 100,
    "dataset_name": "smri",
    "learning_rate": 0.001,
    "min_lr": 0.0001,
    "decay_factor": 1.0005,
    "batch_size": 10,
    "init_momentum": 0.0,
    "final_momentum": 0.5,
    "termination_criterion": {
        "__builder__": "pylearn2.termination_criteria.MonitorBased",
        "channel_name": "\"valid_reconstruction_cost\"",
        "prop_decrease": 0,
        "N": 20
        },
    "niter": 1,
    "data_class": "MRI_Standard",
    "weight_decay": {
        "__builder__": "pylearn2.costs.dbm.L1WeightDecay",
        "coeffs": [0.01]
        }
    }

def make_argument_parser():
    """
    Command-line parser for other scripts.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hyperparams")
    return parser

# List of results that will take priority in table when using the web interface.
results_of_interest = [
    "reconstruction_cost",
    "term_1_l1_weight_decay"
]