import argparse
from jobman.tools import expand
from jobman.tools import flatten

import os
from os import path
import pylab as plt

from pylearn2.config import yaml_parse
from pl2mind.tools import mri_analysis
from pylearn2.scripts.jobman.experiment import ydict

import numpy as np


yaml_file = path.join(path.abspath(path.dirname(__file__)), "nice_mri.yaml")
name = "NICE Experiment"

fileparams = {
    "out_path": "${PYLEARN2_OUTS}"
}

default_hyperparams = {
    "nvis": None,
    "dataset_name": "smri",
    "demean": False,
    "variance_normalize": False,
    "learning_rate": 0.001,
    "min_lr": 0.0001,
    "decay_factor": 1.0005,
    "batch_size": 10,
    "init_momentum": 0.0,
    "final_momentum": 0.5,
    "gaussian_noise": 0.1,
    "termination_criterion": {
        "__builder__": "pylearn2.termination_criteria.MonitorBased",
        "channel_name": "\"valid_objective\"",
        "prop_decrease": 0,
        "N": 20
        },
    "encoder": {
        "__builder__":
            "pl2mind.models.nice_mlp.Simple_TriangularMLP",
        "layer_name": "encoder",
        "layer_depths": [2, 4, 4, 2],
        "nvis": None,
        "nhid": 400
        },
    "prior": {
        "__builder__": "nice.pylearn2.models.nice.StandardNormal"
        },
    "data_class": "MRI_Standard",
    "weight_decay": {
        "__builder__": "nice.pylearn2.costs.log_likelihood.SigmaPenalty",
        "coeff": 0.01
        }
    }

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--learning_rate", default=None)
    parser.add_argument("-b", "--batch_size", default=None)
    return parser

analyze_fn = mri_analysis.main

results_of_interest = [
    "z_S_stddev",
    "z_S_over_2_stdev",
    "z_S_over_1_stdev",
    "objective",
    "cumulative_sum",
    "term_0",
    "term_1_sigma_l1_penalty"
]

outputs = ["montage", "spectrum"]