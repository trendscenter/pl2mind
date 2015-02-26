import argparse
from jobman.tools import expand
from jobman.tools import flatten

import os
from os import path

from pl2mind.tools import mri_analysis


# YAML template for the experiment.
yaml_file = path.join(path.abspath(path.dirname(__file__)), "mlp.yaml")
name = "Gen Experiment"

fileparams = {
    "out_path": "${PYLEARN2_OUTS}"
}

default_hyperparams = {
    "nvis": None,
    "nhid1": 100,
    "nhid2": 100,
    "source_file": None,
    "mixing_file": None,
    "use_real": True,
    "dataset_name": "smri",
    "num_components": 10,
    "learning_rate": 0.0001,
    "min_lr": 0.00001,
    "decay_factor": 1.0005,
    "batch_size": 10,
    "init_momentum": 0.0,
    "final_momentum": 0.5,
    "termination_criterion": {
        "__builder__": "pylearn2.termination_criteria.MonitorBased",
        "channel_name": "\"valid_y_misclass\"",
        "prop_decrease": 0,
        "N": 2000
        },
    "data_class": "MRI_Standard"
    }

default_hyperparams["source_file"] = "/na/homes/dhjelm/tmp/ica_sources.npy"
default_hyperparams["mixing_file"] = "/na/homes/dhjelm/tmp/ica_mixing.npy"


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
    "y_misclass"
]

# Set the analysis function.
analyze_fn = mri_analysis.main
outputs = ["montage"]
