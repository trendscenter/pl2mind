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

def default_hyperparams(input_dim=0):
    """
    Default parameter dictionary for the experiment.

    Parameters
    ----------
    input_dim: int, optional

    Returns
    -------
    hyperparams: dict
    """

    hyperparams = {
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
        "weight_decay": {
            "__builder__": "pylearn2.costs.mlp.L1WeightDecay",
            "coeffs": {"z": 0.01}
            },
        "niter": 1,
        "data_class": "MRI_Standard",
        "weight_decay": {
            "__builder__": "pylearn2.costs.dbm.L1WeightDecay",
            "coeffs": [0.01]
            }
        }
    return hyperparams

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

"""
Results to be plotted automatically.
Each dictionary entry gives a plot with name <key> and plots being each member of the
tuple value.
"""
plot_results = {
    "reconstruction_cost": ("train_reconstruction_cost",
                            "valid_reconstruction_cost"),

}

def extract_results(model, out_dir=None):
    """
    Function to extract result dictionary from model.
    Is called automatically by the experiment function or can be called externally.
    """

    channels = model.monitor.channels

    best_index = np.argmin(channels["valid_reconstruction_cost"].val_record)
    rd = dict((k + "_at_best", float(channels[k].val_record[best_index]))
              for k in channels.keys())
    rd.update(dict((k + "_at_end", float(channels[k].val_record[-1]))
                   for k in channels.keys()))
    rd.update(
        training_epochs=int(
            model.monitor.channels["train_reconstruction_cost"].epoch_record[-1]),
        training_batches=int(
            model.monitor.channels["train_reconstruction_cost"].batch_record[-1]),
        best_epoch=best_index,
        batch_num_at_best=int(
            model.monitor.channels["train_reconstruction_cost"].batch_record[best_index]),
        )
    if out_dir is not None:
        make_plots(model, out_dir)

    return rd
