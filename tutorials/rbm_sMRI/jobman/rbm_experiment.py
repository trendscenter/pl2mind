import argparse
from jobman.tools import expand
from jobman.tools import flatten

import os
from os import path
import pylab as plt

from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.tools import mri_analysis
from pylearn2.scripts.jobman.experiment import ydict

import numpy as np

# YAML template for the experiment.
yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")

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
    parser.add_argument("-l", "--learning_rate", default=None)
    parser.add_argument("-b", "--batch_size", default=None)
    return parser

# List of results that will take priority in table when using the web interface.
results_of_interest = [
    "valid_reconstruction_cost_at_end",
    "valid_term_1_l1_weight_decay_at_end"
]

""" 
Results to be plotted automatically.
Each dictionary entry gives a plot with name <key> and plots being each member of the
tuple value.
"""
plot_results = {
    "reconstruction_cost": ("train_reconstruction_cost", "valid_reconstruction_cost"),
}

def make_plots(model, out_dir):
    """
    Function to plot results.
    TODO: move outside of experiment into tools.
    """

    if not path.isdir(out_dir):
        os.mkdir(out_dir)
    channels = model.monitor.channels
    for key, result in plot_results.iteritems():
        plt.close()
        handles = []
        base = [float(a) for a in channels[result[0]].val_record]
        x = range(len(base))
        for r in result:
            l, = plt.plot(x, channels[r].val_record, label=r)
            handles.append(l)
        plt.legend(handles)

        y_min = sorted(base)[max(2, len(base) - 1)]
        y_max = sorted(base)[-(max(2, len(base) - 1))]
        plt.axes([x[0], x[-1], y_min, y_max])
        plt.savefig(path.join(out_dir, "%s.pdf" % key))

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

def experiment(state, channel):
    """
    Experiment function.
    Used by jobman to run jobs. Must be loaded externally.

    Parameters
    ----------
    state: WRITEME
    channel: WRITEME
    """

    yaml_template = open(yaml_file).read()
    hyper_parameters = expand(flatten(state.hyper_parameters), dict_type=ydict)
    
    file_params = expand(flatten(state.file_parameters), dict_type=ydict)
    # Hack to fill in file parameter strings first
    for param in file_params:
        yaml_template = yaml_template.replace("%%(%s)s" % param, file_params[param])

    yaml = yaml_template % hyper_parameters
    train_object = yaml_parse.load(yaml)

    state.pid = os.getpid()
    channel.save() # Need to save channel or else PID wont make it to the table
    train_object.main_loop()

    state.results = extract_results(train_object.model)
    return channel.COMPLETE
