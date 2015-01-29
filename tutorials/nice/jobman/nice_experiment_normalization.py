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

def default_hyperparams(input_dim=0):
    hyperparams = {
        "nvis": input_dim,
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
            "nvis": input_dim,
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
    return hyperparams

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--learning_rate", default=None)
    parser.add_argument("-b", "--batch_size", default=None)
    return parser

analysis_fn = mri_analysis.main

results_of_interest = [
    "valid_z_S_stddev_at_best",
    "valid_z_S_over_2_stdev_at_best",
    "valid_z_S_over_1_stdev_at_best",
    "valid_objective_at_best",
    "valid_objective_at_end",
    "valid_cumulative_sum_at_best",
    "valid_term_0_at_best",
    "valid_term_1_sigma_l1_penalty_at_best"
]

plot_results = {
    "objective": ("train_objective", "valid_objective"),
    "over_std": ("train_z_S_over_2_stdev", "valid_z_S_over_2_stdev",
                 "train_z_S_over_1_stdev", "valid_z_S_over_1_stdev"),
    "cumulative_sum": ("train_cumulative_sum", "valid_cumulative_sum"),
    "sigma_l1_penalty": ("train_term_1_sigma_l1_penalty",
                         "valid_term_1_sigma_l1_penalty")
}

def make_plots(model, out_dir):
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

        y_min = sorted(base)[2]
        y_max = sorted(base)[-2]
        plt.axes([x[0], x[-1], y_min, y_max])
        plt.savefig(path.join(out_dir, "%s.pdf" % key))

def extract_results(model):
    """
    Function to extract result dictionary from model.
    Is called automatically by the experiment function
    or can be called externally.
    """

    channels = model.monitor.channels

    best_index = np.argmin(channels["valid_objective"].val_record)
    rd = dict((k + "_at_best", float(channels[k].val_record[best_index]))
        for k in channels.keys())
    rd.update(dict((k + "_at_end", float(channels[k].val_record[-1]))
        for k in channels.keys()))
    rd.update(
        training_epochs=int(
            model.monitor.channels["train_objective"].epoch_record[-1]),
        training_batches=int(
            model.monitor.channels["train_objective"].batch_record[-1]),
        best_epoch=best_index,
        batch_num_at_best=model.monitor.channels["train_objective"].batch_record[best_index],
        )

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
    channel.save()
    train_object.main_loop()

    state.results = extract_results(train_object.model)
    return channel.COMPLETE
