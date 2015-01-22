import argparse
from jobman.tools import expand
from jobman.tools import flatten

import multiprocessing as mp
import os
from os import path

from pylearn2.config import yaml_parse
from pl2mind.tools import mri_analysis
from pylearn2.scripts.jobman.experiment import ydict

import numpy as np


yaml_file = path.join(path.abspath(path.dirname(__file__)), "mlp.yaml")

def default_hyperparams(input_dim=0):
    hyperparams = {
        "nvis": input_dim,
        "nhid1": 100,
        "nhid2": 100,
        "dataset_name": "smri",
        "learning_rate": 0.001,
        "batch_size": 10,
        "init_momentum": 0.1,
        "final_momentum": 0.9,
        "termination_criterion": {
            "__builder__": "pylearn2.termination_criteria.MonitorBased",
            "channel_name": "\"test_y_misclass\"",
            "prop_decrease": 0,
            "N": 20
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

def extract_results(model):
    """
    Function to extract result dictionary from model.
    Is called automatically by the experiment function or can be called externally.
    """

    channels = model.monitor.channels
    test_cost = channels["test_objective"]

    best_index = np.argmin(
        test_cost.val_record)
    batch_num = test_cost.batch_record[best_index]

    rd = dict(
        training_epochs=int(model.monitor.channels["train_objective"].epoch_record[-1]),
        training_batches=int(model.monitor.channels["train_objective"].batch_record[-1]),
        best_epoch=best_index,
        batch_num_at_best=batch_num,
        train_objective_end_epoch=float(model.monitor.channels["train_objective"].val_record[-1]),
        test_objective_best=float(test_cost.val_record[best_index]),
        test_misclass_best=float(model.monitor.channels["test_y_misclass"].val_record[best_index]),
        train_misclass_end_epoch=float(model.monitor.channels["train_y_misclass"].val_record[-1])
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
