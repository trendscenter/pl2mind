import argparse
from jobman.tools import expand
from jobman.tools import flatten

from os import path

from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.tools import mri_analysis
from pylearn2.scripts.jobman.experiment import ydict

import numpy as np


yaml_file = path.join(path.abspath(path.dirname(__file__)), "nice_mri.yaml")

def default_hyperparams(input_dim=0):
    hyperparams = {
        "nvis": input_dim,
        "dataset_name": "smri",
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
            "__builder__": "pylearn2.neuroimaging_utils.models.nice_mlp.Simple_TriangularMLP",
            "layer_name": "encoder",
            "layer_depths": [2, 4, 4, 2],
            "nvis": input_dim,
            "nhid": 200
            },
        "prior": {
            "__builder__": "nice.pylearn2.models.nice.StandardNormal"
            },
        "data_class": "MRI_Transposed",
        "weight_decay": {
            "__builder__": "pylearn2.costs.mlp.L1WeightDecay",
            "coeffs": {"z": 0.01}
            }
        }
    return hyperparams

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-l", "--learning_rate", default=None)
    parser.add_argument("-b", "--batch_size", default=None)
    parser.add_argument("-1", "--l1_decay", default=0.0, type=float)
    return parser

results_of_interest = [
    "valid_z_S_stddev_at_end",
    "valid_z_S_over_2_stdev_at_end",
    "valid_z_S_over_1_stdev_at_end",
    "valid_term_1_l1_penalty_at_end",
    "valid_objective_at_end",
    "valid_cumulative_sum_at_end"
]

def extract_results(model):
    """
    Function to extract result dictionary from model.
    Is called automatically by the experiment function or can be called externally.
    """

    channels = model.monitor.channels

    best_index = np.argmin(channels["valid_objective"].val_record)
    rd = dict((k + "_at_best", float(channels[k].val_record[best_index])) for k in channels.keys())
    rd.update(dict((k + "_at_end", float(channels[k].val_record[-1])) for k in channels.keys()))
    rd.update(
        training_epochs=int(model.monitor.channels["train_objective"].epoch_record[-1]),
        training_batches=int(model.monitor.channels["train_objective"].batch_record[-1]),
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
#    with open("/na/homes/dhjelm/pylearn2/pylearn2/jobman/nice_lr_search/%d.yaml"
#              % state.id, "w") as f:
#        f.write(yaml)
    train_object = yaml_parse.load(yaml)
    
    train_object.main_loop()
    state.results = extract_results(train_object.model)
    return channel.COMPLETE
