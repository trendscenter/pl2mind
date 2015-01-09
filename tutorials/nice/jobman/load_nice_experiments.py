"""
Module to run jobman over NICE hypermarameters.
"""

import argparse
import collections
import logging
from itertools import product

from jobman import api0
from jobman import sql
from jobman.tools import DD
from jobman.tools import expand
from jobman.tools import flatten
from jobman.tools import resolve

from math import exp
from math import log
import numpy as np
from os import path

from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets import MRI
from pylearn2.neuroimaging_utils.dataset_utils import mri_nifti
from pylearn2.neuroimaging_utils.tutorials.nice.jobman import nice_experiment
from pylearn2.utils import serial


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def hidden_generator(param_name, num, scale=100):
    assert num > 0
    assert isinstance(num, int)
    assert isinstance(scale, int)
    for hid in xrange(scale, scale * (num + 1), scale):
        yield (param_name, hid)

def float_generator(param_name, num, start, finish, log_scale=False):
    assert start > 0
    assert finish > 0
    assert isinstance(num, int)
    for n in xrange(num):
        if log_scale:
            yield (param_name, exp(log(start) + float(n) / float(num - 1) * (log(finish) - log(start))))
        else:
            yield (param_name, (start + float(n) / float(num - 1) * (finish - start)))

def layer_depth_generator(param_name, num, min_depth, max_depth):
    assert min_depth > 0
    assert max_depth > min_depth
    assert isinstance(min_depth, int)
    assert isinstance(max_depth, int)
    assert isinstance(num, (int, collections.Iterable))
    if isinstance(num, int):
        iterator = xrange(num, num + 1)
    else:
        iterator = num

    for num in iterator:
        assert num > 1
        for outer in xrange(min_depth, max_depth + 1):
            for inner in xrange(min_depth, max_depth + 1):
                yield (param_name, [outer] + [inner] * (num - 2) + [outer])

def nested_generator(*args):
    for x in product(*args):
        yield x

def load_experiments(args):
    dataset_name = args.dataset_name
    db = sql.db("postgres://%(user)s@%(host)s:%(port)d/%(database)s?table=%(table)s"
                % {"user": args.user,
                   "host": args.host,
                   "port": args.port,
                   "database": args.database,
                   "table": args.table,
                   })

    logger.info("Getting dataset info for %s" % dataset_name)
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + dataset_name)
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = (mask == 1).sum()
    if input_dim % 2 == 1:
        input_dim -= 1
    mri = MRI.MRI_Standard(which_set="full",
                           dataset_name=dataset_name,
                           unit_normalize=True,
                           even_input=True,
                           apply_mask=True)
    variance_map_file = path.join(data_path, "variance_map.npy")
    mri_nifti.save_variance_map(mri, variance_map_file)

    for items in nested_generator(layer_depth_generator("encoder.layer_depths", xrange(4, 6), 3, 4),
                                  hidden_generator("encoder.nhid", 4),
                                  float_generator("weight_decay.coeffs.z", 3, 0.1, 0.001, log_scale=True)):
#        logger.info("Adding NICE experiment with hyperparameters %s" % (items, ))
        state = DD()

        experiment_hyperparams = nice_experiment.default_hyperparams(input_dim)                
        for key, value in items:
            split_keys = key.split(".")
            entry = experiment_hyperparams
            for k in split_keys[:-1]:
                entry = entry[k]
            entry[split_keys[-1]] = value
        experiment_hyperparams["dataset_name"] = dataset_name
        h = abs(hash(frozenset(flatten(experiment_hyperparams).keys() +\
                                   [tuple(v) if isinstance(v, list) else v for v in flatten(experiment_hyperparams).values()])))

        user = path.expandvars("$USER")
        save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/%d"
                                      % (user, h))

        file_params = {
            "save_path": save_path,
            "variance_map_file": variance_map_file,
            }

        state.file_parameters = file_params
        state.hyper_parameters = experiment_hyperparams

        sql.insert_job(
            nice_experiment.experiment,
            flatten(state),
            db
            )

    db.createView("%s_view" % args.table)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("user")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("database")
    parser.add_argument("table")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    load_experiments(args)
