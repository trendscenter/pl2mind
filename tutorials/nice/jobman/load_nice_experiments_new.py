"""
Module to run jobman over NICE hypermarameters.
"""

import argparse
import collections
import logging
import imp

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
from pl2mind.datasets import MRI
from pl2mind.dataset_utils import mri_nifti
from pl2mind.tools import jobman_generators as jg
from pl2mind.tutorials.nice.jobman\
    import nice_experiment_normalization as nice_experiment
from pylearn2.utils import serial


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

class InputHandler(object):
    def __init__(self):
        self.d = {}

    def get_input_params(self, args, hyperparams):
        data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" +
                                      args.dataset_name)

        data_class = hyperparams["data_class"]
        variance_normalize = hyperparams.get("variance_normalize", False)
        unit_normalize = hyperparams.get("unit_normalize", False)
        demean = hyperparams.get("demean", False)
        assert not (variance_normalize and unit_normalize)

        logger.info((data_class, variance_normalize, unit_normalize, demean))
        h = hash((data_class, variance_normalize, unit_normalize, demean))

        if self.d.get(h, False):
            return self.d[h]
        else:
            if data_class == "MRI_Transposed":
                assert not variance_normalize
                mri = MRI.MRI_Transposed(dataset_name=args.dataset_name,
                                         unit_normalize=unit_normalize,
                                         demean=demean,
                                         even_input=True,
                                         apply_mask=True)
                input_dim = mri.X.shape[1]
                variance_file_name = ("variance_map_transposed%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_dm" if demean else ""))

            elif data_class == "MRI_Standard":
                assert not demean
                mask_file = path.join(data_path, "mask.npy")
                mask = np.load(mask_file)
                input_dim = (mask == 1).sum()
                if input_dim % 2 == 1:
                    input_dim -= 1
                mri = MRI.MRI_Standard(which_set="full",
                                       dataset_name=args.dataset_name,
                                       unit_normalize=unit_normalize,
                                       variance_normalize=variance_normalize,
                                       even_input=True,
                                       apply_mask=True)
                variance_file_name = ("variance_map%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_vn" if variance_normalize else ""))
                logger.info(variance_file_name)
                logger.info((data_class,
                             variance_normalize,
                             unit_normalize,
                             demean))

        variance_map_file = path.join(data_path, variance_file_name)
        if not path.isfile(variance_map_file):
            logger.info("Saving variance file %s" % variance_map_file)
            mri_nifti.save_variance_map(mri, variance_map_file)
        self.d[h] = (input_dim, variance_map_file)
        return self.d[h]

def load_experiments(args):

    dataset_name = args.dataset_name
    db = sql.db("postgres://%(user)s@%(host)s:"
                "%(port)d/%(database)s?table=%(table)s"
                % {"user": args.user,
                   "host": args.host,
                   "port": args.port,
                   "database": args.database,
                   "table": args.table,
                   })
    input_handler = InputHandler()

    for items in jg.nested_generator(
        jg.list_generator("encoder.layer_depths",
                          [[3, 5, 5, 5, 3], [5, 5, 5, 5, 5], [2, 4, 4, 2]]),
        jg.list_generator("variance_normalize", [False, 2]),
        jg.float_generator("weight_decay.coeff", 4, 0.1, 0.0001,
                           log_scale=True),
        jg.list_generator("prior.__builder__",
                          ["nice.pylearn2.models.nice.StandardNormal",
                           "nice.pylearn2.models.nice.StandardLogistic"])):

        logger.info("Adding NICE experiment across hyperparameters %s"
                    % (items, ))
        state = DD()

        experiment_hyperparams = nice_experiment.default_hyperparams()

        for key, value in items:
            split_keys = key.split(".")
            entry = experiment_hyperparams
            for k in split_keys[:-1]:
                entry = entry[k]
            assert split_keys[-1] in entry,\
                ("Key not found in hyperparams: %s, "
                 "found: %s" % (split_keys[-1], entry.keys()))
            entry[split_keys[-1]] = value
        experiment_hyperparams["dataset_name"] = dataset_name
        input_dim, variance_map_file = input_handler.get_input_params(
            args, experiment_hyperparams)
        logger.info("%s\n%s\n" % (input_dim, variance_map_file))
        experiment_hyperparams["nvis"] = input_dim
        experiment_hyperparams["encoder"]["nvis"] = input_dim

        h = abs(hash(frozenset(
            flatten(experiment_hyperparams).keys() +\
            [tuple(v) if isinstance(v, list)
             else v for v in flatten(experiment_hyperparams).values()])))

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
