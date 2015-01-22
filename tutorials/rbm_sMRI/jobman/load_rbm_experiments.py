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
from pl2mind.tutorials.rbm_sMRI.jobman import rbm_experiment as experiment
from pylearn2.utils import serial


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

class InputHandler(object):
    """
    Generalized sMRI input handler.
    TODO: move out of loader.
    """

    def __init__(self):
        self.d = {}

    def get_input_params(self, args, hyperparams):
        data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + args.dataset_name)

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
                logger.info((data_class, variance_normalize, unit_normalize, demean))

        variance_map_file = path.join(data_path, variance_file_name)
        if not path.isfile(variance_map_file):
            logger.info("Saving variance file %s" % variance_map_file)
            mri_nifti.save_variance_map(mri, variance_map_file)
        self.d[h] = (input_dim, variance_map_file)
        return self.d[h]

def load_experiments(args):
    dataset_name = args.dataset_name

    # Load the database and table.
    db = sql.db("postgres://%(user)s@%(host)s:%(port)d/%(database)s?table=%(table)s"
                % {"user": args.user,
                   "host": args.host,
                   "port": args.port,
                   "database": args.database,
                   "table": args.table,
                   })

    # Don't worry about this yet.
    input_handler = InputHandler()

    # For generating models, we use a special set of jobman generators, made
    # for convenience.
    for items in jg.nested_generator(
        jg.float_generator("learning_rate", 3, 0.01, 0.0001, log_scale=True),
        jg.list_generator("nhid", [50, 100, 200, 300]),
        ):

        logger.info("Adding RBM experiment across hyperparameters %s" % (items, ))
        state = DD()

        # Load experiment hyperparams from experiment
        experiment_hyperparams = experiment.default_hyperparams()

        # Set them with values in our loop.
        for key, value in items:
            split_keys = key.split(".")
            entry = experiment_hyperparams
            for k in split_keys[:-1]:
                entry = entry[k]
            assert split_keys[-1] in entry, ("Key not found in hyperparams: %s"
                                             % split_keys[-1])
            entry[split_keys[-1]] = value

        # Set the dataset name
        experiment_hyperparams["dataset_name"] = dataset_name

        # Get the input dim and variance map. Don't worry about variance maps right now,
        # they aren't used here.
        input_dim, variance_map_file = input_handler.get_input_params(args,
                                                                      experiment_hyperparams)
        logger.info("%s\n%s\n" % (input_dim, variance_map_file))

        # Set the input dimensionality by the data
        experiment_hyperparams["nvis"] = input_dim

        # Set the minimum learning rate relative to the initial learning rate.
        experiment_hyperparams["min_lr"] = experiment_hyperparams["learning_rate"] / 10

        # Make a unique hash for experiments. Remember that lists, dicts, and other data
        # types may not be hashable, so you may need to do some special processing. In
        # this case we convert the lists to tuples.
        h = abs(hash(frozenset(flatten(experiment_hyperparams).keys() +\
                                   [tuple(v) if isinstance(v, list)
                                    else v 
                                    for v in flatten(experiment_hyperparams).values()])))

        # Save path for the experiments. In this case we are sharing a directory in my 
        # export directory so IT can blame me.
        save_path = serial.preprocess("/export/mialab/users/dhjelm/pylearn2_outs/rbm_demo/%d"
                                      % h)

        # We save file params separately as they aren't model specific.
        file_params = {
            "save_path": save_path,
            "variance_map_file": variance_map_file,
            }

        state.file_parameters = file_params
        state.hyper_parameters = experiment_hyperparams
        user = path.expandvars("$USER")
        state.created_by = user

        # Finally we add the experiment to the table.
        sql.insert_job(
            experiment.experiment,
            flatten(state),
            db
            )

    # A view can be used when querying the database using psql. May not be needed in future.
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
