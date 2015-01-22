"""
Module to train NICE on sMRI.
"""

import argparse
import logging
import numpy as np
from os import path
from pylearn2.config import yaml_parse
from pl2mind.datasets import MRI
from pylearn2.utils import serial
from random import shuffle
import sys
from theano import config


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, nvis,
          transposed, logistic, variance_map_file, dataset_name):
    if transposed:
        data_class = "MRI_Transposed"
    else:
        data_class = "MRI_Standard"
    if logistic:
        prior = "StandardLogistic"
    else:
        prior = "StandardNormal"

    logger.info("Data class is %s and prior is %s" % (data_class, prior))
    yaml = open(yaml_file, "r").read()
    hyperparams = {"nvis": nvis,
                   "half_nvis": nvis // 2,
                   "save_path": save_path,
                   "data_class": data_class,
                   "prior": prior,
                   "variance_map_file": variance_map_file,
                   "dataset_name": dataset_name
                   }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def save_variance_map(dataset, save_path):
    logger.info("Saving variance file")
    variance_map = dataset.X.std(axis=0)
    np.save(save_path, variance_map)

def train_nice(args):
    vn = True
    center = True
    logger.info("Getting dataset info for %s" % args.dataset_name)
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + args.dataset_name)
    if args.transposed:
        logger.info("Data in transpose...")
        mri = MRI.MRI_Transposed(dataset_name=args.dataset_name,
                                 unit_normalize=True,
                                 even_input=True,
                                 apply_mask=True)
        input_dim = mri.X.shape[1]
        variance_map_file = path.join(data_path, "transposed_variance_map.npy")
    else:
        mask_file = path.join(data_path, "mask.npy")
        mask = np.load(mask_file)
        input_dim = (mask == 1).sum()
        if input_dim % 2 == 1:
            input_dim -= 1
        mri = MRI.MRI_Standard(which_set="full",
                               dataset_name=args.dataset_name,
                               unit_normalize=True,
                               even_input=True,
                               apply_mask=True)
        variance_map_file = path.join(data_path, "variance_map.npy")
    save_variance_map(mri, variance_map_file)

    logger.info("Input shape: %d" % input_dim)

    p = path.abspath(path.dirname(__file__))
    yaml_file = path.join(p, "nice_mri.yaml")
    user = path.expandvars("$USER")

    if args.out_name is not None:
        out_name = args.out_name
    else:
        out_name = args.dataset_name
    save_file = "nice_%s%s%s" % (out_name,
                                 "_transposed" if args.transposed else "",
                                 "_logistic" if args.logistic else "")
    save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/%s"
                                  % (user, save_file))
    if path.isfile(save_path + ".pkl") or path.isfile(save_path + "_best.pkl"):
        answer = None
        while answer not in ["Y", "N", "y", "n"]:
            answer = raw_input("%s already exists, continuing will overwrite."
                               "\nOverwrite? (Y/N)[N]: " % save_path) or "N"
            if answer not in ["Y", "N", "y", "n"]:
                print "Please answer Y or N"
        if answer in ["N", "n"]:
            print "If you want to run without overwrite, consider using the -o option."
            sys.exit()

    logger.info("Saving to prefix %s" % save_path)

    if not path.isfile(variance_map_file):
        raise ValueError("Variance map file %s not found."
                         % variance_map_file)
    train(yaml_file, save_path, input_dim,
          args.transposed, args.logistic, variance_map_file, args.dataset_name)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--transposed", action="store_true")
    parser.add_argument("-l", "--logistic", action="store_true")
    parser.add_argument("-o", "--out_name", default=None)
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    train_nice(args)
