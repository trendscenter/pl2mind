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
from theano import config


logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, nvis,
          transposed, logistic, variance_map_file):
    if transposed:
        data_class = "MRI_Transposed"
    else:
        data_class = "MRI_Standard"
    if logistic:
        prior = "StandardLogistic"
    else:
        prior = "StandardNormal"

    yaml = open(yaml_file, "r").read()
    hyperparams = {"nvis": nvis,
                   "half_nvis": nvis // 2,
                   "save_path": save_path,
                   "data_class": data_class,
                   "prior": prior,
                   "variance_map_file": variance_map_file
                   }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_nice(args):
    vn = True
    center = True
    if args.transposed:
        fmri = MRI.MRI_Transposed(dataset_name=args.dataset_name,
                                  even_input=True)
        input_dim = fmri.X.shape[1]
        del fmri
    else:
        data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + args.dataset_name)
        mask_file = path.join(data_path, "mask.npy")
        mask = np.load(mask_file)
        input_dim = (mask == 1).sum()
        if input_dim % 2 == 1:
            input_dim -= 1

    logging.info("Input shape: %d" % input_dim)

    p = path.abspath(path.dirname(__file__))
    yaml_file = path.join(p, "nice_%s.yaml" % args.dataset_name)
    user = path.expandvars("$USER")
    save_file = "nice_%s%s%s" % (args.dataset_name,
                                 "_transposed" if args.transposed else "",
                                 "_logistic" if args.logistic else "")
    save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/%s"
                                  % (user, save_file))
    variance_map_file = path.join(data_path, "variance_map.npy")
    if not path.isfile(variance_map_file):
        raise ValueError("Variance map file %s not found."
                         % variance_map_file)
    train(yaml_file, save_path, input_dim,
          args.transposed, args.logistic, variance_map_file)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-t", "--transposed", action="store_true")
    parser.add_argument("-l", "--logistic", action="store_true")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    train_nice(args)
