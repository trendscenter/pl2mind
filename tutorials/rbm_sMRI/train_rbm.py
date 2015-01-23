"""
Module for training a simple RBM on sMRI data.
"""

import logging
import numpy as np
import os
from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, epochs):
    yaml = open(yaml_file, "r").read()

    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri")
    logger.info("Loading data from %s" % data_path)
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = len(np.where(mask.flatten() == 1)[0].tolist())
    del mask

    hyperparams = {"nvis": input_dim,
                   "batch_size": 5,
                   "detector_layer_dim": 64,
                   "monitoring_batches": 5,
                   "save_path": save_path,
                   "max_epochs": epochs
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm(epochs=300, save_path = None):
    yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")
    if save_path is None:
        save_path = path.abspath(path.dirname(__file__))
    train(yaml_file, save_path, epochs)

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    save_path = path.join(serial.preprocess("${PYLEARN2_OUTS}"), "tutorials")
    if not path.isdir(serial.preprocess("${PYLEARN2_OUTS}")):
        raise IOError("PYLEARN2_OUTS environment variable not set")
    if not path.isdir(save_path):
        os.mkdir(save_path)
    train_rbm(save_path)
