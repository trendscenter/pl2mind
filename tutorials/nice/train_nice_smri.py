"""
Module to train NICE on sMRI.
"""

import logging
import numpy as np
from os import path
from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets import MRI
from pylearn2.utils import serial
from theano import config

logging.basicConfig(format="[%(levelname)s]:%(message)s")

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, nvis, vn, center):
    yaml = open(yaml_file, "r").read()
    hyperparams = {"nvis": nvis,
                   "half_nvis": nvis // 2,
                   "center": center,
                   "vn": vn,
                   "save_path": save_path
                   }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_nice():
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri")
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = (mask == 1).sum()
    if input_dim % 2 == 1:
        input_dim -= 1
    logging.info("Input shape: %d" % input_dim)
    del mask
    
    vn = False
    center = True

    p = path.abspath(path.dirname(__file__))
    yaml_file = path.join(p, "nice_smri.yaml")
    user = path.expandvars("$USER")
    save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/" % user)
    assert path.isdir(save_path)
    train(yaml_file, save_path, input_dim, vn, center)

if __name__ == "__main__":
    train_nice()
