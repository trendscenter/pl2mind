"""
Module to train VAE on sMRI.
"""

import numpy as np
from os import path
from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets import MRI
from pylearn2.utils import serial
from random import shuffle
from theano import config

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path, nvis):
    yaml = open(yaml_file, "r").read()
    hyperparams = {"nvis": nvis,
                   "half_nvis": nvis // 2,
                   "save_path": save_path,
                   }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_nice():
    vn = True
    center = True
    smri = MRI.MRI_Transposed(dataset_name="smri",
                              even_input=True)
    input_dim = smri.X.shape[1]

    p = path.abspath(path.dirname(__file__))
    yaml_file = path.join(p, "nice_smri_transposed.yaml")
    save_path = "/export/mialab/users/$USER/pylearn2_outs/"
    train(yaml_file, save_path, input_dim)

if __name__ == "__main__":
    train_nice()
