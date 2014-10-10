"""
Module for training snps from a yaml file.
"""

import numpy as np
from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path):
    # For now just deal with one chromosome
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/snp")
    data_file = path.join(data_path, "gen.chr1.npy")
    label_file = path.join(data_path, "gen.chr1_labels.npy")

    data = np.load(data_file)
    samples = data.shape[0]
    dim = data.shape[1]
    del data

    yaml = open(yaml_file , "r").read()
    hyperparams = {'nvis': dim,
                   'dim_h0': 20,
                   'dim_h1': 100,
                   'n_classes': 2,
                   'batch_size': 10,
                   'train_stop': (3 * samples) // 4,
                   'valid_stop': samples,
                   'max_epochs': 300,
                   'save_path': save_path,
                   }
    yaml = yaml % hyperparams

    train_yaml(yaml)
    
def train_mlp():
    # Assumes yaml is in same directory as module for now.
    yaml_file = path.join(path.abspath(path.dirname(__file__)), "snp_mlp.yaml")
    save_path = path.abspath(path.dirname(__file__))
    train(yaml_file, save_path)

if __name__ == "__main__":
    train_mlp()
