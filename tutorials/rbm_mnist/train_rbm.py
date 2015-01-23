"""
Module to train RBM on MNIST.
"""

import os
from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

def train_yaml(yaml_file):
    # Makes a Pylearn2 train object
    train = yaml_parse.load(yaml_file)

    # Trains
    train.main_loop()

def train(yaml_file, save_path, epochs):
    yaml = open(yaml_file, "r").read()
    input_dim = 784 # MNIST input size

    # Fills in the blanks of the yaml file
    hyperparams = {"nvis": input_dim,
                    "batch_size": 50,
                    "detector_layer_dim": 200,
                    "monitoring_batches": 10,
                    "train_stop": 50000,
                    "max_epochs": epochs,
                    "save_path": save_path
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm(epochs = 300, save_path=None):
    # Load the yaml file
    yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")
    if save_path is None:
        save_path = path.abspath(path.dirname(__file__))
    train(yaml_file, save_path, epochs)

if __name__ == "__main__":
    save_path = path.join(serial.preprocess("${PYLEARN2_OUTS}"), "tutorials")
    if not path.isdir(serial.preprocess("${PYLEARN2_OUTS}")):
        raise IOError("PYLEARN2_OUTS environment variable not set")
    if not path.isdir(save_path):
        os.mkdir(save_path)
    train_rbm(save_path)
