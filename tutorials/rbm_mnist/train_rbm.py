"""
Module to train RBM on MNIST.
"""

from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path):
    yaml = open(yaml_file, "r").read()
    input_dim = 784 # MNIST input size
    hyperparams = {"nvis": input_dim,
                    "batch_size": 50,
                    "detector_layer_dim": 200,
                    "monitoring_batches": 10,
                    "train_stop": 50000,
                    "max_epochs": 300,
                    "save_path": save_path
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm():
    yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")
    save_path = path.abspath(path.dirname(__file__))
    train(yaml_file, save_path)

if __name__ == "__main__":
    train_rbm()
