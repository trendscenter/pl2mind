import numpy as np
from os import path

from pylearn2.config import yaml_parse
from pylearn2.utils import serial

def train_yaml(yaml_file):
    train = yaml_parse.load(yaml_file)
    train.main_loop()

def train(yaml_file, save_path):
    yaml = open(yaml_file, "r").read()

    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri")
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = len(np.where(mask.flatten() == 1)[0].tolist())
    del mask

    hyperparams = {"nvis": input_dim,
                   "batch_size": 5,
                   "detector_layer_dim": 64,
                   "monitoring_batches": 5,
                   "save_path": save_path,
                   "max_epochs": 300
                  }
    yaml = yaml % hyperparams
    train_yaml(yaml)

def train_rbm():
    yaml_file = path.join(path.abspath(path.dirname(__file__)), "rbm.yaml")
    save_path = path.abspath(path.dirname(__file__))
    train(yaml_file, save_path)

if __name__ == "__main__":
    train_rbm()
