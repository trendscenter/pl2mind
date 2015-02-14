"""
Module for nice experiment using sigma as our param
"""

from pl2mind.experiments.nice_exp.nice_experiment import *


yaml_file = path.join(path.abspath(path.dirname(__file__)), "nice_mri_pretrained.yaml")

default_hyperparams["transformer"] =\
    "!pkl: \"/export/mialab/users/dhjelm/Experiments/rbm_for_nice.pkl\""
default_hyperparams["encoder"]["nhid"] = 50
default_hyperparams["weight_decay"] = ""
default_hyperparams["nvis"] = 100
default_hyperparams["encoder"]["nvis"] = 100
default_hyperparams["encoder"]["layer_depths"] = [3, 5, 3]