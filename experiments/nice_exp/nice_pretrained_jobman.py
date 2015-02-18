"""
Module for training many nice experiments pretrained from RBM.
"""

from pl2mind.experiments.nice_exp.nice_pretrained_experiment import *
from pl2mind.tools import jobman_generators as jg


generator = jg.nested_generator(
    jg.list_generator("encoder.layer_depths", [
        [2, 4, 2],
        [3, 5, 3],
        [2, 4, 4, 2],
        [3, 5, 5, 3],
        [2, 4, 4, 4, 2],
        [3, 5, 5, 5, 3]])
    )

default_hyperparams["weight_decay"] = {
    "__builder__": "nice.pylearn2.costs.log_likelihood.SigmaPenalty",
    "coeff": 0.01
}