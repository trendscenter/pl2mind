"""
Module demonstrating a rbm jobman experiment
"""

from pl2mind.experiments.rbm_exp.rbm_experiment import *
from pl2mind.tools import jobman_generators as jg


# Generators are defined here. See pl2mind/tools/jobman_generators.py for
# details.
# Note all generators need to be within a nested_generator.
generator = jg.nested_generator(
    jg.list_generator(
        "weight_decay.coeffs", [[0.1], [0.01], [0.001], [0.0001]]))

# Define some other hyperparameters
default_hyperparams["nhid"] = 50