"""
Module for nice jobman experiment
"""

from pl2mind.experiments.nice_exp.nice_experiment import *
from pl2mind.tools import jobman_generators as jg


# Generators are defined here. See pl2mind/tools/jobman_generators.py for
# details.
# Note all generators need to be within a nested_generator.
generator = jg.nested_generator(
    jg.list_generator(
        "encoder.nhid", [400, 700, 1000]),
    jg.float_generator("weight_decay.coeff", 3, 0.01, 0.0001)
    )

default_hyperparams.update(
    layer_depths = [3, 5, 5, 5, 3],
    demean = True,
    variance_normalize = True)