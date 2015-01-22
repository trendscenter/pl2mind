"""
Module to handle multimodal multichromosome models
"""

import numpy as np

from pylearn2.models import mlp
from pylearn2.utils import sharedX

from theano.compat.python2x import OrderedDict


class MultimodalMLP(mlp.MLP):
    def __init__(self, dataset, layers):
        super(MultimodalMLP, self).__init__(layers,
                                            input_space=dataset.X_space,
                                            input_source=dataset.X_source)

    @functools.wraps(mlp.Layer.get_monitoring_channels)
    def get_monitoring_channels(self, data):
        rval = super(MultimodalMLP, self).get_monitoring_channels(data)
        rval = OrderedDict((r,rval[r]) for r in rval if "misclass" in r)
        return rval

class MultiChromosomeLayer(mlp.CompositeLayer):
    def __init__(self, num_layers, layer_to_copy, layer_dims=None):
        layer_name = 'multi_chromosome_layer'

        layers = []
        inputs_to_layers = {}
        if layer_dims:
            assert len(layer_dims) == num_layers
        for n in range(num_layers):
            layer = deepcopy(layer_to_copy)
            layer.layer_name += "_%d" % n
            if layer_dims:
                layer.dim = layer_dims[n]
                logger.warning("Init bias in copying ignored.")
                layer.b = sharedX(np.zeros((layer.dim,)), 
                             name=(layer.layer_name + '_b'))
            layers.append(layer)
            inputs_to_layers[n] = [n]

        super(MultiChromosomeLayer, self).__init__(layer_name=layer_name,
                                                   layers=layers,
                                                   inputs_to_layers=inputs_to_layers)

    @functools.wraps(mlp.Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                      state=None, targets=None):
        rval = OrderedDict()
        return rval
