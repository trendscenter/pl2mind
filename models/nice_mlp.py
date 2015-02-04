"""
Module for classes to simplify MLPs for NICE training.
"""

import pylearn2
import pylearn2.models
import nice
import nice.pylearn2.models.mlp
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import RectifiedLinear
from nice.pylearn2.models.mlp import CouplingLayer
from nice.pylearn2.models.mlp import Homothety
from nice.pylearn2.models.mlp import TriangularMLP


class Simple_MLP(MLP):
    def __init__(self, layer_name, depth, half_vis, nhid, irange=0.01):
        layers = []
        for i, d in enumerate(xrange(depth)):
            layer = RectifiedLinear(dim=nhid,
                                    layer_name="%s_h%d" % (layer_name, i),
                                    irange=irange)
            layers.append(layer)
        layer = Linear(dim=half_vis,
                       layer_name="%s_out" % layer_name,
                       irange=irange)
        layers.append(layer)

        super(Simple_MLP, self).__init__(layers, layer_name=layer_name)


class Simple_TriangularMLP(TriangularMLP):
    def __init__(self, layer_name, layer_depths, nvis, nhid):
        layers = []
        for i, depth in enumerate(layer_depths):
            layer = CouplingLayer(split=nvis // 2,
                                  coupling=Simple_MLP("coupling_%d" % (i + 1),
                                                      depth,
                                                      nvis // 2,
                                                      nhid))
            layers.append(layer)

        layer = Homothety(layer_name="z")
        layers.append(layer)

        super(Simple_TriangularMLP, self).__init__(layers, layer_name=layer_name)
