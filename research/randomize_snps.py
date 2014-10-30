"""
.. todo::

    WRITEME
"""
<<<<<<< HEAD
__authors__ = "Devon Hjelm"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Devon Hjelm"]
=======
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
>>>>>>> df1c7653784efe5fd3d91d52cca2793e6a0b2de1
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.blocks import Block
from pylearn2.utils.rng import make_theano_rng


class RandomizeSNPs(Block):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    theano_rng : WRITEME
    seed : WRITEME
    input_space : WRITEME
    """
    def __init__(self, theano_rng = None, seed=None,
                 input_space=None, corruption_prob=0.05):
        super(RandomizeSNPs, self).__init__()
        assert theano_rng is None or seed is None
        theano_rng = make_theano_rng(theano_rng if theano_rng is not None else seed,
                                     2012+11+22, which_method='binomial')
        self.__dict__.update(locals())
        del self.self
        self.set_fn()

    def set_fn(self):
        """
        .. todo::

            WRITEME
        """
        inputs = T.matrix()

        a = self.theano_rng.binomial(
            size=(self.input_space.dim, ),
            p=(1 - self.corruption_prob),
            dtype=theano.config.floatX
        )

        b = self.theano_rng.binomial(
            size=(self.input_space.dim, ),
            p=0.5,
            dtype=theano.config.floatX
            ) + 1

        c = T.eq(a, 0) * b
        self.fn = theano.function([inputs], ((2 * inputs + c) % 3 / 2.0))
    
    def __call__(self, X):
        return self.perform(X)

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME
        """
        self.input_space = space

    def get_input_space(self):
        """
        .. todo::

            WRITEME
        """
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        """
        .. todo::

            WRITEME
        """
        return self.get_input_space()
