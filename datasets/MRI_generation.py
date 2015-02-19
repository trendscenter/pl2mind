"""
Module for a generator MRI class.
"""

import functools
import numpy as np

from pl2mind.datasets import MRI
from pl2mind.logger import logger
from pl2mind.tools import ica

from pylearn2.blocks import Block
from pylearn2.datasets import Dataset
from pylearn2.utils import contains_nan
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

def test_fn():
    node_set = T.dmatrix("node_set")
    edge_set = T.dmatrix("edge_set")
    batch_size = 100;
    n = T.iscalar("n")

    def get_columns(n, node_set, edge_set, target_size):

        [columns, br], updates = theano.scan(get_column,
                                      outputs_info=[None, None],
                                      sequences=[node_set, edge_set],
                                      non_sequences=[n, target_size])
        return columns, theano.scan_module.until(T.eq(br.sum(), 0))

    As, updates = theano.scan(get_columns,
                                 outputs_info=[None],
                                 non_sequences=[node_set, edge_set, batch_size])

    new_A = As[-1]

    return theano.function([node_set, edge_set, n], new_A)

def test_columns():
    edge_set = T.dmatrix("edge_set")
    node_set = T.dmatrix("node_set")
    n = T.iscalar("n")
    target_size = T.iscalar("target_size")

    values, updates = theano.scan(get_column,
                                  outputs_info=[None, None],
                                  sequences=[node_set, edge_set],
                                  non_sequences=[n, target_size])

    return theano.function([node_set, edge_set, n, target_size], values)

def test_column():
    edges = T.dvector("edges")
    nodes = T.dvector("nodes")
    n = 6
    r = make_theano_rng(2015+02+14, which_method="uniform")

    scaled_nodes = nodes * theano.tensor.extra_ops.diff(edges)
    es = r.uniform(size=(n,), low=edges[0], high=edges[-1])
    us = r.uniform(size=(n,))
    output, updates = theano.scan(edge_compare,
                                  outputs_info=None,
                                  sequences=[es],
                                  non_sequences=[edges])

    tests, updates = theano.scan(node_compare,
                                 outputs_info=None,
                                 sequences=[us, scaled_nodes[output]])
    indices = tests.nonzero()[0]
    rval = es[indices]

    return theano.function([nodes, edges], rval)

class MRI_Generator(Block):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    theano_rng : WRITEME
    seed : WRITEME
    input_space : WRITEME
    """
    # Params here.
    def __init__(self, model, sources, theano_rng = None, seed=None,
                 input_space=None):
        super(MRI_Generator, self).__init__()
        assert theano_rng is None or seed is None
        self.theano_rng = make_theano_rng(theano_rng if theano_rng is not None else seed,
                                     2015+02+14, which_method='uniform')
        self.sources = sharedX(sources)
        self.edge_set = sharedX([e for n, e in model])
        self.node_set = sharedX([n for n, e in model])
        self.set_fn()

    def set_fn(self):
        inputs = T.matrix()

        def edge_compare(e, edges):
            return T.gt(e, edges[1:]).argmin()

        def node_compare(u, node):
            return T.lt(node, u)

        def get_column(nodes, edges, n, target_size):
                r = make_theano_rng(2015+02+14, which_method="uniform")
                es = r.uniform((n,), low=edges[0], high=edges[-1])
                us = r.uniform((n,))

                scaled_nodes = nodes * theano.tensor.extra_ops.diff(edges)

                output, updates = theano.scan(edge_compare,
                                              outputs_info=None,
                                              sequences=[es],
                                              non_sequences=[edges])

                tests, updates = theano.scan(node_compare,
                                             outputs_info=None,
                                             sequences=[us, scaled_nodes[output]])

                indices = tests.nonzero()[0]
                br = T.switch(T.lt(indices.shape[0], target_size), 1, 0)
                rem = T.switch(T.eq(br, 1), target_size - indices.shape[0], 0)
                filled_indices = T.concatenate([indices, T.zeros([rem],
                    dtype="int32")])[:target_size]
                rval = T.switch(T.eq(br, 1),
                                filled_indices,
                                es[filled_indices])

                return [rval, br], theano.scan_module.until(T.eq(br, 1))

        def get_columns(n, node_set, edge_set, target_size):
            [columns, br], updates = theano.scan(get_column,
                                                 outputs_info=[None, None],
                                                 sequences=[node_set, edge_set],
                                                 non_sequences=[n, target_size])
            return columns, theano.scan_module.until(T.eq(br.sum(), 0))

        As, updates = theano.scan(get_columns,
                                  outputs_info=[None],
                                  non_sequences=[inputs.shape[0] * 2,
                                                 self.node_set, self.edge_set,
                                                 inputs.shape[0]],
                                  n_steps=16)

        new_A = As[-1].T

        self.fn = theano.function([inputs], new_A.dot(self.sources))

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


class MRI_Gen(MRI.MRI):
    def __init__(self,
                 which_set,
                 num_components,
                 batch_size,
                 source_file = None,
                 mixing_file = None,
                 shuffle=False,
                 apply_mask=False,
                 center=False,
                 demean=False,
                 variance_normalize=False,
                 unit_normalize=False,
                 dataset_name="smri",
                 dataset_root="${PYLEARN2_NI_PATH}"):
        assert not center
        assert not variance_normalize
        assert not unit_normalize
        assert not demean

        self.__dict__.update(locals())
        del self.self
        logger.info("Setting up generating MRI dataset.")

        data_path, label_path = self.resolve_dataset(which_set, dataset_name)

        logger.info("Loading %s data from %s." % (which_set, dataset_name))
        topo_view = np.load(data_path)

        y = np.atleast_2d(np.load(label_path)).T
        logger.info("Dataset shape is %r" % (topo_view.shape,))

        if apply_mask:
            logger.info("Applying mask")
            mask = self.get_mask(dataset_name)
        else:
            mask = None

        logger.info("Getting unmasked data")
        axes = ('b', 0, 1, 'c')
        r, c, d = tuple(topo_view.shape[axes.index(i)] for i in (0, 1, 'c'))
        raw_view_converter = MRI.MRIViewConverter((r, c, d),
            mask=mask, axes=axes)
        X = raw_view_converter.topo_view_to_design_mat(topo_view)

        logger.info("Generating source and mixing matrix from ICA")
        process = True
        if (source_file is not None) and (mixing_file is not None):
            try:
                self.A = np.load(mixing_file)
                self.S = np.load(source_file)
                nS, nC = self.A.shape
                nV = X.shape[1]
                assert self.S.shape == (nC, nV), ("Source matrix shape "
                                                  "mismatch: %s vs %s"
                                                  % (self.S.shape, (nC, nV)))
                assert X.shape == (nS, nV), ("Mixing matrix shape mismatch: "
                                             "%s vs %s" % (self.A.shape,
                                                           (X.shape[0],
                                                            self.S.shape[1])))
                process = False
            except Exception as e:
                logger.exception(e)

        if process:
            self.A, self.S = ica.ica(X - X.mean(axis=0), num_components)
        self.generator = [MRI_Generator(
            [np.histogram(a, density=True) for a in self.A.T],
            self.S), None]
        self.set_mri_topological_view(topo_view, mask)
        super(MRI_Gen, self).__init__(X=X, y=y)

    def set_mri_topological_view(self, topo_view, mask=None,
                                 axes=('b', 0, 1, 'c')):
        """
        Set the topological view.

        Parameters
        ----------
        topo_view: array-like
            Topological view of a matrix, 4D. Should be MRI 4D data.
        mask: array-like
            Mask for data.
        axes: tuple, optional
            Axis to use to set topological view.

        Returns
        -------
        design_matrix: array-like
            The corresponding design matrix for the topological view.
        """
        assert not contains_nan(topo_view)
        r, c, d = tuple(topo_view.shape[axes.index(i)] for i in (0, 1, 'c'))

        self.view_converter = MRI.MRIViewConverter((r, c, d), mask=mask, axes=axes)
        design_matrix = self.view_converter.topo_view_to_design_mat(topo_view)

        return design_matrix

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        space, source = self.data_specs
        subspaces = space.components
        subsources = source
        mode = resolve_iterator_class("shuffled_sequential")

        rng = self.rng
        assert rng is not None
        subset_iterator = mode(self.y.shape[0],
                               batch_size,
                               num_batches,
                               rng=rng)

        return FiniteDatasetIterator(
            self,
            subset_iterator=subset_iterator,
            data_specs=data_specs,
            return_tuple=return_tuple,
            convert=self.generator)