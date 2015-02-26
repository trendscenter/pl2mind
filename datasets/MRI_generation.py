"""
Module for a generator MRI class.
"""

import functools
import logging
import numpy as np

from pl2mind.datasets import MRI
from pl2mind.tools import ica

from pylearn2.blocks import Block
from pylearn2.datasets import Dataset
from pylearn2.utils import contains_nan
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.rng import make_theano_rng
from pylearn2.utils import sharedX

import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams


logger = logging.getLogger("pl2mind")


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
    MRI generation from ICA.

    Parameters
    ----------
    theano_rng : WRITEME
    seed : WRITEME
    input_space : WRITEME
    """
    # Params here.
    def __init__(self, A, sources, y, X_mean, use_real=False, theano_rng=None,
                 seed=None, input_space=None):
        assert theano_rng is None or seed is None
        self.__dict__.update(locals())
        del self.self

        super(MRI_Generator, self).__init__()

        if self.theano_rng is None:
            self.theano_rng = make_theano_rng(None,
                                              2015+02+14,
                                              which_method='uniform')

        # Figure out indices of labels
        idx0 = [i for i, j in enumerate(y) if j == 0]
        idx1 = [i for i, j in enumerate(y) if j == 1]

        # Contruct 2 models
        model0 = [np.histogram(a, density=True) for a in A[idx0].T]
        model1 = [np.histogram(a, density=True) for a in A[idx1].T]

        self.sources = sharedX(sources)
        self.edge_set0 = sharedX([e for h, e in model0])
        self.hist_set0 = sharedX([h for h, e in model0])
        self.edge_set1 = sharedX([e for h, e in model1])
        self.hist_set1 = sharedX([h for h, e in model1])
        self.X_mean = sharedX(X_mean)

        if theano_rng is None:
            self.rng = RandomStreams(2015+02+14)
            #self.rng = make_theano_rng(2015+02+14, which_method="uniform")
        else:
            self.rng = theano_rng
        self.set_fn()

    def edge_compare(self, e, edges):
        return T.gt(e, edges[1:]).argmin()

    def hist_compare(self, u, hist):
        return T.le(u, hist)

    def get_column(self, hist, edges, n, target_size):
        #if rng is None:
        #    rng = self.rng
        es = self.rng.uniform((n,), low=edges[0], high=edges[-1])
        us = self.rng.uniform((n,))

        scaled_hist = hist * theano.tensor.extra_ops.diff(edges)

        output, updates = theano.scan(self.edge_compare,
                                      outputs_info=None,
                                      sequences=[es],
                                      non_sequences=[edges])

        tests, updates = theano.scan(self.hist_compare,
                                     outputs_info=None,
                                     sequences=[us, scaled_hist[output]])

        indices = tests.nonzero()[0]

        br = T.switch(T.lt(indices.shape[0], target_size), 1, 0)
        rem = T.switch(T.eq(br, 1), target_size - indices.shape[0], 0)
        filled_indices = T.concatenate([indices,
                                        T.zeros([rem],
                                            dtype="int32")])[:target_size]
        rval = T.switch(T.eq(br, 1),
                        filled_indices,
                        es[filled_indices])

        return [rval, br, es, us, filled_indices], theano.scan_module.until(T.eq(br, 1))

    def get_columns(self, n, hists, edge_set, target_size):
        [columns, br, es, us, indices], updates = theano.scan(self.get_column,
                                             outputs_info=[None, None, None, None, None],
                                             sequences=[hists, edge_set],
                                             non_sequences=[n, target_size])
        return columns.T, theano.scan_module.until(T.eq(br.sum(), 0))

    def set_fn(self):
        inputs = T.matrix()
        batch_size0 = inputs.shape[0] // 2
        batch_size1 = (inputs.shape[0] + 1) // 2

        A0, updates = theano.scan(self.get_columns,
                                  outputs_info=[None],
                                  non_sequences=[batch_size0 * 100,
                                                 self.hist_set0, self.edge_set0,
                                                 batch_size0],
                                  n_steps=16)

        A1, updates = theano.scan(self.get_columns,
                                  outputs_info=[None],
                                  non_sequences=[batch_size1 * 100,
                                                 self.hist_set1, self.edge_set1,
                                                 batch_size1],
                                  n_steps=16)

        new_A0 = A0[-1]
        new_A1 = A1[-1]
        output = T.concatenate([new_A0.dot(self.sources),
                                new_A1.dot(self.sources)])

        if self.use_real:
            switch = sharedX(1)
            switch = T.switch(T.eq(switch, 0), 1, 0)
            output = T.switch(T.eq(switch, 0), output, inputs + self.X_mean)
            #output = T.concatenate([output, inputs])
        else:
            output = output + self.X_mean

        self.fn = theano.function([inputs], output)

    def __call__(self, X):
        return self.perform(X)

    def set_input_space(self, space):
        self.input_space = space

    def get_input_space(self):
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        return self.get_input_space()


class MRI_LabelGenerator(Block):
    """
    Alternating label block generator.
    Only two classes supported currently.
    """
    def __init__(self, use_real=False, theano_rng=None, seed=None,
                 input_space=None):
        self.__dict__.update(locals())
        del self.self
        super(MRI_LabelGenerator, self).__init__()
        self.set_fn()

    def set_fn(self):
        inputs = T.matrix()
        batch_size0 = inputs.shape[0] // 2
        batch_size1 = (inputs.shape[0] + 1) // 2

        zeros = T.concatenate([T.fill(inputs[:batch_size0], 1),
                               T.fill(inputs[:batch_size1], 0)])
        ones = T.concatenate([T.fill(inputs[:batch_size0], 0),
                               T.fill(inputs[:batch_size1], 1)])
        outputs = T.concatenate([zeros, ones], axis=1)

        if self.use_real:
            switch = sharedX(1)
            switch = T.switch(T.eq(switch, 0), 1, 0)
            convert_inputs = T.concatenate([T.switch(T.eq(inputs, 1), 0, 1),
                                            inputs], axis=1)
            outputs = T.switch(T.eq(switch, 0), outputs, convert_inputs)
            #outputs = T.concatenate([outputs, convert_inputs])

        self.fn = theano.function([inputs], outputs)

    def __call__(self, y):
        return self.perform(y)

    def set_input_space(self, space):
        self.input_space = space

    def get_input_space(self):
        if self.input_space is not None:
            return self.input_space
        raise ValueError("No input space was specified for this Block (%s). "
                "You can call set_input_space to correct that." % str(self))

    def get_output_space(self):
        return self.get_input_space()

class MRI_Gen(MRI.MRI):
    _default_seed = (18, 4, 500)
    def __init__(self,
                 which_set,
                 num_components,
                 use_real=False,
                 source_file=None,
                 mixing_file=None,
                 shuffle=False,
                 apply_mask=False,
                 center=False,
                 demean=False,
                 variance_normalize=False,
                 unit_normalize=False,
                 dataset_name="smri",
                 rng=_default_seed,
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

        num_classes = np.amax(y) + 1
        class_counts = [(y == i).sum()
                        for i in range(num_classes)]
        min_count = min(class_counts)
        balanced_idx = []
        for i in range(num_classes):
            idx = [idx for idx, j in enumerate(y) if j == i][:min_count]
            balanced_idx += idx
        balanced_idx.sort()
        assert len(balanced_idx) / min_count == num_classes
        assert len(balanced_idx) % min_count == 0

        y = y[balanced_idx]
        for i in range(num_classes):
            assert (y == i).sum() == min_count
        X = X[balanced_idx]

        process = True
        if (source_file is not None) and (mixing_file is not None):
            try:
                logger.info("Loading source and mixing matrix for ICA from "
                            "file")
                self.A = np.load(mixing_file)
                self.S = np.load(source_file)
                nS, nC = self.A.shape
                assert nC == num_components, "Components do not match"
                nV = X.shape[1]
                assert self.S.shape == (nC, nV), ("Source matrix shape "
                                                  "mismatch: %s vs %s"
                                                  % (self.S.shape, (nC, nV)))
                assert X.shape == (nS, nV), ("Mixing matrix shape mismatch: "
                                             "%s vs %s" % (self.A.shape,
                                                           (X.shape[0],
                                                            self.S.shape[0])))
                process = False
            except Exception as e:
                logger.exception(e)

        if process:
            logger.info("Generating source and mixing matrix from ICA")
            self.A, self.S = ica.ica(X - X.mean(axis=0), num_components)
            if (source_file is not None) and (mixing_file is not None):
                np.save(source_file, self.S)
                np.save(mixing_file, self.A)

        self.rng = make_np_rng(rng, which_method="random_integers")
        assert self.rng is not None

        self._iter_mode = resolve_iterator_class("sequential")
        self._iter_topo = False
        self._iter_targets = False
        #self._iter_data_specs = self.data_specs

        self.generator = [MRI_Generator(self.A, self.S, y, X.mean(axis=0),
                                        use_real=use_real),
                          MRI_LabelGenerator(use_real=use_real)]

        #MRI_LabelGenerator()
        #y = np.atleast_2d((([[0]] * (batch_size // 2))
        #    + ([[1]] * (batch_size // 2))) * 10)
        self.set_mri_topological_view(topo_view, mask)

        if use_real:
            X = np.repeat(X, 2, axis=0)
            y = np.repeat(y, 2, axis=0)

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
        mode = resolve_iterator_class("sequential")

        #if self.use_real:
        #    samples = self.y.shape[0] * 2
        #else:
        samples = self.y.shape[0]

        subset_iterator = mode(samples,
                               batch_size,
                               num_batches,
                               rng=rng)

        return FiniteDatasetIterator(self,
                                     subset_iterator=subset_iterator,
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=self.generator)