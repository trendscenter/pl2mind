"""
This module is for handling MRI datasets (sMRI and fMRI).
TODO: handle functional aspects for fMRI here.
"""

import functools
from nipy import load_image
from nipy.core.api import Image
import numpy as np
from os import path

from pylearn2 import corruption
from pylearn2.datasets import control
from pylearn2.datasets import Dataset
from pylearn2.datasets import dense_design_matrix

from pl2mind.datasets import dataset_info
from pl2mind.logger import logger

from pylearn2.space import CompositeSpace
from pylearn2.utils import contains_nan
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import sharedX

import sys
import theano
from theano import config
import warnings


class GaussianMRICorruptor(corruption.Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero mean non-isotropic
    noise.
    Parameters
    ----------
    stdev : WRITEME
    rng : WRITEME
    """

    def __init__(self, stdev, variance_map, rng=2001):
        self.variance_map = sharedX(variance_map)
        super(GaussianMRICorruptor, self).__init__(corruption_level=stdev,
                                                rng=rng)

    def _corrupt(self, x):
        """
        Corrupts a single tensor_like object.
        Parameters
        ----------
        x : tensor_like
            Theano symbolic representing a (mini)batch of inputs to be
            corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.
        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted input.
        """
        noise = self.s_rng.normal(
            size=x.shape,
            avg=0.,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        return noise * self.variance_map + x

    def corruption_free_energy(self, corrupted_X, X):
        """
        .. todo::
            WRITEME
        """
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval


class MRI(dense_design_matrix.DenseDesignMatrix):
    """
    Base class for MRI datasets.
    This includes fMRI and sMRI or any other datasets with 3D voxels.
    """

    def __init__(self, X, y):
        if self.dataset_name in dataset_info.aod_datasets and self.which_set == "full":
            self.targets, self.novels = self.load_aod_gts()
            assert self.targets.shape == self.novels.shape
            if X.shape[0] % self.targets.shape[0] != 0:
                raise ValueError("AOD data and targets seems to have "
                                 "incompatible shapes: %r vs %r"
                                 % (X.shape, self.targets.shape))

        if self.center:
            X -= X.mean()

        if self.demean:
            if isinstance(self.demean, tuple):
                self.demean = self.demean[0]
            assert isinstance(self.demean, int), self.demean
            if self.demean == 1:
                X -= X.mean(axis = 0)
            elif self.demean == 2:
                X = (X.T - X.mean(1)).T
            else:
                raise NotImplementedError

        if self.variance_normalize:
            if isinstance(self.variance_normalize, tuple):
                self.variance_normalize = self.variance_normalize[0]
            assert isinstance(self.variance_normalize, int)
            if self.variance_normalize == 1:
                X /= X.std(axis = 0)
            elif self.variance_normalize == 2:
                X = (X.T / X.std(axis = 1)).T
            else:
                raise NotImplementedError

        if self.unit_normalize:
            X -= X.min()
            X /= X.max()
            X = (X - .5) * 2
            assert abs(np.amax(X) - 1) < 0.08, np.amax(X)
            assert np.amin(X) == -1, np.amin(X)

        if self.shuffle:
            self.shuffle_rng = make_np_rng(None, [1 ,2 ,3], which_method="shuffle")
            for i in xrange(m):
                j = self.shuffle_rng.randint(m)
                tmp = X[i].copy()
                X[i] = X[j]
                X[j] = tmp
                tmp = y[i:i+1].copy()
                y[i] = y[j]
                y[j] = tmp

        max_labels = np.amax(y) + 1
        logger.info("%d labels found." % max_labels)

        super(MRI, self).__init__(X=X,
                                  y=y,
                                  view_converter=self.view_converter,
                                  y_labels=max_labels)

        assert not np.any(np.isnan(self.X))

    def resolve_dataset(self, which_set, dataset_name):
        """
        Resolve the dataset from the file directories.

        Parameters
        ----------
        which_set: str
            train, test, or full.

        """
        p = path.join(self.dataset_root, dataset_name + "/")

        if not(path.isdir(serial.preprocess(p))):
            raise IOError("MRI dataset directory %s not found."
                           % serial.preprocess(p))

        if which_set == 'train':
            data_path = p + 'train.npy'
            label_path = p + 'train_labels.npy'
        elif which_set == 'test':
            data_path = p + 'test.npy'
            label_path = p + 'test_labels.npy'
        else:
            if which_set != "full":
                raise ValueError("dataset \'%s\' not supported." % which_set)
            data_path = p + "full_unshuffled.npy"
            label_path = p + "full_labels_unshuffled.npy"

        data_path = serial.preprocess(data_path)
        label_path = serial.preprocess(label_path)

        if not(path.isfile(data_path)):
            raise ValueError("Dataset %s not found in %s" %(which_set,
                                                            serial.preprocess(p)))
        return data_path, label_path

    def load_aod_gts(self):
        p = path.join(self.dataset_root, "aod_extra/")

        if not(path.isdir(serial.preprocess(p))):
            raise IOError("AOD extras directory %s not found."
                          % serial.preprocess(p))

        targets = np.load(serial.preprocess(p + "targets.npy"))
        novels = np.load(serial.preprocess(p + "novels.npy"))
        return targets, novels

    def get_mask(self, dataset_name):
        """
        Get mask for dataset.

        Parameters
        ----------
        dataset_name: str
            Name of dataset.

        Returns
        -------
        mask: array-like
            4D array of 1 and 0 values.
        """
        p = path.join(self.dataset_root, dataset_name + "/")
        mask_path = serial.preprocess(p + "mask.npy")
        mask = np.load(mask_path)
        if not np.all(np.bitwise_or(mask == 0, mask == 1)):
            raise ValueError("Mask has incorrect values.")
        return mask

    def set_mri_topological_view(self, topo_view, axes=('b', 0, 1, 'c')):
        """
        Set the topological view.

        Parameters
        ----------
        topo_view: array-like
            Topological view of a matrix, 4D. Should be MRI 4D data.
        axes: tuple, optional
            Axis to use to set topological view.

        Returns
        -------
        design_matrix: array-like
            The corresponding design matrix for the topological view.
        """
        raise NotImplementedError()

    def get_weights_view(self, mat):
        """
        Get the weights view from a matrix.

        Parameters
        ----------
        mat: array-like
            Matrix to convert.

        Returns
        -------
        weights_view: array-like
            Weights view of a matrix (see MRIViewConverter).
        """
        if self.view_converter is None:
            raise NotImplementedError("Tried to call get_weights_view on a dataset "
                            "that has no view converter.")
#        if self.X.shape[1] != mat.shape[1]:
#            raise ValueError("mat samples have different size than data: "
#                             "%d vs %d" % (mat.shape[1], self.X.shape[1]))
        weights_view = self.view_converter.design_mat_to_weights_view(mat)
        return weights_view

    def get_design_matrix(self, topo=None):
        """
        Return topo (a batch of examples in topology preserving format),
        in design matrix format.

        Parameters
        ----------
        topo : ndarray, optional
            An array containing a topological representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

        Returns
        -------
        design_matrix: array-like
            Design matrix, 2D. Either self.X or converted topo_view.
        """
        if topo is not None:
            if self.view_converter is None:
                raise NotImplementedError("Tried to call get_design_matrix"
                                          "on a dataset "
                                          "that has no view converter.")

            design_matrix = self.view_converter.topo_view_to_design_mat(topo)
        else:
            design_matrix = self.X

        return design_matrix

    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

        Returns
        -------
        topo_view: array-like
            4D topological view.
        """
        if self.view_converter is None:
            raise NotImplementedError("Tried to call get_topological_view on a dataset "
                                      "that has no view converter")
        if mat is None:
            mat = self.X
        topo_view = self.view_converter.design_mat_to_topo_view(mat)

        return topo_view

    def get_nifti(self, topo_view):
        """
        Process the nifti

        Parameters
        ----------
        topo_view: array-like
            Topological view to create nifti. 4D.

        Returns
        -------
        image: nipy image
            Nifti image from topological view.
        """
        m, r, c, d = topo_view.shape
        base_nifti_path = serial.preprocess(path.join(self.dataset_root, "mri_extra", "basenifti.nii"))
        base_nifti = load_image(base_nifti_path)

        image = Image.from_image(base_nifti, data=topo_view.transpose((1, 2, 3, 0)))
        return image

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None,
                 return_tuple=False):

        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            if src == 'features' and getattr(self, 'view_converter', None) is not None:
                if self.distorter is None:
                    conv_fn = (lambda batch, self=self, space=sp:
                                   self.view_converter.get_formatted_batch(batch,
                                                                           space))
                else:
                    conv_fn = (lambda batch, self=self, space=sp:
                                   self.distorter._distort(
                            self.view_converter.get_formatted_batch(batch,
                                                                    space)))
            else:
                conv_fn = None

            convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.X.shape[0],
                                          batch_size,
                                          num_batches,
                                          rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)


class MRI_Standard(MRI):
    """
    Class for MRI datasets with standard topological view.
    """

    def __init__(self,
                 which_set,
                 even_input=False,
                 center=False,
                 demean=False,
                 variance_normalize=False,
                 unit_normalize=False,
                 shuffle=False,
                 apply_mask=False,
                 preprocessor=None,
                 distorter=None,
                 dataset_name="smri",
                 start=None,
                 stop=None,
                 dataset_root="${PYLEARN2_NI_PATH}"):

        self.__dict__.update(locals())
        del self.self
        logger.info("Setting up standard MRI dataset.")

        data_path, label_path = self.resolve_dataset(which_set,
                                                     dataset_name)

        logger.info("Loading %s data from %s." % (which_set, dataset_name))
        topo_view = np.load(data_path)

        y = np.atleast_2d(np.load(label_path)).T
        logger.info("Dataset shape is %r" % (topo_view.shape,))

        if apply_mask:
            mask = self.get_mask(dataset_name)
        else:
            mask = None

        if self.even_input:
            assert mask is not None
            if (reduce(lambda x, y: x * y, topo_view[0].shape) - (mask == 0).sum()) % 2 == 1:
                logger.warn("Removing one voxel to mask to even input.")
                ons = np.where(mask == 1)
                i, j, k = (ons[r][0] for r in range(3))
                mask[i, j, k] = 0
                assert (reduce(lambda x, y: x * y, topo_view[0].shape) - (mask == 0).sum()) % 2 == 0

        X = self.set_mri_topological_view(topo_view, mask=mask)
        if mask is not None:
            logger.info("Masked shape is %r" % (X.shape,))
            assert X.shape[1] == (mask == 1).sum()
        if even_input:
            assert X.shape[1] % 2 == 0

        super(MRI_Standard, self).__init__(X=X, y=y)

    def set_mri_topological_view(self, topo_view, mask=None, axes=('b', 0, 1, 'c')):
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

        self.view_converter = MRIViewConverter((r, c, d), mask=mask, axes=axes)
        design_matrix = self.view_converter.topo_view_to_design_mat(topo_view)

        return design_matrix


class MRI_On_Memory(MRI_Standard):
        def __init__(self,
                     which_set,
                     center=False,
                     variance_normalize=False,
                     shuffle=False,
                     apply_mask=False,
                     preprocessor=None,
                     dataset_name="smri",
                     start=None,
                     stop=None,
                     dataset_root="${PYLEARN2_NI_PATH}"):
            warnings.warn("MRI_On_Memory should be replaced by MRI_Standard", DeprecationWarning)
            super(MRI_On_Memory, self).__init__(which_set, center, variance_normalize,
                                                shuffle, apply_mask, preprocessor, dataset_name,
                                                start, stop)


class MRI_Transposed(MRI):
    """
    Class to do MRI in transpose.
    """

    def __init__(self,
                 which_set="full",
                 dataset_name="smri",
                 even_input=False,
                 center=False,
                 demean=False,
                 variance_normalize=False,
                 unit_normalize=False,
                 shuffle=False,
                 apply_mask=False,
                 distorter=None,
                 start=None,
                 stop=None,
                 dataset_root="${PYLEARN2_NI_PATH}"):

        self.__dict__.update(locals())
        del self.self
        logger.info("Setting up transposed MRI dataset.")

        if which_set != "full":
            warnings.warn("Only full dataset appropriate for transpose, setting to full.")
            which_set = "full"

        data_file, label_file = self.resolve_dataset(which_set, dataset_name)

        logger.info("Loading %s data from %s." % (which_set, dataset_name))
        topo_view = np.load(data_file)
        y = np.atleast_2d(np.load(label_file)).T
        logger.info("Dataset shape is %r" % (topo_view.shape,))

        if even_input and topo_view.shape[0] % 2 == 1:
            logger.info("Evening input from %d to %d with mask."
                        % (topo_view.shape[0], topo_view.shape[0] // 2 * 2))
            topo_view = topo_view[:topo_view.shape[0] // 2 * 2]
            y = y[:y.shape[0] // 2 * 2]

        if apply_mask:
            mask = self.get_mask(dataset_name)
        else:
            mask = None

        X = self.set_mri_topological_view(topo_view, mask=mask)

        super(MRI_Transposed, self).__init__(X=X, y=y)

    def set_mri_topological_view(self, topo_view, mask=None, axes=('b', 0, 1, 'c')):
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

        self.view_converter = MRIViewConverterTransposed(
            (r, c, d), mask=mask, axes=axes)
        design_matrix = self.view_converter.topo_view_to_design_mat(topo_view)

        return design_matrix


class MRI_Big(dense_design_matrix.DenseDesignMatrixPyTables):
    """
    This class is for read-only memory MRI.

    Note: eventually this should be an option in one class as the non-pytables version.
    """

    def __init__(self, which_set, center=False, variance_normalize=False,
                 shuffle=False, apply_mask=False, preprocessor=None, dataset_name='smri',
                 reprocess=False, save_dummy=False):
        """
        Parameters
        ----------
        which_set: string
            "train" or "test"
        center: bool
            If True, then data -> data - data.mean()
        variance_normalize: True
            If True, then data -> data / data.std()
        shuffle: bool
            If True, then shuffle data when writing h5 (does nothing if not processing an h5).
        apply_mask: bool:
            If True, then the h5 file is masked with a mask file found in the data directory.
        preprocessor: not supported yet, TODO.
        dataset_name: string
            Dataset sub-directory name from ${PYLEARN_NI_PATH}
        reprocess: bool
            Some might want to reprocess the h5 file.
        save_dummy: bool
            Use a dummy file. This is for tests.
        """

        if not path.isdir(serial.preprocess("${PYLEARN2_NI_PATH}")):
            raise ValueError("Did you set the PYLEARN_NI_PATH variable?")

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "Currently validation dataset is not supported with"
                    "sMRI.  This can be added in smri_nifti.py.")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        self.__dict__.update(locals())
        del self.self

        p = "${PYLEARN2_NI_PATH}/%s/" % dataset_name
        assert path.isdir(p), "No NI data directory called '%s'" %dataset_name

        if which_set == 'train':
            data_path = p + 'train.h5'
        else:
            assert which_set == 'test'
            data_path = p + 'test.h5'

        # Dummy file is for tests, don't want to resave over data we might actually be
        # using every time we run a test.
        if save_dummy:
            data_path = "".join(data_path.split(".")[0] + '_dummy.h5')

        data_path = serial.preprocess(data_path)

        # Load the mask file and retrieve shape information.
        self.mask = None
        mask_path = serial.preprocess(p + "mask.npy")
        if not path.isfile(mask_path):
            raise IOError("No mask found in %s."
                          "This file is needed to retrieve shape information."
                          "Are you sure this is a MRI dataset?" % mask_path)
        mask = np.load(mask_path)
        rows, columns, depth = mask.shape
        if apply_mask:
            self.mask = mask

        # Make the h5 file if not present or if reprocess flag is set.
        if not os.path.isfile(data_path) or reprocess:
            self.filters = tables.Filters(complib='blosc', complevel=5)
            self.make_data(which_set, serial.preprocess(p),
                           center=center,
                           variance_normalize=variance_normalize,
                           shuffle=shuffle, save_dummy=save_dummy)

        self.h5file = tables.openFile(data_path)
        data = self.h5file.getNode('/', "Data")
        view_converter = MRIViewConverter((rows, columns, depth))

        super(MRI_Big, self).__init__(X=data.X, y=data.y,
                                      view_converter=view_converter)

        self.h5file.flush()

    def make_data(self, which_set, p, center=False, variance_normalize=False,
                  shuffle=False, save_dummy=False):
        """
        Function to make h5 file.
        Note: parameters the same as __init__ function.
        """

        print "Making h5 file for %s" % which_set #TODO(dhjelm): switch to logging.

        if which_set == 'train':
            source_path = serial.preprocess(p + 'train.npy')
            data_path = serial.preprocess(p + 'train.h5')
            label_path = serial.preprocess(p + 'train_labels.npy')
        else:
            assert which_set == 'test'
            source_path = serial.preprocess(p + 'test.npy')
            data_path = serial.preprocess(p + 'test.h5')
            label_path = serial.preprocess(p + 'test_labels.npy')

        data_path = "".join(data_path.split(".")[0] + '_dummy.h5')

        # Get the topological view and labels.
        topo_view = np.load(source_path)
        y = np.load(label_path)
        num_labels = np.amax(y) + 1

        # Shape information and mask.
        samples, rows, columns, depth = topo_view.shape
        if self.mask is not None:
            assert self.mask.shape == (rows, columns, depth)
            size = len(np.where(self.mask.flatten() == 1)[0].tolist())
        else:
            size = rows * columns * depth

        self.view_converter = MRIViewConverter((rows, columns, depth))
        X = self.view_converter.topo_view_to_design_mat(topo_view, self.mask)

        # TODO(dhjelm): one_hot is going away.
        one_hot = np.zeros((size, num_labels), dtype=config.floatX)
        for i in xrange(y.shape[0]):
            one_hot[i, y[i] - 1] = 1.

        if center:
            X -= X.mean(axis=0)

        if variance_normalize:
            X /= X.std(axis=0)

        rng = make_np_rng(None, 322, which_method="shuffle")
        if shuffle:
            index = range(samples)
            rng.shuffle(index)
            X = X[index, :]
            one_hot = one_hot[index, :]

        assert not np.any(np.isnan(X))

        h5file, node = self.init_hdf5(data_path, ([samples, size], [samples, num_labels]))
        MRI_Big.fill_hdf5(h5file, X, one_hot, node)
        h5file.close()

    def get_nifti(self, W):
        """
        Function to make a nifti file from weights.

        Parameters
        ----------
        W: array-like
            Weights.
        """

        m, r, c, d = W.shape
        base_nifti_path = serial.preprocess("${PYLEARN2_NI_PATH}/mri_extra/basenifti.nii")
        base_nifti = load_image(base_nifti_path)

        data = np.zeros([r, c, d, m], dtype=W.dtype)

        for i in range(m):
            data[:, :, :, i]  = W[i]

        image = Image.from_image(base_nifti, data=data)
        return image

    def get_weights_view(self, mat):
        """
        ... todo::

            WRITEME
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_weights_view on a dataset "
                            "that has no view converter")

        return self.view_converter.design_mat_to_weights_view(mat, self.mask)

    def get_design_matrix(self, topo=None):
        """
        Return topo (a batch of examples in topology preserving format),
        in design matrix format.

        Parameters
        ----------
        topo : ndarray, optional
            An array containing a topological representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.

        Returns
        -------
        WRITEME
        """
        if topo is not None:
            if self.view_converter is None:
                raise Exception("Tried to convert from topological_view to "
                                "design matrix using a dataset that has no "
                                "view converter")
            return self.view_converter.topo_view_to_design_mat(topo, mask=self.mask)

        return self.X

    def get_topological_view(self, mat=None):
        """
        Convert an array (or the entire dataset) to a topological view.

        Parameters
        ----------
        mat : ndarray, 2-dimensional, optional
            An array containing a design matrix representation of training
            examples. If unspecified, the entire dataset (`self.X`) is used
            instead.
            This parameter is not named X because X is generally used to
            refer to the design matrix for the current problem. In this
            case we want to make it clear that `mat` need not be the design
            matrix defining the dataset.
        """
        if self.view_converter is None:
            raise Exception("Tried to call get_topological_view on a dataset "
                            "that has no view converter")
        if mat is None:
            mat = self.X
        return self.view_converter.design_mat_to_topo_view(mat, mask=self.mask)


class MRIViewConverter(dense_design_matrix.DefaultViewConverter):
    """
    Class for neuroimaging view converters. Takes account 3D.
    """

    def __init__(self, shape, mask=None, axes=('b', 0, 1, 'c')):
        self.__dict__.update(locals())

        if self.mask is not None:
            if not mask.shape == shape:
                raise ValueError("Mask shape (%r) does not fit data shape (%r)"
                                 % (mask.shape, shape))
        super(MRIViewConverter, self).__init__(shape, axes)

    def design_mat_to_topo_view(self, design_matrix):
        """
        ... todo::

            WRITEME
        """
        if len(design_matrix.shape) != 2:
            raise ValueError("design_matrix must have 2 dimensions, but shape "
                             "was %s." % str(design_matrix.shape))

        expected_row_size = np.prod(self.shape)
        if self.mask is not None:
            mask_idx = np.where(self.mask.transpose([self.axes.index(ax) - 1
                                                     for ax in ('c', 0, 1)]).flatten() == 1)[0].tolist()
            assert self.mask.shape == self.shape
            r, c, d = self.mask.shape
            m = design_matrix.shape[0]
            topo_array_bc01 = np.zeros((m, d, r, c), dtype=design_matrix.dtype)
            for i in range(m):
                sample = topo_array_bc01[i].flatten()
                sample[mask_idx] = design_matrix[i]
                topo_array_bc01[i] = sample.reshape((d, r, c))
            axis_order = [('b', 'c', 0, 1).index(axis) for axis in self.axes]
            topo_array = topo_array_bc01.transpose(*axis_order)
        else:
            topo_array = super(MRIViewConverter, self).design_mat_to_topo_view(design_matrix)

        return topo_array

    def design_mat_to_weights_view(self, X):
        """
        .. todo::

            WRITEME
        """
        rval = self.design_mat_to_topo_view(X)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
                                        for axis in ('b', 0, 1, 'c')))

        return rval

    def topo_view_to_design_mat(self, topo_array):
        """
        ... todo::

            WRITEME
        """

        for shape_elem, axis in safe_zip(self.shape, (0, 1, 'c')):
            if topo_array.shape[self.axes.index(axis)] != shape_elem:
                raise ValueError(
                    "topo_array's %s axis has a different size "
                    "(%d) from the corresponding size (%d) in "
                    "self.shape.\n"
                    "  self.shape:       %s (uses standard axis order: 0, 1, "
                    "'c')\n"
                    "  self.axes:        %s\n"
                    "  topo_array.shape: %s (should be in self.axes' order)")

        if self.mask is not None:
            m = topo_array.shape[0]
            mask_idx = np.where(self.mask.transpose([self.axes.index(ax) - 1
                                                for ax in ('c', 0, 1)]).flatten() == 1)[0].tolist()
            design_matrix = np.zeros((m, len(mask_idx)), dtype=topo_array.dtype)
            for i in range(m):
                topo_array_c01 = topo_array[i].transpose([self.axes.index(ax) - 1
                                                          for ax in ('c', 0, 1)])
                design_matrix[i] = topo_array_c01.flatten()[mask_idx]
        else:
            topo_array_bc01 = topo_array.transpose([self.axes.index(ax)
                                                    for ax in ('b', 'c', 0, 1)])
            design_matrix = topo_array_bc01.reshape((topo_array.shape[0],
                                                     np.prod(topo_array.shape[1:])))

        return design_matrix

class MRIViewConverterTransposed(MRIViewConverter):
    """
    Wrapper class to handler transposed datasets.
    """

    def design_mat_to_topo_view(self, design_matrix):
        return super(MRIViewConverterTransposed, self).design_mat_to_topo_view(
            design_matrix.T)

    def design_mat_to_weights_view(self, design_matrix):
        return super(MRIViewConverterTransposed, self).design_mat_to_weights_view(
            design_matrix.T)

    def topo_view_to_design_mat(self, topo_array):
        design_mat = super(MRIViewConverterTransposed, self).topo_view_to_design_mat(
            topo_array)
        return design_mat.T
