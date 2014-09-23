from nipy import load_image
from nipy.core.api import Image
import numpy as np
import os
from pylearn2.datasets import control
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import contains_nan
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng
import tables
from theano import config


class MRI_Big(dense_design_matrix.DenseDesignMatrixPyTables):
    def __init__(self, which_set, center=False, variance_normalize=False,
                 shuffle=False, apply_mask=False, preprocessor=None, dataset_name='smri',
                 reprocess=False, save_dummy=False):

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "Currently validation dataset is not supported with"
                    "sMRI.  This can be added in smri_nifti.py"
                    )
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')
        
        self.__dict__.update(locals())
        del self.self

        p = "${PYLEARN2_NI_PATH}/%s/" % dataset_name

        if which_set == 'train':
            data_path = p + 'train.h5'
        else:
            assert which_set == 'test'
            data_path = p + 'test.h5'

        if save_dummy:
            data_path = "".join(data_path.split(".")[0] + '_dummy.h5')

        data_path = serial.preprocess(data_path)

        self.mask = None
        mask_path = serial.preprocess(p + "mask.npy")
        mask = np.load(mask_path)
        rows, columns, depth = mask.shape
        if apply_mask:
            self.mask = mask

        if not os.path.isfile(data_path) or reprocess:
            self.filters = tables.Filters(complib='blosc', complevel=5)
            self.make_data(which_set, serial.preprocess(p),
                           center=center,
                           variance_normalize=variance_normalize,
                           shuffle=shuffle, save_dummy=save_dummy)

        self.h5file = tables.openFile(data_path)
        data = self.h5file.getNode('/', "Data")
        view_converter = NIViewConverter((rows, columns, depth))

        super(MRI_Big, self).__init__(X=data.X, y=data.y,
                                      view_converter=view_converter)

        self.h5file.flush()

    def make_data(self, which_set, p, center=False, variance_normalize=False,
                  shuffle=False, save_dummy=False):
        print "Making h5 file for %s" % which_set

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

        topo_view = np.load(source_path)
        y = np.load(label_path)
        num_labels = np.amax(y) + 1

        samples, rows, columns, depth = topo_view.shape
        if self.mask is not None:
            assert self.mask.shape == (rows, columns, depth)
            size = len(np.where(self.mask.flatten() == 1)[0].tolist())
        else:
            size = rows * columns * depth

        self.view_converter = NIViewConverter((rows, columns, depth))
        X = self.view_converter.topo_view_to_design_mat(topo_view, self.mask)
#        X = topo_view.reshape((samples, rows * columns * depth))
        print topo_view.shape
        print X.shape
#        del topo_view

        one_hot = np.zeros((size, num_labels), dtype=config.floatX)
        for i in xrange(y.shape[0]):
            one_hot[i, y[i] - 1] = 1.

        if center:
            print "Centering data"
            X -= X.mean(axis=0)

        if variance_normalize:
            print "Normalizing data"
            X /= X.std(axis=0)
#        X[np.where(np.isnan(X))] = 0

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
        m, r, c, d = W.shape
        base_nifti_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri/basenifti.nii")
        base_nifti = load_image(base_nifti_path)

        data = np.zeros([r, c, d, m])

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


class MRI_On_Memory(dense_design_matrix.DenseDesignMatrix):
    """
    ... todo::

        WRITEME
    """
    
    def __init__(self, which_set, center=False, variance_normalize=False,
                 shuffle=False, apply_mask=False, preprocessor=None, dataset_name='smri'):
        self.args = locals()

        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "Currently validation dataset is not supported with"
                    "sMRI.  This can be added in smri_nifti.py"
                    )
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

#        if control.get_load_data():
        p = "${PYLEARN2_NI_PATH}/%s/" % dataset_name
        if which_set == 'train':
            data_path = p + 'train.npy'
            label_path = p + 'train_labels.npy'
        else:
            assert which_set == 'test'
            data_path = p + 'test.npy'
            label_path = p + 'test_labels.npy'

        data_path = serial.preprocess(data_path)
        label_path = serial.preprocess(label_path)

        # can locally cache here.  see datasets/mnist.py
        
        print "Loading data"
        topo_view = np.load(data_path)

        y = np.atleast_2d(np.load(label_path)).T
        m, r, c, d = topo_view.shape
        print "Data shape is: ", topo_view.shape

        max_labels = np.amax(y) + 1
        
        self.mask = None
        if apply_mask:
            print "Loading mask"
            mask_path = serial.preprocess(p + "mask.npy")
            self.mask = np.load(mask_path)
            assert self.mask.shape == (r, c, d)
            
        if shuffle:
            print "Shuffling data"
            self.shuffle_rng = make_np_rng(None, [1 ,2 ,3], which_method="shuffle")
            for i in xrange(m):
                j = self.shuffle_rng.randint(m)
                tmp = topo_view[i, :, :, :].copy()
                topo_view[i, :, :, :] = topo_view[j, :, :, :]
                topo_view[j, :, :, :] = tmp
                tmp = y[i:i+1].copy()
                y[i] = y[j]
                y[j] = tmp
                
        X = self.set_ni_topological_view(topo_view)
        
        if center:
            print "Centering data"
            X -= X.mean(axis=0)

        if variance_normalize:
            print "Normalizing data"
            X /= X.std(axis=0)

        super(MRI_On_Memory, self).__init__(X=X, y=y, view_converter=self.view_converter,
                                   y_labels=max_labels)

        assert not np.any(np.isnan(self.X))

    def set_ni_topological_view(self, topo_view, axes=('b', 0, 1, 'c')):
        """
        ... todo::

            WRITEME
        """
        assert not contains_nan(topo_view)
        r, c, d = tuple(topo_view.shape[axes.index(i)] for i in (0, 1, 'c'))
        self.view_converter = NIViewConverter((r, c, d), axes=axes)
        X = self.view_converter.topo_view_to_design_mat(topo_view, self.mask)
        return X

    def get_nifti(self, W):
        m, r, c, d = W.shape
        base_nifti_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri/basenifti.nii")
        base_nifti = load_image(base_nifti_path)

        data = np.zeros([r, c, d, m])

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


class NIViewConverter(dense_design_matrix.DefaultViewConverter):
    def __init__(self, shape, axes=('b', 0, 1, 'c')):
        super(NIViewConverter, self).__init__(shape, axes)

    def design_mat_to_topo_view(self, design_matrix, mask=None):
        """
        ... todo::

            WRITEME
        """
        if len(design_matrix.shape) != 2:
            raise ValueError("design_matrix must have 2 dimensions, but shape "
                             "was %s." % str(design_matrix.shape))

        expected_row_size = np.prod(self.shape)
        if mask is not None:
            mask_idx = np.where(mask.transpose([self.axes.index(ax)-1
                                                for ax in ('c', 0, 1)]).flatten() == 1)[0].tolist()
            assert mask.shape == self.shape
            r, c, d = mask.shape
            m = design_matrix.shape[0]
            topo_array_bc01 = np.zeros((m, d, r, c))
            for i in range(m):
                sample = topo_array_bc01[i].flatten()
                sample[mask_idx] = design_matrix[i]
                topo_array_bc01[i] = sample.reshape((d, r, c))
            axis_order = [('b', 'c', 0, 1).index(axis) for axis in self.axes]
            topo_array = topo_array_bc01.transpose(*axis_order)
        else:
            topo_array = super(NIViewConverter, self).design_mat_to_topo_view(design_matrix)

        return topo_array

    def design_mat_to_weights_view(self, X, mask=None):
        """
        .. todo::

            WRITEME
        """
        rval = self.design_mat_to_topo_view(X, mask)

        # weights view is always for display
        rval = np.transpose(rval, tuple(self.axes.index(axis)
                                        for axis in ('b', 0, 1, 'c')))

        return rval

    def topo_view_to_design_mat(self, topo_array, mask=None):
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

        if mask is not None:
            m = topo_array.shape[0]
            mask_idx = np.where(mask.transpose([self.axes.index(ax)-1
                                                for ax in ('c', 0, 1)]).flatten() == 1)[0].tolist()
            design_matrix = np.zeros((m, len(mask_idx)))
            for i in range(m):
                topo_array_c01 = topo_array[i].transpose([self.axes.index(ax)-1
                                                          for ax in ('c', 0, 1)])
                design_matrix[i] = topo_array_c01.flatten()[mask_idx]
        else:
            topo_array_bc01 = topo_array.transpose([self.axes.index(ax)
                                                    for ax in ('b', 'c', 0, 1)])
            design_matrix = topo_array_bc01.reshape((topo_array.shape[0],
                                                     np.prod(topo_array.shape[1:])))

        return design_matrix
