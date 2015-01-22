"""
Module to test MRI datasets.
"""

import logging
import numpy as np
from os import path

from pl2mind.datasets.MRI import MRI_Standard
from pl2mind.datasets.MRI import MRI_Transposed
from pl2mind.datasets.MRI import MRI_Big
from pylearn2.utils import serial

import sys

logging.basicConfig(format="%[levelname]s:%(message)s", level=logging.DEBUG)
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

def check_datasets(exp, act):
    assert np.all(exp == act),\
        "Datasets do not match: \n%r\n%r" % (exp, act)

class TestMRI:
    def setUp(self):
        self.p = serial.preprocess("${PYLEARN2_NI_PATH}/smri/")
        self.data_path = path.join(self.p, 'test.npy')
        self.mask_path = path.join(self.p, 'mask.npy')

    def test_data(self):
        mask = np.load(self.mask_path)
        """
        Test confirms training and testing dataset data and labels.
        Checks if data and labels line up with the full dataset giving
        the indices.
        """
        mri_train = MRI_Standard("train", apply_mask=False,
                                 center=False, variance_normalize=False,
                                 dataset_name="smri")
        train_idx = np.load(path.join(self.p, "train_idx.npy"))
        mri_test = MRI_Standard("test", apply_mask=False,
                                center=False, variance_normalize=False,
                                dataset_name="smri")
        test_idx = np.load(path.join(self.p, "test_idx.npy"))
        full = np.load(path.join(self.p, "full_unshuffled.npy"))
        labels = np.load(path.join(self.p, "full_labels_unshuffled.npy"))
        check_datasets(mri_train.get_design_matrix(full[train_idx]),
                       mri_train.X)
        check_datasets(labels[train_idx], mri_train.y.flatten())
        check_datasets(mri_test.get_design_matrix(full[test_idx]),
                       mri_test.X)
        check_datasets(labels[test_idx], mri_test.y.flatten())

    def test_mask_with_evening(self):
        self.test_mask(even_input=True)

    def test_mask(self, transposed=False, even_input=False):
        mask = np.load(self.mask_path)
        rows, columns, depth = mask.shape
        if not transposed:
            mri = MRI_Standard("test", apply_mask=True, dataset_name="smri", even_input=even_input)
        else:
            mri = MRI_Transposed("test", apply_mask=True, dataset_name="smri", even_input=even_input)

        topo_view = mri.get_topological_view()
        if even_input:
            ons = np.where(mask == 1)
            i, j, k = (ons[r][0] for r in range(3))
            mask[i, j, k] = 0

        X = mri.X
        samples = topo_view.shape[0]
        assert topo_view.shape == (samples, rows, columns, depth)
        data = np.load(self.data_path)
        assert topo_view.shape[1:] == data.shape[1:], "topo view and data shape mismatch %r vs %r" % (topo_view.shape, data.shape)

        for i in range(rows):
            for j in range(columns):
                for k in range(depth):
                    for s in range(samples):
                        if mask[i][j][k] == 0:
                            assert topo_view[s][i][j][k] == 0,\
                                "Found non-zero outside mask, %d, %d, %d, %d"\
                                % (s, i, j, k)
                        else:
                            assert topo_view[s][i][j][k] == data[s][i][j][k],\
                                "Topo view not equal to data inside mask, %d, %d, %d, %d"\
                                % (s, i, j, k)

        design_mat = mri.get_design_matrix(topo=topo_view)
        assert design_mat.shape == X.shape
        if not transposed:
            assert len(np.where(mask.flatten() == 1)[0].tolist()) == X.shape[1]
        else:
            assert len(np.where(mask.flatten() == 1)[0].tolist()) == X.shape[0]
        for s in range(samples):
            for i in range(design_mat.shape[1]):
                assert X[s][i] == design_mat[s][i]

    def test_transpose(self):
        mri = MRI_Transposed(apply_mask=False, dataset_name="smri")
        topo_view = mri.get_topological_view()
        X = mri.X.T
        assert X.shape[1] == topo_view.shape[1] * topo_view.shape[2] * topo_view.shape[3],\
            "Shaped don't match, X has shape %s while topo_view has shape %s." % (X.shape,
                                                                                topo_view.shape)
        assert np.allclose(topo_view, X.reshape(X.shape[0],
                                                topo_view.shape[3],
                                                topo_view.shape[1],
                                                topo_view.shape[2]).transpose((0,2,3,1))),\
                                                "Transpose reshape failed"

    def test_transpose_with_mask(self):
        self.test_mask(transposed=True)

    def _test_pytable(self, center=False, variance_normalize=False):
        mask = np.load(self.mask_path)
        mri_mem = MRI_Standard('test', apply_mask=True,
                                center=center, variance_normalize=variance_normalize,
                                dataset_name='smri')
        mri_big = MRI_Big('test', apply_mask=True,
                          center=center, variance_normalize=variance_normalize,
                          dataset_name='smri', reprocess=True, save_dummy=True)
        X_mem = mri_mem.X
        X_big = mri_big.X
        assert X_mem.shape == X_big.shape
        for s in range(20):
            for i in range(20):
                assert X_mem[s][i] == X_big[s][i]

        tv_mem = mri_mem.get_topological_view()
        tv_big = mri_big.get_topological_view()
        samples, rows, columns, depth = tv_mem.shape
        assert mask.shape == (rows, columns, depth)
        for i in range(rows):
            for j in range(columns):
                for k in range(depth):
                    for s in range(samples):
                        assert tv_mem[s][i][j][k] == tv_big[s][i][j][k],\
                            "Topo views different at (%d,%d,%d,%d): (%.10f vs %.10f)"\
                            %(s, i, j, k,
                              tv_mem[s][i][j][k],
                              tv_big[s][i][j][k])
        dm_mem = mri_mem.get_design_matrix(tv_mem)
        dm_big = mri_big.get_design_matrix(tv_big)
        assert dm_mem.shape == dm_big.shape
        for s in range(samples):
            for i in range(dm_mem.shape[1]):
                assert dm_mem[s][i] == dm_big[s][i]
        mri_big.h5file.close()

    def _test_pytable_with_normalization(self):
        self.test_pytable(center=True, variance_normalize=True)
