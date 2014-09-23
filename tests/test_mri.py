from pylearn2.neuroimaging_utils.sMRI import MRI_On_Memory
from pylearn2.neuroimaging_utils.sMRI import MRI_Big
from pylearn2.utils import serial
import numpy as np

class TestMRI:
    def setUp(self):
        p = "${PYLEARN2_NI_PATH}/smri/"
        self.data_path = serial.preprocess(p + 'test.npy')
        self.mask_path = serial.preprocess(p + 'mask.npy')

    def test_mask(self):
        mask = np.load(self.mask_path)
        rows, columns, depth = mask.shape
        mri = MRI_On_Memory('test', apply_mask=True, dataset_name='smri')

        topo_view = mri.get_topological_view()
        X = mri.X
        samples = topo_view.shape[0]
        assert topo_view.shape == (samples, rows, columns, depth)
        data = np.load(self.data_path)
        assert topo_view.shape == data.shape

        for i in range(rows):
            for j in range(columns):
                for k in range(depth):
                    for s in range(samples):
                        if mask[i][j][k] == 0:
                            assert topo_view[s][i][j][k] == 0,\
                                "Found non-zero outside mask, %d, %d, %d, %d" % (s, i, j, k)
                        else:
                            assert topo_view[s][i][j][k] == data[s][i][j][k],\
                                "Topo view not equal to data inside mask, %d, %d, %d, %d" % (s, i, j, k)

        design_mat = mri.get_design_matrix(topo=topo_view)
        assert design_mat.shape == X.shape
        assert len(np.where(mask.flatten() == 1)[0].tolist()) == X.shape[1]
        for s in range(samples):
            for i in range(design_mat.shape[1]):
                assert X[s][i] == design_mat[s][i]

    def test_pytable(self, center=False, variance_normalize=False):
        mask = np.load(self.mask_path)
        mri_mem = MRI_On_Memory('test', apply_mask=True,
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
                            "Topo views different at (%d,%d,%d,%d): (%.10f vs %.10f)" %(s, i, j, k, 
                                                                                        tv_mem[s][i][j][k],
                                                                                        tv_big[s][i][j][k])
        dm_mem = mri_mem.get_design_matrix(tv_mem)
        dm_big = mri_big.get_design_matrix(tv_big)
        assert dm_mem.shape == dm_big.shape
        for s in range(samples):
            for i in range(dm_mem.shape[1]):
                assert dm_mem[s][i] == dm_big[s][i]
        mri_big.h5file.close()

    def test_pytable_with_normalization(self):
        self.test_pytable(center=True, variance_normalize=True)
