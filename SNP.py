from pylearn2.datasets import control
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng

from theano import config

class SNP(dense_design_matrix.DenseDesignMatrix):
    """
    Class to read SNP data into pylearn2 format for training.
    """

    def __init__(self, which_set, shuffle=False):
        if which_set not in ['train', 'test']:
            if which_set == 'valid':
                raise ValueError(
                    "Currently valid datasets not supported")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","test"].')

        p = "${PYLEARN2_NI_PATH}/snp/"
        if which_set == 'train':
            data_path = p + 'train.npy'
            label_path = p + 'train_labels.npy'
        else:
            assert which_set == 'test'
            data_path = p + 'test.npy'
            label_path = p + 'test_labels.npy'

        data_path = serial.preprocess(data_path)
        label_path = serial.preprocess(label_path)

        print "Loading data"
        topo_view = np.load(data_path)
        y = np.atleast_2d(np.load(label_path)).T
        samples, number_snps = topo_view.shape
        
        if shuffle:
            self.shuffle_rng = make_np_rng(None, default_seed=[1, 2, 3], which_method="shuffle")
            for i in xrange(samples):
                j = self.shuffle_rng.randint(samples)
                tmp = topo_view[i].copy()
                topo_view[i] = topo_view[j]
                topo_view[j] = tmp
                tmp = y[i,i+1].copy()
                y[i] = y[j]
                y[j] = tmp
        
        super(SNP, self).__init__(topo_view=topo_view, y=y, y_labels=np.amax(y))
