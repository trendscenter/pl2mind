from copy import deepcopy
import functools
from glob import glob
import numpy as np
from os import path
#import tables

from pylearn2.datasets import control
from pylearn2.datasets import Dataset
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrixPyTables as PyTables
from pylearn2.space import CompositeSpace
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.models import mlp
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils import safe_zip
from pylearn2.utils import serial
from pylearn2.utils.rng import make_np_rng

from theano import config
from theano.compat.python2x import OrderedDict

from pylearn2.neuroimaging_utils.research import randomize_snps

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
    def __init__(self, num_layers, layer_to_copy):
        layer_name = 'multi_chromosome_layer'

        layers = []
        inputs_to_layers = {}
        for n in range(num_layers):
            layer = deepcopy(layer_to_copy)
            layer.layer_name += "_%d" % n
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
        
class MultiChromosome(Dataset):
    """
    Class to read multiple chromosome data.
    """
    _default_seed = (18, 4, 646)
    def __init__(self,
                 chromosomes="ALL",
                 dataset_name="snp",
                 read_only=False, balance_classes=False,
                 start=None, stop=None, shuffle=False,
                 add_noise=False, rng=_default_seed):

        assert int(chromosomes) or chromosomes == "ALL",\
            "Can only set chromosomes to be an integer or ALL"

        p = serial.preprocess("${PYLEARN2_NI_PATH}/" + dataset_name)
        data_files = glob(path.join(p, "chr*.npy"))
        label_file = path.join(p, "labels.npy")

        get_int = lambda y: int(''.join(x for x in y if x.isdigit()))
        data_files.sort(key=get_int)

        if (chromosomes == "ALL" or chromosomes > len(data_files)):
            chromosomes = len(data_files)

        self.y = np.atleast_2d(np.load(label_file)).T[start:stop]
        self.Xs = ()
        space = ()
        source = ()

        balanced_idx = None
        if balance_classes:
            num_classes = np.amax(self.y) + 1
            class_counts = [len(np.where(self.y == i)[0].tolist())
                            for i in range(num_classes)]
            min_count = min(class_counts)
            balanced_idx = []
            for i in range(num_classes):
                idx = np.where(self.y == i)[0].tolist()[:min_count]
                balanced_idx += idx
            balanced_idx.sort()
            assert len(balanced_idx) / min_count == num_classes
            assert len(balanced_idx) % min_count == 0

            self.y = self.y[balanced_idx]
            for i in range(num_classes):
                assert len(np.where(self.y == i)[0].tolist()) == min_count
        
        if read_only:
            print "Format is read-only for %s" % which_set
            h5_path = path.join(p, "gen." + which_set + ".h5")
            
            if not path.isfile(h5_path):
                self.make_h5(data_files,
                             h5_path,
                             start=start,
                             stop=stop)

            h5file = tables.openFile(h5_path)
            datas = [h5file.getNode("/", "Chr%d" % (c + 1)) for c in range(chromosomes)]
            self.Xs = tuple(data.X for data in datas)
            sizes = [h5file.getNode("/", "Sizes")[c] for c in range(chromosomes)]

        else:
            print "Format is on-memory for %s" % dataset_name
            sizes = []
            for c in range(0, chromosomes):
                X = np.load(data_files[c])[start:stop, :]

                assert "%d" % (c+1) in data_files[c]

                if balanced_idx is not None:
                    X = X[balanced_idx]

                assert X.shape[0] == self.y.shape[0],\
                    "Data and labels have different number of samples (%d vs %d)" %\
                    (X.shape[0], self.y.shape[0])

                self.Xs = self.Xs + (X / 2.0,)
                sizes.append(X.shape[1])

        print "%s samples are %d" % (dataset_name, self.y.shape[0])

        space = tuple(VectorSpace(dim=size) for size in sizes)
        source = tuple("chromosomes_%d" % (c + 1) for c in range(chromosomes))

        self.X_space = CompositeSpace(space)
        self.X_source = source

        space = space + (IndexSpace(dim=1, max_labels=2),)
        source = source + ("targets",)
        space = CompositeSpace(space)
        
        self.data_specs = (space, source)
        self.rng = make_np_rng(rng, which_method="random_integers")
        assert self.rng is not None

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class("sequential")
        self._iter_topo = False
        self._iter_targets = False
        self._iter_data_specs = self.data_specs

        if add_noise:
            if add_noise is True:
                add_noise = 0.05
            self.convert = list(randomize_snps.RandomizeSNPs(input_space=x_space,
                                                             corruption_prob=add_noise)
                            for x_space in self.X_space.components) + [None]
        else:
            self.convert = None

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        space, source = self.data_specs
        subspaces = space.components
        subsources = source
        mode = resolve_iterator_class("shuffled_sequential")
        if rng is None:
            rng = self.rng
        rng = None
#        assert rng is not None
        subset_iterator = mode(self.y.shape[0],
                               batch_size,
                               num_batches,
                               rng=rng)

        return FiniteDatasetIterator(
            self,
            subset_iterator=subset_iterator,
            data_specs=data_specs,
            return_tuple=return_tuple,
            convert=self.convert)
    
    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.
        """
        return self.data_specs

    def get_data(self):
        return self.Xs + (self.y, )

    def has_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y is not None

    @functools.wraps(Dataset.get_num_examples)
    def get_num_examples(self):
        return self.y.shape[0]

    def make_h5(self, source_paths, h5_path, start=None, stop=None, filters=None):
        print "Making h5 file for %s" % h5_path
        h5file = tables.openFile(h5_path, mode='w', title="SNP Dataset")
        if filters is None:
            filters = tables.Filters(complib='blosc', complevel=5)

        sizes = []
        for c, source_path in enumerate(source_paths):
            X = np.load(source_path)
            if start is not None and stop is not None:
                assert 0 <= start < stop
                X = X[start:stop, :] / (2.0)

            atom = (tables.Float32Atom() if config.floatX == "float32" 
                    else tables.Float64Atom())
            gcolumns = h5file.createGroup(h5file.root,
                                          "Chr%d" % (c + 1),
                                          "Chromosome %d" % (c + 1))
            node = h5file.getNode("/", "Chr%d" % (c + 1))            
            h5file.createCArray(gcolumns, "X",
                                atom=atom,
                                shape=X.shape,
                                title="Chromosome %d" % (c + 1),
                                filters=filters)
            PyTables.fill_hdf5(h5file, X, node=node)

            sizes.append(X.shape[1])

        h5file.createArray(h5file.root, "Sizes", sizes)

        h5file.close()


class SNP(dense_design_matrix.DenseDesignMatrix):
    """
    Class to read SNP data into pylearn2 format for training.
    """

    def __init__(self, which_set, start=None, stop=None, shuffle=False):
        if which_set not in ['train', 'valid']:
            if which_set == 'test':
                raise ValueError(
                    "Currently test datasets not supported")
            raise ValueError(
                'Unrecognized which_set value "%s".' % (which_set,) +
                '". Valid values are ["train","valid"].')

        p = "${PYLEARN2_NI_PATH}/snp/"
        if which_set == 'train':
            data_path = p + 'gen.chr1.npy'
            label_path = p + 'gen.chr1_labels.npy'
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
        
        if start is not None:
            stop = stop if (stop <= samples) else samples
            assert 0 <= start < stop
            topo_view = topo_view[start:stop, :]
            y = y[start:stop]

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
        
        super(SNP, self).__init__(X=topo_view, y=y, y_labels=np.amax(y)+1)
