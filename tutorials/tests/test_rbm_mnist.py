"""
Test for RBM MNIST tutorial
"""


from os import path
from pl2mind.tutorials.rbm_mnist import train_rbm
from pylearn2.scripts import show_weights
from pylearn2.utils import serial

def test_data():
    pylearn2_data_path = path.expandvars("$PYLEARN2_DATA_PATH")
    assert pylearn2_data_path != "", ("PYLEARN2_DATA_PATH environment"
                                      " variable is not set")

    data_path = serial.preprocess("${PYLEARN2_DATA_PATH}/mnist/")
    try:
        assert path.isdir(data_path), data_path
        assert path.isfile(path.join(data_path, "t10k-images-idx3-ubyte")),\
            "t10k-images-idx3-ubyte"
        assert path.isfile(path.join(data_path, "t10k-labels-idx1-ubyte")),\
            "t10k-labels-idx1-ubyte"
        assert path.isfile(path.join(data_path, "train-images-idx3-ubyte")),\
            "train-images-idx3-ubyte"
        assert path.isfile(path.join(data_path, "train-labels-idx1-ubyte")),\
            "train-labels-idx1-ubyte"
    except AssertionError as e:
        raise IOError("File or directory not found (%s), did you set "
                      "PYLEARN2_DATA_PATH correctly? (%s)" % (e, data_path))

def test_rbm():
    save_path = path.join(serial.preprocess("${PYLEARN2_OUTS}"), "tutorials")
    if not path.isdir(serial.preprocess("${PYLEARN2_OUTS}")):
        raise IOError("PYLEARN2_OUTS environment variable not set")

    train_rbm.train_rbm(epochs = 1, save_path=save_path)
    show_weights.show_weights(path.join(save_path, "rbm_mnist.pkl"),
                              out=path.join(save_path, "rbm_mnist_weights.png"))
