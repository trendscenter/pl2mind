"""
Test for RBM sMRI tutorial
"""


from os import path
from pl2mind.tools import mri_analysis
from pl2mind.tutorials.rbm_sMRI import train_rbm
from pylearn2.utils import serial

def test_data():
    pylearn2_out_path = path.expandvars("$PYLEARN2_OUTS")
    assert pylearn2_out_path != "", ("PYLEARN2_OUTS environment variable is "
                                     "not set.")

    pylearn2_data_path = path.expandvars("$PYLEARN2_NI_PATH")
    assert pylearn2_data_path != "", ("PYLEARN2_NI_PATH environment"
                                      " variable is not set")

    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/smri/")
    extras_path = serial.preprocess("${PYLEARN2_NI_PATH}/mri_extra/")

    try:
        assert path.isdir(data_path), data_path
        assert path.isdir(extras_path), extras_path
    except AssertionError as e:
        raise IOError("File or directory not found (%s), did you set your "
                      "PYLEARN2_NI_PATH correctly? (%s)" % (e, data_path))

def test_rbm():
    save_path = path.join(serial.preprocess("${PYLEARN2_OUTS}"), "tutorials")
    if not path.isdir(serial.preprocess("${PYLEARN2_OUTS}")):
        raise IOError("PYLEARN2_OUTS environment variable not set")
    train_rbm.train_rbm(epochs = 1, save_path=save_path)
    mri_analysis.main(path.join(save_path, "rbm_smri.pkl"),
                      save_path, "sz_t")