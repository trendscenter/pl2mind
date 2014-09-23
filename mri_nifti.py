"""
Utility for loading MRI data.
"""

__author__ = "Devon Hjelm"
__copyright__ = "Copyright 2014, Mind Research Network"
__credits__ = ["Devon Hjelm"]
__licence__ = "3-clause BSD"
__email__ = "dhjelm@mrn.org"
__maintainer__ = "Devon Hjelm"

import argparse
import numpy as np
from glob import glob
from nipy import save_image, load_image
import os
from os import path
from random import shuffle
import sys
from sys import stdout
import warnings
from pylearn2.utils import serial

def pull_niftis(source_directory, h_pattern, sz_pattern):
    """
    ... todo::
    
    WRITEME

    """

    h_pattern = h_pattern if h_pattern else "H*"
    sz_pattern = sz_pattern if sz_pattern else "S*"
    h_files = glob(path.join(source_directory, h_pattern))
    sz_files = glob(path.join(source_directory, sz_pattern))

    if len(h_files) == 0: warnings.warn("Healthy files are empty with pattern %s" % 
                                        path.join(source_directory, h_pattern))
    if len(sz_files) == 0: warnings.warn("SZ files are empty with pattern %s" %
                                         path.join(source_directory, sz_pattern))

    return h_files, sz_files

def read_niftis(source_directory, h_pattern=None, sz_pattern=None):
    """
    Read niftis.
    """

    h_files, sz_files = pull_niftis(source_directory, h_pattern, sz_pattern)

    labels = [0] * len(h_files) + [1] * len(sz_files)
    data0 = load_image(h_files[0]).get_data()
    if len(data0.shape) == 3:
        x, y, z = data0.shape
        t = 1
    else:
        x, y, z, t = data0.shape

    data = np.zeros(((len(h_files) + len(sz_files)) * t, x, y, z))

    for i, f in enumerate(h_files + sz_files):
        stdout.write("\rLoading subject from file: %s%s" % (f, ''*30))
        stdout.flush()
        nifti = load_image(f)
        subject_data = nifti.get_data()
        data[i*t:(i+1)*t] = subject_data.transpose((3,0,1,2))
    
    stdout.write("\rLoading subject from file: %s\n" % ('DONE' + ''*30))
    return data, labels

def split_save_data(data, labels, train_percentage, directory='smri'):
    number_subjects = data.shape[0]
    subject_idx = range(number_subjects)
    shuffle(subject_idx)
    train_idx = subject_idx[:int(number_subjects * train_percentage)]
    test_idx = subject_idx[int(number_subjects * train_percentage):]
    assert len(train_idx) + len(test_idx) == len(subject_idx),\
        "Train and test do not add up, %d + %d != %d" %\
        (len(train_idx), len(test_idx), len(subject_idx))

    p = serial.preprocess("${PYLEARN2_NI_PATH}/%s/" % directory)
    np.save(path.join(p, "train.npy"), data[train_idx])
    np.save(path.join(p, "train_labels.npy"), [l for i,l in enumerate(labels) if i in train_idx])
    np.save(path.join(p, "test.npy"), data[test_idx])
    np.save(path.join(p, "test_labels.npy"), [l for i,l in enumerate(labels) if i in test_idx])

def save_mask(data, directory='smri'):
    p = serial.preprocess("${PYLEARN2_NI_PATH}/%s/" % directory)
    mask_path = serial.preprocess(p + "mask.npy")
    m, r, c, d = data.shape
    mask = np.zeros((r, c, d))
    mask[np.where(data.mean(axis=0) > data.mean())] = 1
    np.save(mask_path, mask)

def main(source_directory, out_directory, h_pattern=None, sz_pattern=None):
    data, labels = read_niftis(source_directory, h_pattern=h_pattern, sz_pattern=sz_pattern)
    save_mask(data, out_directory)
    split_save_data(data, labels, .80, out_directory)

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--h", default="H*")
    parser.add_argument("--sz", default="S*")
    parser.add_argument("source_directory")
    parser.add_argument("out_directory")
    return parser

if __name__ == "__main__":
    assert path.isdir(serial.preprocess("${PYLEARN2_NI_PATH}")), "Did you export PYLEARN2_NI_PATH?"
    parser = make_argument_parser()
    args = parser.parse_args()

    assert path.isdir(args.source_directory)
    main(args.source_directory, args.out_directory, h_pattern=args.h, sz_pattern=args.sz)
