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
    Pull healthy and schizophrenia nitfi files from a source_directory. Uses glob to get multiple files.
    
    Parameters
    ----------
    source_directory: string
        Source directory of nifti files.
    h_pattern: string
        unix-style regex string
    sz_pattern: string
        unix-stule regex string.

    Returns
    -------
    pair of lists.  healthy files, schizophrenia files.
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
    Read niftis.  TODO(dhjelm): use logging.
    """

    h_files, sz_files = pull_niftis(source_directory, h_pattern, sz_pattern)

    labels = [0] * len(h_files) + [1] * len(sz_files)
    data0 = load_image(h_files[0]).get_data()
    if len(data0.shape) == 3:
        x, y, z = data0.shape
        t = 1
    else:
        x, y, z, t = data0.shape

    dt = (len(h_files) + len(sz_files)) * t
    data = np.zeros((dt, x, y, z))

    for i, f in enumerate(h_files + sz_files):
        stdout.write("\rLoading subject from file: %s%s" % (f, '' * 30))
        stdout.flush()
        nifti = load_image(f)
        subject_data = nifti.get_data()
        if len(subject_data.shape) == 3:
            data[i] = subject_data
        elif len(subject_data.shape) == 4:
            data[i * t: (i + 1) * t] = subject_data.transpose((3, 0, 1, 2))
        else:
            raise ValueError("Cannot parse subject data with dimensions %r" % subject_data.shape)
    
    stdout.write("\rLoading subject from file: %s\n" % ('DONE' + ''*30))
    return data, labels

def split_save_data(data, labels, train_percentage, out_dir):
    """
    Randomly split data into training and test and then save.
    
    Parameters
    ----------
    data: array-like
        Data.
    labels: array-like or list
        Labels.
    train_percentage: float
        ratio of train examples to test (0.0-1.0).
    out_dir: string
        Output directory.
    """

    number_subjects = data.shape[0]
    subject_idx = range(number_subjects)
    shuffle(subject_idx)
    train_idx = subject_idx[:int(number_subjects * train_percentage)]
    test_idx = subject_idx[int(number_subjects * train_percentage):]
    assert len(train_idx) + len(test_idx) == len(subject_idx),\
        "Train and test do not add up, %d + %d != %d" %\
        (len(train_idx), len(test_idx), len(subject_idx))

    np.save(path.join(out_dir, "train.npy"), data[train_idx])
    np.save(path.join(out_dir, "train_labels.npy"), [l for i,l in enumerate(labels) if i in train_idx])
    np.save(path.join(out_dir, "test.npy"), data[test_idx])
    np.save(path.join(out_dir, "test_labels.npy"), [l for i,l in enumerate(labels) if i in test_idx])

def save_mask(data, out_dir):
    """
    Find and save maks of data.
    """

    mask_path = path.join(out_dir, "mask.npy")
    m, r, c, d = data.shape
    mask = np.zeros((r, c, d))
    mask[np.where(data.mean(axis=0) > data.mean())] = 1
    np.save(mask_path, mask)

def main(source_directory, out_dir, h_pattern=None, sz_pattern=None):
    data, labels = read_niftis(source_directory, h_pattern=h_pattern, sz_pattern=sz_pattern)
    save_mask(data, out_dir)
    split_save_data(data, labels, .80, out_dir)

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--h", default="H*", help="healthy subject file pattern.")
    parser.add_argument("--sz", default="S*", help="schizophrenia subject file pattern")
    parser.add_argument("source_directory", help="source directory for all subjects.")
    parser.add_argument("out", help="output directory under ${PYLEARN2_NI_PATH}")
    return parser

if __name__ == "__main__":
    assert path.isdir(serial.preprocess("${PYLEARN2_NI_PATH}")), "Did you export PYLEARN2_NI_PATH?"
    parser = make_argument_parser()
    args = parser.parse_args()

    assert path.isdir(args.source_directory), "Source directory not found: %s" % source_directory
    out_dir = serial.preprocess("${PYLEARN2_NI_PATH}" + args.out)
    assert path.isdir(out_dir), "No output directory found (%s), you must make it" % out_dir
    
    main(args.source_directory, out_dir, h_pattern=args.h, sz_pattern=args.sz)
