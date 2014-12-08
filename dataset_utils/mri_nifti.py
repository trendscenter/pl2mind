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
from glob import glob
import logging
from nipy import save_image, load_image
import numpy as np
import os
from os import path
import pickle
from random import shuffle
import re
from scipy import io
from scipy.stats import kurtosis
from scipy.stats import skew
import sys
from sys import stdout
import warnings
from pylearn2.utils import serial

logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def pull_niftis(source_directory, *args):
    """
    Pull healthy and schizophrenia nitfi files from a source_directory. Uses glob to get multiple files.
    
    Parameters
    ----------
    source_directory: string
        Source directory of nifti files.
    *args: list of strings or lists. optional
        Either a string for use with glob or a list of files. If a string is passed, then the list of files
        will be derived from glob.

    Returns
    -------
    list of lists. Each a list of files for each class.
    """
    if len(args) == 0:
        file_lists = [glob(path.join(source_directory, "*.nii"))]
    else:
        file_lists = []
        for arg in args:
            if isinstance(arg, list):
                if len(arg) == 0:
                    logger.warn("File list is empty.")
                file_list = []
                for file_name in arg:
                    if source_directory in file_name:
                        file_list.append(file_name)
                    else:
                        file_list.append(path.join(source_directory, file_name))
                file_lists.append(file_list)
            elif isinance(arg, str):
                file_list = glob(path.join(source_directory, arg))
                if len(file_list) == 0: 
                    logger.warn("Files are empty with pattern %s" % 
                                path.join(source_directory, arg))
                file_lists.append(file_list)
            else:
                raise ValueError("%s type not supported" % type(arg))

    for file_list in file_lists:
        for file_name in file_list:
            assert source_directory in file_name, file_name

    return file_lists

def read_niftis(source_directory, *args):
    """
    Read niftis.
    """

    file_lists = pull_niftis(source_directory, *args)

    data0 = load_image(file_lists[0][0]).get_data()
    if len(data0.shape) == 3:
        x, y, z = data0.shape
        t = 1
    elif len(data0.shape) == 4:
        x, y, z, t = data0.shape
    else:
        raise ValueError("Cannot parse data with dimensions %r" % data0.shape)

    dt = (sum([len(fl) for fl in file_lists])) * t
    data = np.zeros((dt, x, y, z))

    labels = [[i] * (len(fl) * t) for i, fl in enumerate(file_lists)]
    labels = [item for sublist in labels for item in sublist]

    for i, fl in enumerate(file_lists):
        assert len([j for j in labels if j == i]) == len(fl) * t

    for i, f in enumerate([item for sublist in file_lists for item in sublist]):
        logger.info("Loading subject from file: %s%s" % (f, '' * 30))
        
        nifti = load_image(f)
        subject_data = nifti.get_data()

        if len(subject_data.shape) == 3:
            data[i] = subject_data
        elif len(subject_data.shape) == 4:
            data[i * t: (i + 1) * t] = subject_data.transpose((3, 0, 1, 2))
        else:
            raise ValueError("Cannot parse subject data with dimensions %r" % subject_data.shape)
    
    logger.info("\rLoading subject from file: %s\n" % ('DONE' + " "*30))
    if data.shape[0] != len(labels):
        raise ValueError("Data and labels have different number of samples.")
    return data, labels

def test_distribution(data, mask=None):
    logger.info("Testing distribution.")
    data = data.reshape(data.shape[0], reduce(lambda x, y: x * y, data.shape[1:4]))
    if mask is not None:
        mask_idx = np.where(mask.flatten() == 1)[0].tolist()
        data = data[:, mask_idx]
    k = kurtosis(data, axis=0)
    s = skew(data, axis=0) 
    
    logger.info("Proportion voxels k <= -1: %.2f"
                % (len(np.where(k <= -1)[0].tolist()) * 1. / data.shape[1]))
    logger.info("Proportion voxels -1 < k < 1: %.2f"
                % (len(np.where(np.logical_and(k > -1, k < 1))[0].tolist()) * 1. / data.shape[1]))
    logger.info("Proportion voxels 1 < k < 2: %.2f"
                % (len(np.where(np.logical_and(k >= 1, k < 2))[0].tolist()) * 1. / data.shape[1]))
    logger.info("Proportion voxels 2 < k < 3: %.2f"
                % (len(np.where(np.logical_and(k >= 2, k < 3))[0].tolist()) * 1. / data.shape[1]))
    logger.info("Proportion voxels k >= 3: %.2f"
                % (len(np.where(k >= 3)[0].tolist()) * 1. / data.shape[1]))

    values = len(np.unique(data))
    if (values * 1. / reduce(lambda x, y: x * y, data.shape) < 10e-4):
        logger.warn("Quantization probable (%d unique values out of %d)."
                    % (values, reduce(lambda x, y: x * y, data.shape)))
    logger.info("Number of unique values in data: %d" % values)

    logger.info("Krutosis k: %.2f (%.2f std) and skew s: %.2f (%.2f std)"
                % (k.mean(), k.std(), s.mean(), s.std()))

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
    if len(train_idx) + len(test_idx) != len(subject_idx):
        raise ValueError("Train and test do not add up, %d + %d != %d" %
                         (len(train_idx), len(test_idx), len(subject_idx)))

    np.save(path.join(out_dir, "train_idx.npy"), train_idx)
    np.save(path.join(out_dir, "test_idx.npy"), test_idx)
    np.save(path.join(out_dir, "full_unshuffled.npy"), data)
    np.save(path.join(out_dir, "full_labels_unshuffled.npy"), labels)
    np.save(path.join(out_dir, "train.npy"), data[train_idx])
    np.save(path.join(out_dir, "train_labels.npy"), [labels[i] for i in train_idx])
    np.save(path.join(out_dir, "test.npy"), data[test_idx])
    np.save(path.join(out_dir, "test_labels.npy"), [labels[i] for i in test_idx])

def save_mask(data, out_dir):
    """
    Find and save maks of data.
    """
    logger.info("Finding mask.")
    mask_path = path.join(out_dir, "mask.npy")
    m, r, c, d = data.shape
    mask = np.zeros((r, c, d))

    zero_freq = (data.reshape((m, r * c * d)) == 0).sum(1) * 1. / reduce(lambda x, y: x * y, data.shape[1:4])
    if zero_freq.mean() > 0.2:
        logger.info("Masked data found, deriving zeros from data zeros.")
        for freq in zero_freq:
            assert isinstance(freq, float), freq
            if abs(zero_freq.mean() - freq) > .05:
                raise ValueError("Spurious datapoint, mean zeros frequency is %.2f,"
                                 "datapoint is %.2f" % (zero_freq.mean(), freq))
        mask[np.where(np.invert((data < 0.07).sum(0) > .01 * data.shape[0]))] = 1 
    else:
        logger.info("Deriving mask from mean image.")
        mask[np.where(data.mean(axis=0) > data.mean())] = 1

    logger.info("Masked out %d out of %d voxels" % ((mask == 0).sum(), reduce(lambda x, y: x * y, mask.shape)))
    np.save(mask_path, mask)
    return mask

def load_simTB_data(source_directory):
    """
    Load simTB data along with simulation info.
    """
    nifti_files = natural_sort(glob(path.join(source_directory, "*_DATA.nii")))
    sim_files = natural_sort(glob(path.join(source_directory, "*_SIM.mat")))
    if len(nifti_files) != len(sim_files):
        raise ValueError("Different number of DATA and SIM files found int %s" % source_directory)
    assert len(nifti_files) > 0

    param_files = glob(path.join(source_directory, "*PARAMS.mat"))
    if len(param_files) != 1:
        raise ValueError("Exactly one param file needed, found %d in %s"
                         % (len(param_files), source_directory))
    params = tuple(io.loadmat(param_files[0])["sP"][0][0])

    sim_dict = {}
    for i, (nifti_file, sim_file) in enumerate(zip(nifti_files, sim_files)):
        assert "%03d" % (i + 1) in nifti_file
        assert "%03d" % (i + 1) in sim_file
        sims = io.loadmat(sim_file)
        tcs = sims["TC"].T
        sms = sims["SM"]
        sim_dict[i] = {"SM": sms, "TC": tcs}
    sim_dict["params"] = params

    data, labels = read_niftis(source_directory, nifti_files)
    return data, labels, sim_dict

def is_simTBdir(source_directory):
    """
    Returns True is source_directory fits the criteria of being a simTB source directory.
    """
    nifti_files = natural_sort(glob(path.join(source_directory, "*_DATA.nii")))
    sim_files = natural_sort(glob(path.join(source_directory, "*_SIM.mat")))

    if (len(nifti_files) == len(sim_files)) and len(nifti_files) > 0:
        logger.info("simTB directory detected")
        return True
    return False

def main(source_directory, out_dir, args):
    if is_simTBdir(source_directory):
        data, labels, sim_dict = load_simTB_data(source_directory)
    else:
        if args.h_pattern is not None or args.sz_pattern is not None:
            read_args = [patt for patt in (args.h_pattern, args.sz_pattern) if patt is not None]
        else:
            read_args = []
        data, labels = read_niftis(source_directory, *read_args)
        sim_dict = None

    mask = save_mask(data, out_dir)
    if args.verbose:
        test_distribution(data, mask)
    split_save_data(data, labels, .80, out_dir)
    if sim_dict is not None:
        pickle.dump(sim_dict, open(path.join(out_dir, "sim_dict.pkl"), "wb"))

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """

    parser = argparse.ArgumentParser()
 
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more verbosity!")
    parser.add_argument("--h_pattern", default=None, help="healthy subject file pattern.")
    parser.add_argument("--sz_pattern", default=None, help="schizophrenia subject file pattern")
    parser.add_argument("source_directory", help="source directory for all subjects.")
    parser.add_argument("out", help="output directory under ${PYLEARN2_NI_PATH}")
    return parser

if __name__ == "__main__":
    if not path.isdir(serial.preprocess("${PYLEARN2_NI_PATH}")):
        raise ValueError("Did you export PYLEARN2_NI_PATH?")
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    assert path.isdir(args.source_directory), "Source directory not found: %s" % source_directory
    out_dir = serial.preprocess("${PYLEARN2_NI_PATH}" + args.out)
    assert path.isdir(out_dir), "No output directory found (%s), you must make it" % out_dir
    
    main(args.source_directory, out_dir, args)
