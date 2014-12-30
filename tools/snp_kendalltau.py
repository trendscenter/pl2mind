"""
Module to extract top SNPs from snp dataset.
"""
import argparse
import contextlib
import ctypes
import functools
import glob
import logging
import multiprocessing as mp
import numpy as np
from os import path
from pylearn2.neuroimaging_utils.dataset_utils import read_snps
from scipy.stats import kendalltau
from sys import stdout
import time


info = mp.get_logger().info

def load_snp_data(source_directory):
    snps = None
    file_name = path.join(source_directory, "real.chr%d.npy")
    for c in range(1, 22+1):
        chr_snps = np.load(file_name % c)
        if snps is None:
            snps = chr_snps
        else:
            assert chr_snps.shape[0] == snps.shape[0]
            snps = np.concatenate((snps, chr_snps), axis=1)
    return snps

def load_snp_labels(source_directory):
    labels = np.load(path.join(source_directory, "real_labels.npy"))
    return labels

def load_snp_names(snp_dir):
    snp_files = [path.join(snp_dir, "chr" + str(c),
                           "chr%d_risk_n1000.controls.gen" % c) for c in range(1,22+1)]
    names = [read_snps.read_SNP_file(f, read_value="NAMES") for f in snp_files]
    names = [item for sublist in names for item in sublist]
    return names    

def init(shared_arr_):
    global shared_arr
    shared_arr = shared_arr_  # must be inherited, not passed as an argument

def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())

"""
def g(i):
    info("start %s" % (i,))

    taus = tonumpyarray(shared_taus)
    taus[i] = kendalltau(snps[:, i], labels)
    info("Taus: %r" % taus[i])
    info("end   %s" % (i,))
"""
def g(i):
    taus = tonumpyarray(shared_taus)
    taus[i] = taus[i] + kendalltau(snps[:, i], labels)[0]

def kendall_tau_snps(source_directory):
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
    global snps
    global labels
    global shared_taus
    snps = load_snp_data(source_directory)
    labels = load_snp_labels(source_directory)
    info("Getting names")
    names = load_snp_names(
        "/export/research/analysis/human/collaboration/SzImGen/IMPUTE/forsergey/readyforuse/")
    info("Found %d names" % len(names))

    N = snps.shape[1]
    assert len(names) == N
    shared_taus = mp.Array(ctypes.c_double, N)

    t0 = time.clock()
    with contextlib.closing(mp.Pool(initializer=init, initargs=(shared_taus,))) as p:
        x = p.map_async(g, range(N))
    p.join()
    print "Kendall tau with %d SNPs took %r seconds" % (snps.shape[1], time.clock() - t0)
    taus = tonumpyarray(shared_taus)
    return taus

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory", help="Directory for SNPs.")
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    kendall_tau_snps(args.source_directory)
    
