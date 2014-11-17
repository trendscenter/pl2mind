"""
Module to perform analysis on MRI models.
Note: fMRI support to be added in the future.
"""

import argparse
import logging
import matplotlib
matplotlib.use("Agg")
from nice.pylearn2.models.nice import NICE
import nipy
import numpy as np
from os import path

from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets.MRI import MRI
from pylearn2.neuroimaging_utils.datasets.MRI import MRI_Standard
from pylearn2.neuroimaging_utils.datasets.MRI import MRI_Transposed 
from pylearn2.neuroimaging_utils.tools import rois
from pylearn2.neuroimaging_utils.tools import nifti_viewer
from pylearn2.models.dbm import DBM
from pylearn2.models.vae import VAE
from pylearn2.utils import serial
from pylearn2.utils import sharedX

import sys


logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def get_nifti(dataset, features, out_file=None):
    """
    Function to get nifti image and save nifti files.

    Parameters
    ----------
    dataset: MRI class.
        A dataset of the MRI class for processing the nifti from. Must implement get_nifti.
    features: array-like.
        Features for nifti processing.
    out_file: str, optional.
        Output file for nifti image.

    Returns
    -------
    nifti: nipy image.
    """
    logger.info("Getting nifti for dataset of type %r and %d features."
                % (type(dataset), features.shape[0]))
    if not isinstance(dataset, MRI):
        raise ValueError("Dataset type is %r and not an instance of %r" % (type(dataset), MRI))
    weights_view = dataset.get_weights_view(features)
    nifti = dataset.get_nifti(weights_view)
    if out_file is not None:
        nipy.save_image(nifti, out_file)
    return nifti
    
def save_montage(nifti, nifti_file, out_file, anat_file=None):
    """
    Saves a montage from a nifti file.
    This will also process an region of interest dictionary (ROIdict)

    Parameters
    ----------
    nifti: nipy Image.
        Nifti file for processing.
    nifti_file: str
        Path to nifti file.
        Needed to process the roi dictionary.
    out_file: str
        Path to output file.
    anat_file: str, optional
        Path to anat file. If not provided,
        ${PYLEARN2_NI_PATH}/mri_extra/ch2better_aligned2EPI.nii is used.
    """
    logger.info("Saving montage from %s to %s." % (nifti_file, out_file))
    roi_dict = rois.main(nifti_file)
    if anat_file is None:
        anat_file = serial.preprocess(
            "${PYLEARN2_NI_PATH}/mri_extra/ch2better_aligned2EPI.nii")
    nifti_viewer.montage(nifti, anat_file, roi_dict, out_file=out_file)

def get_features(model, zscore=True, transposed_features=False, dataset=None):
    """
    Extracts the features given a number of model types.
    Included are special methods for VAE and NICE. Also if the data is transposed,
    the appropriate matrix multiplication of data x features is used.

    Parameters
    ----------
    model: pylearn2 Model class.
        Model from which to extract features.
    transposed_data: bool, optional.
        Whether the model was trained in transpose.
    dataset: pylearn2 Dataset class.
        Dataset to process transposed features.

    Returns
    -------
    features: array-like.
    """
    logger.info("Getting features%s for model of type %r%s."
                % (" (zscored)" if zscore else "",
                   type(model),
                   "(transposed data)" if transposed_features else ""))
    if isinstance(model, VAE):
        z = sharedX(np.eye(model.hid))
        theano = model.decode_theta(z)
        features = theta[0].eval()
    elif isinstance(model, NICE):
        spectrum = model.encoder.layers[-1].D.get_value()
        idx = np.argsort(spectrum).tolist()[::-1]
        num_features = len(idx)
        idx = idx[:100]
        z = np.zeros((len(idx), num_features))
        for i, j in enumerate(idx):
            z[i][j] = 1.
        Z = sharedX(z)
        features = model.encoder.inv_fprop(Z).eval()
    else:
        features = model.get_weights()
        weights_format = model.get_weights_format()
        assert hasattr(weights_format, '__iter__')
        assert len(weights_format) == 2
        assert weights_format[0] in ['v', 'h']
        assert weights_format[1] in ['v', 'h']
        assert weights_format[0] != weights_format[1]
        if weights_format[0] == 'v':
            features = features.T

    if transposed_features:
        if dataset is None:
            raise ValueError("Must provide a dataset to transpose features (None provided).")
        data = dataset.get_design_matrix()
        assert data.shape[1] == features.shape[1]
        features = features.dot(data.T)

    if zscore:
        features = (features - features.mean()) / features.std()

    return features

def main(model_path, out_path, args):
    """
    Main function of moduel.
    This function controls the high end analysis functions.

    Parameters
    ----------
    model_path: str
        Path for the model.
    out_path: str
        Path for the output directory.
    args: dict
        argparse arguments (defined below).
    """
    logger.info("Loading model from %s" % model_path)
    model = serial.load(model_path)
    logger.info("Extracting dataset")
    dataset = yaml_parse.load(model.dataset_yaml_src)
    if isinstance(dataset, MRI_Transposed):
        transposed_features = True
    else:
        transposed_features = False
    features = get_features(model, args.zscore, transposed_features, dataset)
    if args.prefix is None:
        prefix = ".".join(path.basename(model_path).split(".")[:-1])
    nifti_path = path.join(out_path, prefix + ".nii")
    pdf_path = path.join(out_path, prefix + ".pdf")
    nifti = get_nifti(dataset, features, out_file=nifti_path)
    save_montage(nifti, nifti_path, pdf_path)
    logger.info("Done.")

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", help="Path for the model .pkl file.")
    parser.add_argument("--out_dir", default=None, help="output path for the analysis files.")
    parser.add_argument("--prefix", default=None, help="Prefix for output files.")
    parser.add_argument("--zscore", default=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more verbosity!")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.out_dir is None:
        out_path = path.abspath(path.dirname(args.model_path))
    else:
        out_path = args.out_dir
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args.model_path, out_path, args)
