"""
Module to perform analysis on MRI models.
Note: fMRI support to be added in the future.
"""

import argparse
import logging
import matplotlib
matplotlib.use("Agg")
import nipy
import numpy as np
import os
from os import path
from matplotlib import pyplot as plt
from math import log
from math import sqrt

from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets import dataset_info
from pylearn2.neuroimaging_utils.datasets.MRI import MRI
from pylearn2.neuroimaging_utils.datasets.MRI import MRI_Standard
from pylearn2.neuroimaging_utils.datasets.MRI import MRI_Transposed
from pylearn2.neuroimaging_utils.tools import ols
from pylearn2.neuroimaging_utils.tools import rois
from pylearn2.neuroimaging_utils.tools import nifti_viewer
from pylearn2.neuroimaging_utils.tools import simtb_viewer
from pylearn2.models.dbm import DBM
from pylearn2.models.vae import VAE
from pylearn2.utils import serial
from pylearn2.utils import sharedX

from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import sys
import warnings


logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

try:
    from nice.pylearn2.models.nice import NICE
except ImportError:
    class NICE (object):
        pass
    logger.warn("NICE not found, so hopefully you're not trying to load a NICE model.")

def get_activations(model, dataset):
    """
    Get latent variable activations given a dataset.

    Parameters
    ----------
    model: pylearn2.Model
        Model from which to get activations.
    dataset: pylearn2.datasets.DenseDesignMatrix
        Dataset from which to generate activations.

    Returns
    -------
    activations: numpy array-like
    """

    logger.info("Getting activations for model of type %s and model %s"
                % (type(model), dataset.dataset_name))
    data = dataset.get_design_matrix()
    if isinstance(model, NICE):
        if isinstance(dataset, MRI_Transposed):
            S = model.encoder.layers[-1].D.get_value()
            sigma = np.exp(-S)
            num_features = model.nvis
            y = np.zeros((1, num_features))
            Y = sharedX(y)
            mean_activations = model.encoder.inv_fprop(Y).eval()
            z = np.zeros((num_features, num_features))
            for i, j in enumerate(range(num_features)):
                z[i, j] = 2 * sigma[j]
            Z = sharedX(z)
            activations = (model.encoder.inv_fprop(Z).eval() - mean_activations)            
        else:
            X = sharedX(data)
            activations = model.encode(X).eval()
    elif isinstance(model, VAE):
        X = sharedX(data)
        epsilon = model.sample_from_epsilon((X.shape[0], model.nhid))
        epsilon *= 0
        phi = model.encode_phi(X)
        activations = model.sample_from_q_z_given_x(epsilon=epsilon, phi=phi).eval()
        assert activations.shape[1] == model.nhid
    else:
        raise NotImplementedError("Cannot get activations for model of type %r. "
                                  "Needs to be implemented"
                                  % type(model))

    return activations

def get_sz_info(dataset, activations):
    """
    Get schizophrenia classification experiment related info from activations.
    Info is a 2-sided t test for each latent variable of healthy vs control.

    Parameters
    ----------
    dataset: pylearn2.datasets.DenseDesignMatrix
        Dataset must be in dataset_info.sz_datasets.
        Labels must be in {0, 1}. Singleton labels not tested ({0}) and will
        likely not work.
    activations: numpy array_like
        Activations from which to ttest sz statistics.

    Returns
    -------
    ttests: list of tuples
        The 2-sided ttest (t, p) for each latent variable.
    """

    if dataset.dataset_name not in dataset_info.sz_datasets:
        raise ValueError("Dataset %s not designated as sz classification,"
                         "please edit \"datasets/dataset_info.py\""
                         "if you are sure this is an sz classification related"
                         "dataset" % dataset.dataset_name)
    logger.info("t testing features for relevance to Sz.")
    labels = dataset.y
    assert labels is not None
    for label in labels:
        assert label == 0 or label == 1
    sz_idx = [i for i in range(len(labels)) if labels[i] == 1]
    h_idx = [i for i in range(len(labels)) if labels[i] == 0]

    sz_acts = activations[sz_idx]
    h_acts = activations[h_idx]

    ttests = []
    for sz_act, h_act in zip(sz_acts.T, h_acts.T):
        ttests.append(ttest_ind(h_act, sz_act))

    return ttests

def get_aod_info(dataset, activations):
    """
    Get AOD task experiment related info from activations.
    Info is multiple regression of latent variable activations between
    target and novel stimulus.

    Parameters
    ----------
    dataset: pylearn2.datasets.DenseDesignMatrix
        Dataset must be in dataset_info.aod_datasets.
    activations: numpy array_like
        Activations from which to do multiple regression.

    Returns
    -------
    target_ttests, novel_ttests: lists of tuples
    """

    if dataset.dataset_name not in dataset_info.aod_datasets:
        raise ValueError("Dataset %s not designated as AOD task,"
                         "please edit \"datasets/dataset_info.py\""
                         "if you are sure this is an AOD task related"
                         "dataset" % dataset.dataset_name)
    logger.info("t testing features for relevance to AOD task.")
    targets = dataset.targets
    novels = dataset.novels
    dt = targets.shape[0]
    assert targets.shape == novels.shape
    assert dataset.X.shape[0] % dt == 0
    num_subjects = dataset.X.shape[0] // dt

    targets_novels = np.zeros([targets.shape[0], 2])
    targets_novels[:, 0] = targets
    targets_novels[:, 1] = novels
        
    target_ttests = []
    novel_ttests = []
    for i in xrange(activations.shape[1]):
        betas = np.zeros((num_subjects, 2))
        for s in range(num_subjects):
            act = activations[dt * s : dt * (s + 1), i]
            stats = ols.ols(act, targets_novels)
            betas[s] = stats.b[1:]
        target_ttests.append(ttest_1samp(betas[:, 0], 0))
        novel_ttests.append(ttest_1samp(betas[:, 1], 0))

    return target_ttests, novel_ttests

def set_experiment_info(model, dataset, feature_dict):
    logger.info("Finding experiment related analysis for model of type %r and dataset %s"
                % (type(model), dataset.dataset_name))
    activations = get_activations(model, dataset)    

    if dataset.dataset_name in dataset_info.sz_datasets:
        ttests = get_sz_info(dataset, activations)
        for feature in feature_dict:
            feature_dict[feature]["sz_t"] = ttests[feature_dict[feature]["real_id"]][0]

    if dataset.dataset_name in dataset_info.aod_datasets:
        target_ttests, novel_ttests = get_aod_info(dataset, activations)
        for feature in feature_dict:
            i = feature_dict[feature]["real_id"]

            feature_dict[feature]["tg_t"] = target_ttests[i][0]
            feature_dict[feature]["nv_t"] = novel_ttests[i][0]

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

def save_simtb_montage(dataset, features, out_file, feature_dict,
                       target_stat=None, target_value=None):
    """
    Saves a simtb montage.
    """

    logger.info("Saving simtb montage")
    weights_view = dataset.get_weights_view(features)
    simtb_viewer.montage(weights_view, out_file=out_file,
                         feature_dict=feature_dict,
                         target_stat=target_stat,
                         target_value=target_value)

def save_nii_montage(nifti, nifti_file, out_file,
                     anat_file=None, feature_dict=None,
                     target_stat=None, target_value=None):
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
    logger.info("Saving montage from %s to %s."
                % (nifti_file, out_file))
    roi_dict = rois.main(nifti_file)
    if anat_file is None:
        anat_file = serial.preprocess(
            "${PYLEARN2_NI_PATH}/mri_extra/ch2better_aligned2EPI.nii")
    nifti_viewer.montage(nifti, anat_file,
                         roi_dict,
                         out_file=out_file,
                         feature_dict=feature_dict,
                         target_stat=target_stat,
                         target_value=target_value)

def load_feature_dict(feature_dict, stat, stat_name):
    for i in range(len(feature_dict)):
        feature_dict[i][stat_name] = stat[feature_dict[i]["real_id"]]

    return feature_dict

def get_features(model, zscore=True, transposed_features=False,
                 dataset=None, feature_dict=None, max_features=100):
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
        if transposed_features:
            raise NotImplementedError()
        Y = sharedX(np.zeros((model.nhid, model.nhid)))
        theta_mean = model.decode_theta(Y)
        mean_features = model.means_from_theta(theta_mean).eval()

        Z = sharedX(np.eye(model.nhid) * 10)
        theta = model.decode_theta(Z)
        features = model.means_from_theta(theta).eval() - mean_features
    elif isinstance(model, NICE):
        logger.info("NICE layers: %r" % model.encoder.layers)
        S = model.encoder.layers[-1].D.get_value()
        sigma = np.exp(-S)
        idx = np.argsort(S).tolist()
        num_features = len(idx)
        idx = idx[:max_features]

        if feature_dict is not None:
            assert feature_dict == {}
            for i, j in enumerate(idx):
                feature_dict[i] = {"real_id": j}
            load_feature_dict(feature_dict, sigma, "s")

        if transposed_features:
            if dataset is None:
                raise ValueError("Must provide a dataset to transpose features (None provided)")
            data = dataset.get_design_matrix()
            X = sharedX(data)
            features = model.encode(X).T.eval()[idx]
            assert features.shape[0] == len(idx), features.shape
        else:
            y = np.zeros((1, num_features))
            Y = sharedX(y)
            mean_features = model.encoder.inv_fprop(Y).eval()
            z = np.zeros((len(idx), num_features))
            for i, j in enumerate(idx):
                z[i, j] = 2 * sigma[j]
            Z = sharedX(z)
            features = (model.encoder.inv_fprop(Z).eval() - mean_features)
    else:
        if transposed_features:
            raise NotImplementedError()
        features = model.get_weights()
        weights_format = model.get_weights_format()
        assert hasattr(weights_format, '__iter__')
        assert len(weights_format) == 2
        assert weights_format[0] in ['v', 'h']
        assert weights_format[1] in ['v', 'h']
        assert weights_format[0] != weights_format[1]
        if weights_format[0] == 'v':
            features = features.T

    if (features > features.mean() + 10 * features.std()).sum() > 1:
        logger.warn("Founds some spurious voxels. Don't know why they exist, but setting to 0.")
        features[features > features.mean() + 10 * features.std()] = 0

    return features

def nice_spectrum(model):
    """
    Generates the NICE spectrum from a NICE model.
    """

    logger.info("Getting NICE spectrum")
    if not isinstance(model, NICE):
        raise NotImplementedError("No spectrum analysis available for %r" % type(model))

    spectrum = model.encoder.layers[-1].D.get_value()
    spectrum = np.sort(spectrum)
    spectrum = np.exp(-spectrum)
    return spectrum

def resolve_dataset(model):
    """
    Resolves the full dataset from the model.
    """

    logger.info("Resolving full dataset from training set.")
    dataset = yaml_parse.load(model.dataset_yaml_src)
    if isinstance(dataset, MRI_Standard):
        dataset = MRI_Standard(
            which_set = "full",
            even_input=dataset.even_input,
            center=dataset.center,
            variance_normalize=dataset.variance_normalize,
            unit_normalize=dataset.unit_normalize,
            apply_mask=dataset.apply_mask,
            dataset_name=dataset.dataset_name,
            )
    return dataset

def main(model_path, out_path, args):
    """
    Main function of module.
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
    if args.prefix is None:
        prefix = ".".join(path.basename(model_path).split(".")[:-1])
    out_path = path.join(out_path, prefix)
    if not path.isdir(out_path):
        os.mkdir(out_path)

    logger.info("Loading model from %s" % model_path)
    model = serial.load(model_path)
    logger.info("Extracting dataset")
    dataset = resolve_dataset(model)
    if isinstance(dataset, MRI_Transposed):
        transposed_features = True
    else:
        transposed_features = False

    feature_dict = {}

    if isinstance(model, NICE):
        spectrum_path = path.join(out_path, prefix + "_spec.pdf")
        f = plt.figure()
        spectrum = nice_spectrum(model)
        plt.plot(spectrum)
        f.savefig(spectrum_path)

    logger.info("Getting features")
    features = get_features(model, args.zscore, transposed_features,
                            dataset, feature_dict=feature_dict)
    set_experiment_info(model, dataset, feature_dict)
    

    pdf_path = path.join(out_path, prefix + ".pdf")
    if dataset.dataset_name in dataset_info.simtb_datasets:
        save_simtb_montage(dataset, features, pdf_path,
                           feature_dict=feature_dict,
                           target_stat=args.target_stat,
                           target_value=2.)
    else:
        nifti_path = path.join(out_path, prefix + ".nii")
        nifti = get_nifti(dataset, features, out_file=nifti_path)
        save_nii_montage(nifti, nifti_path,
                         pdf_path, feature_dict=feature_dict,
                         target_stat=args.target_stat,
                         target_value=2.)
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
    parser.add_argument("--zscore", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more verbosity!")
    parser.add_argument("--target_stat", default="sz_t")
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
