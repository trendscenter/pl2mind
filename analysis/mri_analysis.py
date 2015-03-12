"""
Module to perform analysis on MRI models.
Note: fMRI support to be added in the future.
"""

__author__ = "Devon Hjelm"
__copyright__ = "Copyright 2014, Mind Research Network"
__credits__ = ["Devon Hjelm"]
__licence__ = "3-clause BSD"
__email__ = "dhjelm@mrn.org"
__maintainer__ = "Devon Hjelm"

import argparse
import json
import logging
from math import log
from math import sqrt

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

import nipy
import numpy as np
import os
from os import path

from pl2mind import logger
from pl2mind.datasets import dataset_info
from pl2mind.datasets import MRI as MRI_module
from pl2mind.datasets.MRI import MRI
from pl2mind.datasets.MRI import MRI_Standard
from pl2mind.datasets.MRI import MRI_Transposed
from pl2mind.analysis import feature_extraction as fe
from pl2mind.tools import ols
from pl2mind.tools import rois
from pl2mind.tools import nifti_viewer

from pylearn2.config import yaml_parse
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.neuroimaging_utils.datasets.MRI import MRI as MRI_old
from pylearn2.utils import serial
from pylearn2.utils import sharedX

import sys
import theano.tensor as T


logger = logger.setup_custom_logger("pl2mind", logging.ERROR)

def set_experiment_info(model, dataset, feature_dict):
    if isinstance(dataset, TransformerDataset):
        transformer = dataset.transformer
        dataset = dataset.raw
        logger.info("Found transformer of type %s. Returning (TODO)"
                    % type(transformer))
        return
    else:
        transformer = None
    logger.info("Finding experiment related analysis for model of type %r and "
                "dataset %s" % (type(model), dataset.dataset_name))
    activations = get_activations(model, dataset)

    if dataset.dataset_name in dataset_info.sz_datasets:
        ttests = get_sz_info(dataset, activations)
        for feature in feature_dict:
            feature_dict[feature]["sz_t"] = ttests[
                feature_dict[feature]["real_id"]][0]

    if dataset.dataset_name in dataset_info.aod_datasets:
        target_ttests, novel_ttests = get_aod_info(dataset, activations)
        for feature in feature_dict:
            i = feature_dict[feature]["real_id"]

            feature_dict[feature]["tg_t"] = target_ttests[i][0]
            feature_dict[feature]["nv_t"] = novel_ttests[i][0]

def get_nifti(dataset, features, out_file=None, split_files=False,
              base_nifti=None):
    """
    Function to get nifti image and save nifti files.

    Parameters
    ----------
    dataset: MRI class.
        A dataset of the MRI class for processing the nifti from.
        Must implement get_nifti.
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
        raise ValueError("Dataset type is %r and not an instance of %r"
                         % (type(dataset), MRI))
    weights_view = dataset.get_weights_view(features)

    image = dataset.get_nifti(weights_view, base_nifti=base_nifti)
    if out_file is not None:
        nipy.save_image(image, out_file + ".gz")

    return image

def save_niftis(dataset, features, image_dir, base_nifti=None, **kwargs):
    """
    Saves a series of niftis.
    """
    logger.info("Saving mri images")
    spatial_maps = features.spatial_maps
    spatial_maps = dataset.get_weights_view(spatial_maps)
    for i, feature in features.f.iteritems():
        image = dataset.get_nifti(spatial_maps[i], base_nifti=base_nifti)
        nipy.save_image(image, path.join(image_dir, "%d.nii.gz" % feature.id))

    nifti_files = [path.join(image_dir, "%d.nii.gz" % feature.id)
                   for feature in features.f.values()]
    roi_dict = rois.main(nifti_files)

    anat_file = ("/export/mialab/users/mindgroup/Data/mrn/"
                 "mri_extra/ch2better_aligned2EPI.nii")
    anat = nipy.load_image(anat_file)
    nifti_viewer.save_images(nifti_files, anat, roi_dict, image_dir, **kwargs)

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

def compare_models(feature_dict):
    """
    Inter-model comparison.
    """

    feature_list = [(name, features)
        for name, features in feature_dict.iteritems()]

    for i in range(len(feature_list)):
        for j in range(i + 1, len(feature_list)):
            logger.info("Analyzing %s compared to %s" % (feature_list[i][0],
                                                         feature_list[j][0]))
            indices = fe.match_parameters(feature_list[i][1].spatial_maps,
                                          feature_list[j][1].spatial_maps)
            for pi, qi in indices:
                feature_list[i][1].f[pi].match_indices[feature_list[j][0]] = qi

def main(model, out_path=None, prefix=None, **anal_args):
    """
    Main function of module.
    This function controls the high end analysis functions.

    Parameters
    ----------
    model: Pylearn2.Model or str
        Model instance or path for the model.
    out_path: str, optional
        Path for the output directory.
    prefix: str, optional
        If provided, prefix for all output files.
    dataset_root: str, optional
        If provided, use as the root dir for dataset extraction.
    anal_args: dict
        argparse arguments (defined below).
    """

    if out_path is None and prefix is None and isinstance(model, str):
        prefix = ".".join(path.basename(model).split(".")[:-1])
        sm_prefix = prefix
        nifti_prefix = prefix
    else:
        nifti_prefix = "image"

    if out_path is None:
        assert isinstance(model, str)
        out_path = path.abspath(path.dirname(model))

    if isinstance(model, str):
        logger.info("Loading model from %s" % model)
        model = serial.load(model)

    if not path.isdir(out_path):
        os.mkdir(out_path)

    logger.info("Getting features")
    feature_dict = fe.extract_features(model, **anal_args)

    logger.info("Getting features")
    feature_dict = fe.extract_features(model, **anal_args)
    dataset = fe.resolve_dataset(model, **anal_args)
    if isinstance(dataset, TransformerDataset):
        dataset = dataset.raw

    anal_dict = dict()

    compare_models(feature_dict)
    for name, features in feature_dict.iteritems():
        image_dir = path.join(out_path, "%s_images" % name)
        if not path.isdir(image_dir):
            os.mkdir(image_dir)
        save_niftis(dataset, features, image_dir, **anal_args)

        features.set_histograms(tolist=True)
        fds = dict()
        for k, f in features.f.iteritems():
            fd = dict(
                image=path.join("%s_images" % name, "%d.png" % f.id),
                image_type="mri",
                index=f.id,
                hists=f.hists,
                match_indices=f.match_indices
            )
            fd.update(**f.stats)

            fds[k] = fd

        anal_dict[name] = dict(
            name=name,
            image_dir=image_dir,
            features=fds
        )

    ms = fe.ModelStructure(model, dataset)
    #anal_dict["graph"] = nx.node_link_data(fe.get_nx())

    json_file = path.join(out_path, "analysis.json")
    with open(json_file, "w") as f:
        json.dump(anal_dict, f)

    logger.info("Analysis done")

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("model_path", help="Path for the model .pkl file.")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show more verbosity!")
    parser.add_argument("-o", "--out_dir", default=None,
                        help="output path for the analysis files.")
    parser.add_argument("-p", "--prefix", default=None,
                        help="Prefix for output files.")

    parser.add_argument("-z", "--zscore", action="store_true")
    parser.add_argument("-t", "--target_stat", default=None)
    parser.add_argument("-i", "--image_threshold", default=2)
    parser.add_argument("-m", "--max_features", default=100)
    parser.add_argument("-d", "--dataset_root", default=None,
                        help="If specified, use another user's data root")
    parser.add_argument("-b", "--base_nifti", default=None)
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    anal_args = dict(
        zscore=args.zscore,
        target_stat=args.target_stat,
        max_features=args.max_features,
        dataset_root=args.dataset_root,
        base_nifti=args.base_nifti,
        image_threshold=args.image_threshold
    )

    main(args.model_path, args.out_dir, args.prefix, **anal_args)
