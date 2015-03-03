"""
Module for simtb analysis
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import json
import logging
from matplotlib import pyplot as plt
import os
from os import path

from pl2mind.analysis import feature_extraction as fe
from pl2mind import logger
from pl2mind.tools import simtb_viewer
from pylearn2.datasets.transformer_dataset import TransformerDataset
from pylearn2.utils import serial


logger = logger.setup_custom_logger("pl2mind", logging.ERROR)

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

def save_simtb_spatial_maps(dataset, features, out_path):
    """
    Saves a series of simtb images.
    """
    logger.info("Saving simtb images")
    spatial_maps = features.spatial_maps
    spatial_maps = dataset.get_weights_view(spatial_maps)
    for i, feature in features.f.iteritems():
        spatial_map = spatial_maps[i]
        simtb_viewer.make_spatial_map_image(
            spatial_map, out_file=path.join(out_path, "%d.png" % i))

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
    dataset = fe.resolve_dataset(model, **anal_args)
    if isinstance(dataset, TransformerDataset):
        dataset = dataset.raw

    for name, features in feature_dict.iteritems():
        json_file = path.join(out_path, "%s_analysis.json" % name)
        image_dir = path.join(out_path, "%s_images" % name)
        if not path.isdir(image_dir):
            os.mkdir(image_dir)

        with open(json_file, "w") as f:
            json.dump(dict(
                name=name,
                image_dir=image_dir,
                spatial_maps=features.spatial_maps.tolist(),
                activations=features.activations.tolist(),
                feature_stats=dict(
                    (k, features[k].stats) for k in features.f.keys()
                )
            ), f)
        save_simtb_spatial_maps(dataset, features, image_dir)

    logger.info("Done.")

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
    parser.add_argument("-m", "--max_features", default=100)
    parser.add_argument("-d", "--dataset_root", default=None,
                        help="If specified, use another user's data root")
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
    )

    main(args.model_path, args.out_dir, args.prefix, **anal_args)