#!/usr/bin/env python
"""
... todo::

    WRITEME
"""

import argparse
import logging
from nipy import load_image
from nipy import save_image
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
from pylearn2.neuroimaging_utils.tools import rois
from pylearn2.neuroimaging_utils.tools import nifti_viewer
from pylearn2.utils import serial
import sys
import warnings

logger = logging.getLogger(__name__)

def save_weights(model_path=None,
                 nifti_file=None,
                 pdf_file=None,
                 zscore=True,
                 level=0):
    """
    Function to save nifti files or just a pdf from a MRI model.
    """
    
    logger.info('making weights report')
    logger.info('loading model')
    model = serial.load(model_path)
    logger.info('loading done')
    assert model is not None

    control.push_load_data(False)
    dataset = yaml_parse.load(model.dataset_yaml_src)
    control.pop_load_data()
    logging.info('...done')

    W = model.get_weights2(level=level)
    #except TypeError:
    #    warnings.warn("Tried to pass a level to get_weights(), but not supported.")
    #    W = model.get_weights()

    weights_view = None

    weights_format = model.get_weights_format()
    assert hasattr(weights_format, '__iter__')
    assert len(weights_format) == 2
    assert weights_format[0] in ['v', 'h']
    assert weights_format[1] in ['v', 'h']
    assert weights_format[0] != weights_format[1]
    if weights_format[0] == 'v':
        W = W.T
    h = W.shape[0]

    weights_view = dataset.get_weights_view(W)
    assert weights_view.shape[0] == h
    if zscore:
        weights_view = (weights_view - weights_view.mean()) / weights_view.std()
    
    nifti = dataset.get_nifti(weights_view)
    
    if nifti_file:
        save_image(nifti, nifti_file)
    if pdf_file:
        assert nifti_file is not None, "If you want a pdf, you must have a nifti. How could you have your pdf if you don't have your nifti???"
        roi_dict = rois.main(nifti_file)
        anat_file = serial.preprocess("${PYLEARN2_NI_PATH}/mri_extra/ch2better_aligned2EPI.nii")
        nifti_viewer.montage(nifti, anat_file, roi_dict, out_file=pdf_file)

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--nifti", default=None, help="output file for the nifti file.")
    parser.add_argument("--pdf", default=None, help="output file for the pdf.")
    parser.add_argument("path", help="path for the model .pkl file")
    parser.add_argument("--level", default=0)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    save_weights(args.path, nifti_file=args.nifti, pdf_file=args.pdf, level=int(args.level))
