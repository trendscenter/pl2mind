#!/usr/bin/env python
"""
... todo::

    WRITEME
"""

import argparse
import logging
from nipy import save_image
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import control
import sys

logger = logging.getLogger(__name__)

def save_weights(model_path=None,
                       out=None):
    """
    ... todo::

        WRITEME
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

    W = model.get_weights()

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
    
    nifti = dataset.get_nifti(weights_view)
    
    if out:
        save_image(nifti, out)

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default=None)
    parser.add_argument("path")
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    save_weights(args.path, args.out)
