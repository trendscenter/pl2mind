"""
Module to train a simple MLP for demo.
"""

from jobman.tools import expand
from jobman.tools import flatten

import logging
import nice_experiment
import numpy as np
from os import path
from pylearn2.config import yaml_parse
from pl2mind.datasets import MRI
from pl2mind.dataset_utils import mri_nifti
from pylearn2.scripts.jobman.experiment import ydict
from pylearn2.utils import serial

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

yaml_file = nice_experiment.yaml_file

def main(args):
    logger.info("Getting dataset info for %s" % args.dataset_name)
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + args.dataset_name)
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = (mask == 1).sum()
    if input_dim % 2 == 1:
        input_dim -= 1
    mri = MRI.MRI_Standard(which_set="full",
                           dataset_name=args.dataset_name,
                           unit_normalize=True,
                           even_input=True,
                           apply_mask=True)
    variance_map_file = path.join(data_path, "variance_map.npy")
    mri_nifti.save_variance_map(mri, variance_map_file)

    user = path.expandvars("$USER")
    save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/%s"
                                  % (user, "nice_jobman_test"))

    file_params = {"save_path": save_path,
                   "variance_map_file": variance_map_file
                   }

    yaml_template = open(yaml_file).read()
    hyperparams = expand(flatten(nice_experiment.default_hyperparams(input_dim=input_dim)),
                         dict_type=ydict)

    for param in file_params:
        yaml_template = yaml_template.replace("%%(%s)s" % param, file_params[param])

    yaml = yaml_template % hyperparams
    print yaml

    logger.info("Training")
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == "__main__":
    parser = nice_experiment.make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
