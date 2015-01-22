"""
Module to train a simple MLP for demo.
"""

from jobman.tools import expand
from jobman.tools import flatten

import logging
import rbm_experiment as experiment
import numpy as np
from os import path
from pylearn2.config import yaml_parse
from pylearn2.neuroimaging_utils.datasets import MRI
from pylearn2.neuroimaging_utils.dataset_utils import mri_nifti
from pylearn2.scripts.jobman.experiment import ydict
from pylearn2.utils import serial

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

yaml_file = experiment.yaml_file

def main(args):
    dataset_name = args.dataset_name

    logger.info("Getting dataset info for %s" % dataset_name)
    data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + dataset_name)
    mask_file = path.join(data_path, "mask.npy")
    mask = np.load(mask_file)
    input_dim = (mask == 1).sum()

    user = path.expandvars("$USER")
    save_path = serial.preprocess("/export/mialab/users/%s/pylearn2_outs/%s"
                                  % (user, "rbm_simple_test"))

    # File parameters are path specific ones (not model specific).
    file_params = {"save_path": save_path,
                   }

    yaml_template = open(yaml_file).read()
    hyperparams = expand(flatten(experiment.default_hyperparams(input_dim=input_dim)),
                         dict_type=ydict)

    # Set additional hyperparams from command line args
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate
    if args.batch_size is not None:
        hyperparams["batch_size"] = args.batch_size

    for param in file_params:
        yaml_template = yaml_template.replace("%%(%s)s" % param, file_params[param])

    yaml = yaml_template % hyperparams

    logger.info("Training")
    train = yaml_parse.load(yaml)
    train.main_loop()

if __name__ == "__main__":
    parser = experiment.make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    main(args)
