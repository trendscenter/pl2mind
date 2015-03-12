"""
Module for input handler.
"""

__author__ = "Devon Hjelm"
__copyright__ = "Copyright 2014, Mind Research Network"
__credits__ = ["Devon Hjelm"]
__licence__ = "3-clause BSD"
__email__ = "dhjelm@mrn.org"
__maintainer__ = "Devon Hjelm"

import numpy as np
from os import path
from pl2mind.datasets import MRI
from pl2mind.dataset_utils import mri_utils
from pylearn2.utils import serial


class MRIInputHandler(object):
    """
    Input handler for MRI data.
    This is an object in the case of loading multiple experiments with the same
    data parameters.
    TODO: clean up.
    """
    def __init__(self):
        self.d = {}

    def get_input_params(self, hyperparams):
        """
        Get the input parameters given data hyperparameters.

        Parameters
        ----------
        hyperparams: dict
            Hyperparameters.

        Returns
        -------
        input_dim, variance_map_file: int, str
            Input dimensionality and the location of the variance map file.
        """

        assert "dataset_name" in hyperparams.keys(), hyperparams
        dataset_name = hyperparams["dataset_name"]
        data_class = hyperparams["data_class"]
        variance_normalize = hyperparams.get("variance_normalize", False)
        unit_normalize = hyperparams.get("unit_normalize", False)
        demean = hyperparams.get("demean", False)
        even_input = False # Fix this
        assert not (variance_normalize and unit_normalize)

        data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + dataset_name)

        h = hash((data_class, variance_normalize, unit_normalize, demean))

        if self.d.get(h, False):
            return self.d[h]
        else:
            if data_class == "MRI_Transposed":
                assert not variance_normalize
                mri = MRI.MRI_Transposed(dataset_name=dataset_name,
                                         unit_normalize=unit_normalize,
                                         demean=demean,
                                         even_input=even_input,
                                         apply_mask=True)
                input_dim = mri.X.shape[1]
                variance_file_name = ("variance_map_transposed%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_dm" if demean else ""))

            elif data_class == "MRI_Standard":
                mask_file = path.join(data_path, "mask.npy")
                mask = np.load(mask_file)
                input_dim = (mask == 1).sum()
                #if input_dim % 2 == 1:
                #    input_dim -= 1
                mri = MRI.MRI_Standard(which_set="full",
                                       dataset_name=dataset_name,
                                       unit_normalize=unit_normalize,
                                       demean=demean,
                                       variance_normalize=variance_normalize,
                                       even_input=even_input,
                                       apply_mask=True)
                variance_file_name = ("variance_map%s%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_vn" if variance_normalize else "",
                                         "_dm" if demean else ""))
            else:
                raise ValueError("Unknown data class: %s" % data_class)

        variance_map_file = path.join(data_path, variance_file_name)
        if not path.isfile(variance_map_file):
            mri_utils.save_variance_map(mri, variance_map_file)
        self.d[h] = (input_dim, variance_map_file)
        return self.d[h]