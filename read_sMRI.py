"""
Utility for loading sMRI data.
"""

__author__ = "Devon Hjelm"
__copyright__ = "Copyright 2014, Mind Research Network"
__credits__ = ["Devon Hjelm"]
__licence__ = "3-clause BSD"
__email__ = "dhjelm@mrn.org"
__maintainer__ = "Devon Hjelm"

import numpy as np
from nipy import save_image, load_image

class open_niftis(object):
    """
    ... todo::
    
    WRITEME

    """
    def __init__(self, d, h_pattern=None, sz_pattern=None):
        self.d = d
        self.h_pattern = h_pattern if h_pattern else "H*"
        self.sz_pattern = sz_pattern if sz_pattern else "S*"

def read_niftis():
    """
    Read niftis.
    """     
    nifti = load_image(file_name)
    subject_data = nifti.get_data()


        stdout.write("\rLoading subjects: DONE\n")
        return data
