"""
Module for analysis of AOD experiments.
"""

import logging
import numpy as np
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind


logger = logging.getLogger("pl2mind")

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