"""
Module for analysis of schizophrenia experiments.
"""

import logging
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind


logger = logging.getLogger("pl2mind")

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