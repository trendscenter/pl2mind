"""
Module from Alvaro's ICA
"""

import numpy as np
from pl2mind.logger import logger

def PCAwhiten(X, Ncomp, **kwargs):
    """
    Whitens X with Ncomp components.
    """
    eps = kwargs["eps"]

    u, s, v = np.linalg.svd(X, full_matrices=False)
    logger.info("%.2f%% retained variance"
                % (100 * (sum(s[:Ncomp]) / sum(s))))
    u = u[:,:Ncomp]
    v = v[:Ncomp,:]
    s = s[:Ncomp]
    white = np.dot(np.diag((s + eps)**(-1)), u.T)
    dewhite = np.dot(u, np.diag(s))
    Xwhite = v
    return (Xwhite, white, dewhite)

def indUpdate(W1, Xwhite1, bias1, lrate1, startW1, **kwargs):
    max_w = kwargs["max_w"]
    anneal = kwargs["anneal"]

    assert len(Xwhite1.shape) == 2
    Ncomp1, Nvox1 = Xwhite1.shape
    block1 = int(np.floor(np.sqrt(Nvox1 / 3)))
    Ib1 = np.ones((1, block1))
    I1 = np.eye(Ncomp1)
    error1 = 0
    permute1 = np.random.permutation(Nvox1)
    for tt in range(0, Nvox1, block1):
        if(tt + block1 < Nvox1):
            tt2 = tt + block1
        else:
            tt2 = Nvox1
            block1 = Nvox1 - tt

        U1 = np.dot(W1,Xwhite1[:, permute1[tt:tt2]]) +\
             np.dot(bias1, Ib1[:, 0:block1])
        Y1 = 1 / (1 + np.exp(-U1))
        W1 = W1 + lrate1 * np.dot(block1 * I1 + \
                                  np.dot(1 - 2 * Y1, U1.T), W1)
        bias1 = (bias1.T + lrate1 * (1 - 2 * Y1).sum(axis=1)).T
        # Checking if W blows up
        if np.isnan(np.sum(W1)) or np.max(np.abs(W1)) > max_w:
            logger.info("Numeric error! restarting with lower learning rate")
            error1 = 1
            lrate1 = lrate1 * anneal
            W1 = startW1
            bias1 = np.zeros((Ncomp1, 1))

            if (lrate1 > 1e-6) and (np.linalg.matrix_rank(Xwhite1) < Ncomp1):
                logger.info("Data 1 is rank defficient. I cannot compute "
                            "%d components." % Ncomp1)
                return (np.nan)

            if lrate1 < 1e-6:
                logger.info("Weight matrix may not be invertible...")
                return (np.nan)
            break

    return(W1, bias1, error1, lrate1)

def infomax(Xwhite, **kwargs):
    """
    Computes ICA infomax in whitened data.
    """

    eps = kwargs["eps"]
    max_step = kwargs["max_step"]
    anneal = kwargs["anneal"]
    w_stop = kwargs["w_stop"]

    Ncomp = Xwhite.shape[0]
    # Initialization
    W = np.eye(Ncomp)
    startW = W
    oldW = startW
    lrate = 0.005 / np.log(Ncomp)
    bias = np.zeros((Ncomp,1))
    logger.info("Beginning ICA training...")
    step = 1

    while (step < max_step):
        # Shuffle variable order at each step
        (W, bias, error, lrate) = indUpdate(W, Xwhite, bias, lrate, startW,
                                            **kwargs)
        if error == 0:

            wtchange = W - oldW
            oldW = W
            change = np.sum(wtchange**2)

            if step == 1:    # initializing variables
                oldwtchange = wtchange
                oldchange = change

            if step > 2:
                angleDelta = np.arccos(np.sum(oldwtchange * wtchange) /\
                                       (np.sqrt(change * oldchange) + eps))
                angleDelta = angleDelta*180/np.pi
                if angleDelta >60:
                    lrate = lrate * anneal
                    oldwtchange = wtchange
                    oldchange = change
                if (step % 10 == 0) or change < w_stop:
                    logger.info("Step %d: Lrate %.1e, "
                                "Wchange %.1e, "
                                "Angle %.2f" %
                                (step, lrate, change, angleDelta))

            # Stopping rule
            if (step > 2) and (change < w_stop):
                step = max_step

            step += 1
        else:
            step = 1

    A = np.linalg.pinv(W)
    S = np.dot(W, Xwhite)

    return (A, S, W)

def ica(X, Ncomp, **kwargs):
    defaults = dict(
        eps = 1e-18,
        max_w = 1e8,
        anneal = 0.9,
        max_step = 500,
        min_lrate = 1e-6,
        w_stop = 1e-6
        )
    assert len(set(kwargs.keys()) - set(defaults.keys())) == 0, "Bad kw passed"

    defaults.update(kwargs)

    logger.info("Whitening data...")
    Xwhite, white, dewhite  = PCAwhiten(X, Ncomp, **defaults)
    logger.info("Done.")
    logger.info("Running INFOMAX-ICA ...")
    A, S, W = infomax(Xwhite, **defaults)
    A =  np.dot(dewhite, A)
    logger.info("Done.")
    return (A, S)