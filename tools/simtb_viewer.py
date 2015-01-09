"""
Module to view models trained with simTB data.
"""

from math import ceil

import logging
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm
from matplotlib.patches import FancyBboxPatch
from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np

logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def montage(weights, fig=None, out_file=None,
            feature_dict=None, target_stat=None, target_value=None):
    features = weights.shape[0]
    iscale = 1
    y = 8
    x = int(ceil(1.0 * features / y))

    font = {'size':8}
    rc('font',**font)
    if fig is None:
        fig = plt.figure(figsize=[iscale * y, iscale * x])
        plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01,
                            top=0.99, wspace=0.1, hspace=0)

    for f in xrange(features):
        logger.debug("Saving simtb montage %d" % f)
        feat = weights[f]
        assert feat.shape[2] == 1, feat.shape
        feat = feat.reshape(feat.shape[0],feat.shape[1])
        feat = feat / feat.std()
        imax = np.max(np.absolute(feat)); imin = -imax
        imshow_args = {'vmax': imax, 'vmin': imin}
        ax = fig.add_subplot(x, y, f + 1)
        plt.axis("off")
        ax.imshow(feat, cmap=cm.RdBu_r, **imshow_args)

        plt.text(0.05, 0.8, str(f),
                 transform=ax.transAxes,
                 horizontalalignment='center',
                 color="white")
        
        pos = [(0.05, 0.05), (0.4, 0.05), (0.8, 0.05)]
        colors = ["purple", "yellow", "green"]
        
        if (feature_dict is not None and
            feature_dict.get(f, None) is not None):
            d = feature_dict[f]
            for i, key in enumerate([k for k in d if k != "real_id"]):
                plt.text(pos[i][0], pos[i][1], "%s=%.2f"
                         % (key, d[key]) ,transform=ax.transAxes,
                         horizontalalignment="left", color=colors[i])
                if key == target_stat:
                    assert target_value is not None
                    if d[key] >= target_value:
                        p_fancy = FancyBboxPatch((0.1, 0.1), 2.5 - .1, 1 - .1,
                                                 boxstyle="round,pad=0.1",
                                                 ec=(1., 0.5, 1.),
                                                 fc="none")
                        ax.add_patch(p_fancy)
                    elif d[key] <= -target_value:
                        p_fancy = FancyBboxPatch((0.1, 0.1), iscale * 2.5 - .1, iscale - .1,
                                                 boxstyle="round,pad=0.1",
                                                 ec=(0., 0.5, 0.),
                                                 fc="none")
                        ax.add_patch(p_fancy)
    
    logger.info("Finished processing simtb montage")
    if out_file is not None:
        logger.info("Saving montage to %s" % out_file)
        plt.savefig(out_file)
    else:
        plt.draw()
