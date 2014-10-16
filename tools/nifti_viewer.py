import argparse
from math import ceil

import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
from matplotlib import rc

from nipy import load_image
from nipy.core.api import xyz_affine
from nipy.labs.viz import plot_map

import numpy as np
from sys import stdout
import pickle


cdict = {'red': ((0.0, 0.0, 0.0),
                 (0.25, 0.2, 0.2),
                 (0.45, 0.0, 0.0),
                 (0.5, 0.5, 0.5),
                 (0.55, 0.0, 0.0),
                 (0.75, 0.8, 0.8),
                 (1.0,  1.0, 1.0)),
         'green': ((0.0, 0.0, 1.0),
                   (0.25, 0.0, 0.0),
                   (0.45, 0.0, 0.0),
                   (0.5, 0.5, 0.5),
                   (0.55, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0,  1.0, 1.0)),
         'blue':  ((0.0, 0.0, 1.0),
                   (0.25, 0.8, 0.8),
                   (0.45, 0.0, 0.0),
                   (0.5, 0.5, 0.5),
                   (0.55, 0.0, 0.0),
                   (0.75, 0.0, 0.0),
                   (1.0,  0.0, 0.0)),}

cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

def montage(nifti, anat, roi_dict,
            thr=2, fig=None, out_file=None):
    assert nifti is not None
    assert anat is not None
    assert roi_dict is not None

    texcol = 1
    bgcol = 0
    iscale=2
    weights = nifti.get_data(); weights /= weights.std()
    features = weights.shape[-1]
    assert features == len(roi_dict)

    indices = [0]        
    y = 8
    x = int(ceil(1.0 * features / y))

    font = {'size':8}
    rc('font',**font)
 
    if fig is None:
        fig = plt.figure(figsize=[iscale * y, iscale * x / 2.5])
    plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.1, hspace=0)
   
    for f in xrange(features):
        stdout.write("\rSaving montage: %d   " % f)
        stdout.flush()
            
        feat = weights[:, :, :, f]
        imax = np.max(np.absolute(feat)); imin = -imax
        imshow_args = {'vmax': imax, 'vmin': imin}
         
        coords = roi_dict[f]["top_clust"]["coords"]
        assert coords is not None

        # For some reason we need to do this.
        coords = ([-coords[0], -coords[1], coords[2]])

        ax = fig.add_subplot(x, y, f + 1)
        plt.axis('off')

        max_idx = np.unravel_index(np.argmax(feat), feat.shape)

        try: plot_map(feat,
                      xyz_affine(nifti),
                      anat=anat.get_data(),
                      anat_affine=xyz_affine(anat), 
                      threshold=thr,
                      figure=fig,
                      axes=ax,
                      cut_coords=coords,
                      annotate=False,
                      cmap=cmap, 
                      draw_cross=False,
                      **imshow_args)            
        except Exception as e:
            print e

        plt.text(0.05, 0.8, str(f),
                 transform=ax.transAxes,
                 horizontalalignment='center',
                 color=(texcol,texcol,texcol))
    
    stdout.write("\rSaving montage: DONE\n")
    if out_file is not None:
        plt.savefig(out_file, transparent=True, facecolor=(bgcol, bgcol, bgcol))
    else:
        plt.draw()

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("nifti", help="Nifti file to be processed.")
    parser.add_argument("--anat", default=None, help="Anat file for montage.")
    parser.add_argument("--rois", default=None, help="Pickled roi file.")
    parser.add_argument("--out", default=None, help="Output of montage.")
    parser.add_argument("--thr", default=2, help="Threshold for features.")
    return parser

def main(nifti_file, anat_file, roi_file, out_file, thr):
    iscale = 2
    nifti = load_image(nifti_file)
    anat = load_image(anat_file)
    roi_dict = pickle.load(open(roi_file, "rb"))
    montage(nifti, anat, roi_dict, out_file=out_file)

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    main(args.nifti, args.anat, args.rois, args.out, args.thr)

