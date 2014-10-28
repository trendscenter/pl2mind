"""
Module to take output of MLP and embed it into a 2d space.
"""

import argparse
from DivideAndConcur import divide_and_concur as dc
from matplotlib import pylab as plt
import numpy as np

from pylearn2.config import yaml_parse
from pylearn2.datasets import control
from pylearn2.utils import serial

from theano import function


def main(model_path, k, out=None):
    model = serial.load(model_path)

    dataset = yaml_parse.load(model.dataset_yaml_src)

    X = model.get_input_space().make_theano_batch()
    Y = model.fprop(X)

    predictor = function([X], Y)
    y = predictor(dataset.raw.X[:1000])
    yhat = dataset.raw.y.flatten()[:1000]
    embeddings = dc.d_and_c(y, K=k)

    classes = np.amax(yhat) + 1
    base_colors = ['r', 'b', 'y', 'c', 'g', 'purple',
                   'orange', 'm', 'black', 'silver']
    f = plt.figure()
    scatters = []
    for c in range(classes):
        idx = np.where(yhat == c)[0].tolist()
        scatters.append(plt.scatter(embeddings[idx, 0],
                                    embeddings[idx, 1],
                                    color=base_colors[c]))
    plt.legend(scatters, range(classes))
    if out is not None:
        f.savefig(out)
    else:
        plt.show()


def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--k", default=10, help="k for knn.")
    parser.add_argument("--out", default=None, help="output file for the pdf.")
    parser.add_argument("model", help="path for the model .pkl file")
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    main(args.model, args.k, args.out)