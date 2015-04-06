"""
Module to perform AM-FM on EEG data.
"""

__author__ = "Alvaro Ulloa"
__copyright__ = "Copyright 2015, Mind Research Network"
__credits__ = ["Alvaro Ulloa"]
__licence__ = "3-clause BSD"
__email__ = "aulloa@mrn.org"
__maintainer__ = "Alvaro Ulloa"


import argparse
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pl2mind import logger

import scipy.signal as ss


logger = logger.setup_custom_logger("pl2mind", logging.WARNING)

def qea(im):
    """
    Quasi-eigen approximation function.

    Parameters
    ----------
    im: array_like
        1d vector that contains a time series

    Returns
    -------
    ia: array_like
        instantaneous amplitude
    ip: array_like
        instantaneous phase
    ifeq: array_like
        instantaneous frequency
    """
    im = im.ravel()
    # computes analytic signal
    H = ss.hilbert(im)
    H = im + 1j * H
    # obtain IA and IP from analytic signal
    ia = np.abs(H)
    ip = np.angle(H)
    # obtain IF using QEA function
    h1 = H[1:-1]
    h0 = H[:-2]
    h2 = H[2:]
    ifeq = np.real(np.arccos((h2 + h0) / (2 * h1)) / np.pi / 2)
    # pad extremes copying
    ifeq = np.hstack((ifeq[:1], ifeq, ifeq[-1:]))
    return(ia, ip, ifeq)

def amfm_CCA(im):
    """
    Channel component analysis for AM-FM.

    Parameters
    ----------
    im: array_like
        1d vector that contains a time series

    Returns
    -------
    ia: array_like
        Instantaneous amplitude computed for 3 channels
    ip: array_like
        Instantaneous phase computed for 3 channels
    ifeq: array_like
        Instantaneous frequency computed for 3 channels
    """
    # Filter bank
    bp = [ ]
    # Low pass 0 0.02
    bp.append(ss.remez(50, [0, 0.02, 0.05, 0.5], [1,0]))
    # Pass band 0.02 0.25
    bp.append(ss.remez(50, [0, 0.02, 0.05, 0.20, 0.25, 0.5], [0, 1, 0]))
    # High pass 0.25 0.5
    bp.append(ss.remez(50, [0, 0.20, 0.25, 0.5], [0, 1], type = "hilbert"))
    # apply filterbank
    filt = lambda x: ss.convolve(im, x, "same")
    in_channels = map(filt, bp)
    # compute IA, IP and IF from filterbank output
    out_channels = map(qea, in_channels)
    # Organize results into a matrix of channels by time points
    ia = []
    ip = []
    ifeq = []
    for chan in out_channels:
        ia.append(chan[0])
        ip.append(chan[1])
        ifeq.append(chan[2])
    return(np.array(ia), np.array(ip), np.array(ifeq))

def amfm_DCA(im):
    """
    Dominant component analysis for AM-FM.

    Parameters
    ----------
    im: array_like
        1d vector that contains a time series

    Returns
    -------
    ia: array_like
        instantaneous amplitude
    ip: list
        instantaneous phase
    ifeq: list
        instantaneous frequency
    """
    ia,ip,ifeq = amfm_CCA(im)
    t = np.argmax(ia, axis = 0)
    ifeq = [f[t[n]] for n, f in enumerate(ifeq.T)]
    ip = [f[t[n]] for n, f in enumerate(ip.T)]
    ia = ia.max(axis=0)
    return(ia, ip, ifeq)

def demo(args):
    """
    Demo of AM-FM.

    Parameters
    ----------
    args: argparse args
    """
    logger.info("Running a demo of AM-FM")
    # number of time points
    N = args.N

    logger.info("Generating a chirp that sweeps all frequencies")
    c = ss.chirp(range(N), 0 , N - 1, 0.49)

    logger.info("Computing fft for plot")
    C = np.fft.fft(c, N * 10)
    # This is the groundtruth IF
    f = 0.5 * np.arange(N) / (N - 1)

    logger.info("Computing AM-FM DCA")
    ia, ip, ifeq = amfm_DCA(c)
    # plot results

    logger.info("Plotting")
    plt.subplot(311)
    plt.plot(c)
    plt.title("Time series")
    plt.subplot(312)
    plt.plot(np.fft.fftfreq(N * 10), np.abs(C), ".")
    plt.title("Frequency spectrum")
    plt.subplot(313)
    plt.plot(f)
    plt.plot(ifeq)
    plt.legend(["Ideal", "Estimated"], loc = "best")
    plt.title("Frequency vs time")
    if args.out_file:
        plt.savefig(args.out_file)
    else:
        plt.show()

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show more verbosity!")
    subparsers = parser.add_subparsers(help="sub-command help")
    subparsers.required = True

    demo_parser = subparsers.add_parser("demo",
                                        help=("demo am-fm"))
    demo_parser.set_defaults(which="demo")
    demo_parser.add_argument("--N", default=1000, help="Number of time points.")
    demo_parser.add_argument("-o", "--out_file", default=None,
                             help="Out file for demo plot")
    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    if args.which == "demo":
        demo(args)
    else:
        raise NotImplementedError("%s is not currently supported" % args.which)
