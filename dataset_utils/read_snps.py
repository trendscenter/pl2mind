"""
Utility for loading SNP data.
"""

__author__ = "Devon Hjelm"
__copyright__ = "Copyright 2014, Mind Research Network"
__credits__ = ["Devon Hjelm"]
__licence__ = "3-clause BSD"
__email__ = "dhjelm@mrn.org"
__maintainer__ = "Devon Hjelm"

import argparse
import gzip
import numpy as np
from os import path
import random
from pylearn2.utils import serial
import warnings

def parse_SNP_line(line, snp_format="ORDERED", minor_majors=None, read_names_only=False):
    """
    Function to parse a single line in SNP file according to a format.
    """
    elems = line.translate(None, '\n').split(" ")
    if snp_format == "ORDERED":
        assert (len(elems) - 5) % 3 == 0,\
            "Incorrect number of elements in SNP line, %d, should be 5 + a, where a %% 3 == 0" % len(elems)
        name, location, minor_allele, major_allele = elems[1:5]
        if read_names_only:
            return name, None, None, 0
        location = int(location)
        values = np.zeros((len(elems) - 5) / 3)
        
        for j in range(5, len(elems), 3):
            for i in range(j, j+3):
                elems[i] = int(elems[i])

            assert (sum(elems[j:j+3]) == 1), "Cannot read format of %r" % elems[j:j+3]
            values[(j - 5) / 3] = elems[j:j+3].index(1)
    elif snp_format == "PAIRS":
        assert (len(elems) - 4) % 2 == 0,\
            "Incorrect number of elements in SNP line, %d, should be 4 + a, where a %% 2 == 0" % len(elems)
        name = elems[1]
        if name not in minor_majors:
            return name, None, None, 0
        assert minor_majors is not None
        minor_major = minor_majors[name]
        minor_allele, major_allele = minor_major

        location = int(elems[3])
        values = np.zeros((len(elems) - 4) / 2)
        for j in range(4, len(elems), 2):
            values[(j - 4) / 2] = sum(map(lambda x : 1 if x is major_allele else 0, elems[j:j+2]))
    else:
        raise ValueError("%s formating not supported, use only ORDERED or PAIRS." % snp_format)

    return name, location, (major_allele, minor_allele), values

def read_minor_major(line):
    elems = line.translate(None, "\n").split("\t")
    assert len(elems) == 6
    return (elems[1], (elems[4], elems[5]))

def read_SNP_file(snp_file, snp_format="ORDERED", read_value="VALUES",
                  bim_file=None, hapz_file=None, reference_file=None):
    assert isinstance(snp_file, (file, str))
    if isinstance(snp_file, file):
        f = snp_file
    else:
        try:
            f = open(snp_file, "r")
        except IOError:
            f = gzip.open(snp_file + ".gz", "r")

    ref_names = None
    ref_probs = None
    if reference_file is not None:
        ref_names = read_SNP_file(reference_file, snp_format=snp_format,
                                  read_value="NAMES", bim_file=bim_file,
                                  hapz_file=hapz_file)
        ref_values = read_SNP_file(reference_file, snp_format=snp_format,
                                   read_value="VALUES", bim_file=bim_file,
                                   hapz_file=hapz_file)
        ref_probs = dict((name, [len(np.where(vs == i)[0].tolist()) * 1./ len(vs) for i in range(3)])
                         for name, vs in zip(ref_names, ref_values))
        for name in ref_probs:
            assert name in ref_names
            assert sum(ref_probs[name]) - 1.0 < 0.0001, ref_probs[name]

    if snp_format == "PAIRS":
        assert bim_file
        assert hapz_file
        with open(bim_file, "r") as bim_f:
            bim_lines = bim_f.readlines()
            minor_majors = dict(map(lambda line : read_minor_major(line), bim_lines))
        with open(hapz_file, "r") as hapz_f:
            for line in hapz_f.readlines():
                parsed = line.split(" ")
                parsed = [p.translate(None, "\n") for p in parsed]
                name = parsed[1]
                first, second = int(parsed[3]), int(parsed[4])
                assert (first, second) in [(1, 2), (2, 1)]
                if (first, second) == (2, 1):
                    minor_majors[name] = (minor_majors[name][1], minor_majors[name][0])
    else:
        minor_majors = None

    names = []
    locations = []
    alleles = []
    lines = f.readlines(); f.close()
    lines = [l for l in lines if ("rsdummy" not in l)]
    if snp_format == "PAIRS":
        line_parsed = lines[0].split(" ")
        name0 = line_parsed[1]
        mm0 = {name0: (line_parsed[4], line_parsed[4])}
    else:
        mm0 = None
    _, _, _, value0 = parse_SNP_line(lines[0],
                                     snp_format=snp_format,
                                     minor_majors=mm0)
    values = {}
    for i, line in enumerate(lines):
        name, location, allele_pair, value = parse_SNP_line(line, snp_format=snp_format,
                                                            minor_majors=minor_majors,
                                                            read_names_only=(read_value=="NAMES"))

        names.append(name)
        if read_value != "NAMES":
            locations.append(locations)
            alleles.append(allele_pair)
            values[name] = value

    if ref_names is not None:
        names = ref_names

    if read_value == "NAMES":
        return names

    missed_names = 0
    for name in names:
        if name not in values:
            missed_names += 1
    if missed_names > 0:
        warnings.warn("Some SNPs missing from new dataset (%d, %.2f%%). Filling with values from MCIC priors."\
                          % (missed_names, 100 * float(missed_names) / len(names)))
    if missed_names == 0:
        values = [values[name] for name in names]
    else:
        values = [values.get(name, np.random.choice(3,
                                                    p=ref_probs[name],
                                                    size=value0.shape[0]).astype(np.int8)) for name in names]
    total_values = np.zeros((len(values), value0.shape[0]), dtype=np.int8)
    for i, value in enumerate(values):
        total_values[i] = value
    
    if read_value == "VALUES":
        return total_values
    else:
        raise NotImplementedError("Can only return VALUES, NAMES, not %s yet" % read_value)

def read_labels(line):
    elems = line.split(" ")
    elems = [int(e) for e in elems]
    assert len(elems) % 2 == 0, "Label line must be in for format major_allele minor_allele, e.g. 1 1 or 0 0 repeated for each subject."
    labels = []
    for i in range(0, len(elems), 2):
        assert elems[i] == elems[i+1], "Can only read 0 0 (healthies) or 1 1 (schizophrenia)."
        labels.append(elems[i])
    return labels

def make_labels(num_samples, label_file, snp_format="ORDERED"):
    assert label_file is not None, "Must supply a label file.  Try with argument --label_file <LABEL_FILE>"
    if snp_format == "ORDERED":
        label_file = label_file.split("/")[-1]
        if "cases" in label_file:
            label = 1
        elif "controls" in label_file:
            label = 0
        else:
            raise ValueError("Cannot parse filename: %s" % label_file)
        return [label] * num_samples
    elif snp_format == "PAIRS":
        with open(label_file, "r") as f:
            labels = read_labels(f.readlines()[0].translate(None, "\r\n"))
        return labels
    else:
        raise ValueError("% formating not supported, use only ORDERED or PAIRS."
                         % snp_format)

def save_gen_data(source_directory, num_chromosomes, num_samples, directory=None,
                  shuffle=True):
    file_string = "chr%d_risk_n%d.%s.gen"
    chr_dir = "chr%d"
    gen_tag = "gen"

    idx = None
    for c in range(1, num_chromosomes+1):
        print "Processing generated chromosome %d" % c
        controls_file = path.join(source_directory, chr_dir % c,
                                  file_string % (c, num_samples, "controls"))
        samples = read_SNP_file(controls_file).T
        
        cases_file = path.join(source_directory, chr_dir % c,
                               file_string % (c, num_samples, "cases"))
        cases = read_SNP_file(cases_file).T

        # The code was already set to do this as (dim, samples),
        # which is backwards for pylearn2
        # TODO(dhjelm): fix this
        samples = np.concatenate((samples, cases), axis=0)

        if shuffle:
            if idx is None:
                idx = range(samples.shape[0])
                random.shuffle(idx)
            
            samples = samples[idx, :]

        if directory:
            p = serial.preprocess("${PYLEARN2_NI_PATH}/%s" % directory)
            chr_tag = chr_dir % c
            data_path = path.join(p, gen_tag + "." + chr_tag + ".npy")
            np.save(data_path, samples)
    
    # Save the labels
    labels =\
        make_labels(samples.shape[0] - cases.shape[0], "controls") +\
        make_labels(cases.shape[0], "cases")
    assert len(labels) == samples.shape[0]
    if shuffle:
        assert idx is not None
        labels = [labels[i] for i in idx]
    if directory:
        label_path = path.join(p, gen_tag + "_labels.npy")
        np.save(label_path, labels)

def save_tped_data(source_directory,
                   num_chromosomes,
                   reference_directory=None,
                   directory=None,
                   label_file=None):

    assert label_file is not None
    file_string = "chr%d.tped"
    chr_dir = "chr%d"
    tag = "real"
    if reference_directory is None:
        reference_directory = source_directory 

    sample_num = None
    for c in range(1, num_chromosomes + 1):
        print "Processing subject chromosome %d" % c

        bim_file = path.join(reference_directory, chr_dir % c, "chr%d.bim" % c)
        hapz_file = path.join(reference_directory, chr_dir % c, "chr%d.hapz" % c)
        chr_file = path.join(source_directory, chr_dir % c, file_string % c)
        if reference_directory != source_directory:
            reference_file = path.join(reference_directory, chr_dir % c, "chr%d.tped" % c)
        else:
            reference_file = None

        samples = read_SNP_file(chr_file,
                                snp_format="PAIRS",
                                bim_file=bim_file,
                                hapz_file=hapz_file,
                                reference_file=reference_file).T
        
        if sample_num is None:
            sample_num = samples.shape[0]
        else:
            assert sample_num == samples.shape[0]

        if directory:
            p = serial.preprocess("${PYLEARN2_NI_PATH}/%s" % directory)
            chr_tag = chr_dir % c
            data_path = path.join(p, tag + "." + chr_tag + ".npy")
            np.save(data_path, samples)

    labels = make_labels(sample_num,
                         label_file=label_file,
                         snp_format="PAIRS")
    if directory:
        label_path = path.join(p, tag + "_labels.npy")
        np.save(label_path, labels)
    else:
        return samples, labels

def check_snps(directory, chromosome):
    gen_file = "chr%d_risk_n10000.cases.gen" % chromosome
    sub_file = "chr%d.tped" % chromosome
    bim_file = "chr%d.bim" % chromosome
    chr_dir = "chr%d" % chromosome

    gen_names = read_SNP_file(path.join(directory, chr_dir, gen_file),
                              snp_format="ORDERED",
                              read_value="NAMES")
    sub_names = read_SNP_file(path.join(directory, chr_dir, sub_file),
                              snp_format="PAIRS",
                              read_value="NAMES",
                              bim_file=path.join(directory, chr_dir, bim_file))
#    print zip(gen_names, sub_names)
    assert gen_names == sub_names
    print "SNP names the same for chromosome %d" % chromosome

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("source_directory")
    parser.add_argument("--format", default="ORDERED")
    parser.add_argument("--label_file", default=None)
    parser.add_argument("--out", default="snp")
    parser.add_argument("--chromosomes", default=22)
    parser.add_argument("--samples", default=1000)
    parser.add_argument("--check", default=0)
    parser.add_argument("--reference_directory", default=None,
                        help="Directory to pull hapz and bim files from. Use when aligning to other data.")
    return parser

if __name__ == "__main__":
    assert path.isdir(serial.preprocess("${PYLEARN2_NI_PATH}")),\
        "Did you export PYLEARN2_NI_PATH?"

    parser = make_argument_parser()
    args = parser.parse_args()

    if args.check:
        check_snps(args.source_directory, int(args.check))
    else:
        if args.format == "ORDERED":
            save_gen_data(args.source_directory,
                          num_chromosomes=int(args.chromosomes),
                          num_samples=int(args.samples),
                          directory=args.out)
        elif args.format == "PAIRS":
            save_tped_data(args.source_directory,
                           num_chromosomes=int(args.chromosomes),
                           directory=args.out,
                           label_file=args.label_file,
                           reference_directory=args.reference_directory
                           )
        else:
            raise ValueError("%s format not supported, must be ORDERED or PAIRS")
    """
    samples = read_SNP_file(args.snp_file, bim_file=args.bim, snp_format=args.format)
    print samples
    num_samples = samples.shape[1]
    if args.label_file is not None:
        labels = make_labels(num_samples, args.label_file, snp_format=args.format)
        print labels
    """
