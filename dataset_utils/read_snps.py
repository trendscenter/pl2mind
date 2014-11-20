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
import logging
import gzip
import numpy as np
from os import listdir
from os import path
import random
from pylearn2.utils import serial
import sys
import warnings


logging.basicConfig(format="[%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def parse_bim_line(line):
    """
    Parse a bim line.
    Format should be: chromosome SNP_name 0 location allele_1 allele_2. allele_1 != allele_2.

    Parameters
    ----------
    line: str
        bim line to parse.

    Returns: tuple
        Dictionary entry with SNP name as key.
    """
    elems = line.translate(None, "\n").split("\t")
    try:
        assert len(elems) == 6
        chromosome = int(elems[0])
        SNP_name = elems[1]
        assert int(elems[2]) == 0, "Third index is not 0"
        location = int(elems[3])
        allele_1 = elems[4]
        assert allele_1 in "TCAG", "Allele not in TCAG"
        allele_2 = elems[5]
        assert allele_2 in "TCAG", "Allele not in TCAG"
        assert allele_1 != allele_2, "Allele 1 and 2 are equal."
    except AssertionError as e:
        raise ValueError("Could not parse bim line \"%s\"(%s)" % (line, e))
    return (SNP_name, {"location": location, "chromosome": chromosome,
                       "allele_1": allele_1, "allele_2": allele_2})

def parse_haps_line(line):
    """
    Parse a haps line.
    Format should be: chromosome SNP_name location minor(index) major(index) + subject data. Subject data
    are pairs 00->0, 01->1, 10->1, 11->2

    Parameters
    ----------
    line: str
        haps line to parse.

    Returns: tuple
        Dictionary entry with SNP name as key.
    """
    elems = line.translate(None, "\n").split(" ")

    try:
        assert (len(elems) - 5) % 2 == 0
        chromosome = int(elems[0])
        SNP_name = elems[1]
        location = int(elems[2])
        minor = int(elems[3])
        major = int(elems[4])
        assert (minor, major) in [(1, 2), (2, 1)]
        values = np.zeros((len(elems) - 5) // 2, dtype=np.int8)
        for i in range(5, len(elems), 2):
            x = int(elems[i])
            y = int(elems[i+1])
            assert (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)]
            values[(i - 5) // 2] = x + y
    except AssertionError:
        raise ValueError("Could not parse haps line \"%s\"" % line)

    return (SNP_name, {"location": location, "chromosome": chromosome,
                       "minor": minor, "major": major, "values": values})

def parse_tped_line(line):
    """
    Parse a line in tped format.
    Line format should be: chromosome SNP_name 0 location + subject data.
    Subject data should be pairs in "TCAG".

    Parameters
    ----------
    line: str
        tped line to be parsed.

    Returns
    -------
    dictionary entry: tuple
        Dictionary entry with the name of the SNP as the key and a dictionary
        of location and values.
    """
    try:
        elems = line.translate(None, '\n').split(" ")
        assert (len(elems) - 4) % 2 == 0        
        chromosome = int(elems[0])
        SNP_name = elems[1]
        assert int(elems[2]) == 0, "Third element is not 0"
        location = int(elems[3])
        values = []
        for j in range(4, len(elems), 2):
            assert elems[j] in "TCAG", "Element %d not in TCAG" % j
            assert elems[j+1] in "TCAG", "Element %d not in TCAG" % j+1
            values.append((elems[j], elems[j+1]))
    except AssertionError as e:
        raise ValueError("Could not parse tped line \"%s\"(%s)" % (line, e))

    return (SNP_name, {"location": location, "chromosome": chromosome,
                       "values": values})

def parse_gen_line(line):
    """
    Parse a line in gen format.
    Format should be snp_%d name location minor_allele major_allele + subject data.
    Subject data is in format binary triples where the number of on bits sums to 1,
    e.g., 001 or 100.

    Parameters
    ----------
    line: str
        Line to be parsed.

    Returns
    -------
    dictionary entry: tuple
        Dictionary entry with the name of the SNP as the key and a dictionary
        of location and values.
    """
    try:
        elems = line.translate(None, '\n').split(" ")
        assert (len(elems) - 5) % 3 == 0, "Incorrect line length (%d)." % len(elems)
        assert "snp" in elems[0], "First element not snp number."
        SNP_name, location, minor_allele, major_allele = elems[1:5]
        assert minor_allele in "TCAG", "Minor allele not in TCAG"
        assert major_allele in "TCAG", "Major allele not in TCAG"
        assert minor_allele != major_allele, "Minor and major allele are equal."
        location = int(location)
        
        values = np.zeros((len(elems) - 5) // 2, dtype=np.int8)
        for j in range(5, len(elems), 3):
            for i in range(j, j+3):
                elems[i] = int(elems[i])
            assert (sum(elems[j:j+3]) == 1), "Line segment value does not add to 1 (%d,%d,%d)" % elems[j:j+3]
            values[(j - 5) / 3] = elems[j:j+3].index(1)
    except AssertionError as e:
        raise ValueError("Could not parse gen line \"%s\"(%s)" % (line, e))

    return (SNP_name, {"location": location, "values": values,
                      "minor_allele": minor_allele, "major_allele": major_allele})

def parse_labels_file(label_file):
    label_file = path.abspath(label_file)
    def read_line(line):
        elems = line.translate(None, "\n").split(" ")
        try:
            elems = [int(e) for e in elems]
            assert len(elems) % 2 == 0, "Label line must have even elements."
            labels = []
            for i in range(0, len(elems), 2):
                assert elems[i] == elems[i+1], "Can only read 0 0 (healthies) or 1 1 (schizophrenia)."
                labels.append((elems[i] + 1) % 2)
        except AssertionError:
            raise ValueError("Could not parse label line \"%s\"(%s)" % (line, e))
        return labels

    with open(label_file, "r") as f:
        lines = f.readlines()
        if len(lines) != 1:
            raise ValueError("Could not read label file %s, only one line allowed, %d found"\
                                 % (label_file, len(lines)))
        labels = read_line(lines[0])

    return labels

def parse_file(file_name):
    """
    Read a file into a dictionary.
    Extensions are .bim, .haps, .tped, or .gen
    Keys are SNP names, entries depend on the filetype.

    Parameters
    ----------
    file_name: str
        Location of file to parse.
    
    Returns
    -------
    parse_dict: dictionary
        Dictionary with SNP name keys.
    """
    logger.info("Parsing %s" % file_name)
    exts = ["bim", "haps", "tped", "gen"]
    ext = file_name.split(".")[-1]
    if ext == "ped":
        return
    if ext not in exts:
        raise NotImplementedError("Extension not supported (%s), must be in %s" % (ext, exts))
    
    ignore = ["rsdummy"]
    
    method_dict = {"bim": parse_bim_line,
                   "haps": parse_haps_line,
                   "tped": parse_tped_line,
                   "gen": parse_gen_line,
                   }

    parse_dict = {}
    parse_dict["ext"] = ext
    with open(file_name, "r") as f:
        for line in f.readlines():
            entry = method_dict[ext](line)
            if entry[0] in ignore:
                continue
            if entry[0] in parse_dict:
                raise ValueError("Found a duplicate SNP(%s) in .%s file." % (entry[0], ext))
            parse_dict[entry[0]] = entry[1]
    return parse_dict

def read_chr_directory(directory):
    """
    Read a directory with SNP data.
    Extras data and other details from SNP files.

    Parameters
    ----------
    directory: str
        Path to SNP directory.

    Returns
    -------
    snp_dict: dict with extension keys and dictionary values. Dictionaries depend on the extension.
    """
    directory = path.abspath(directory)
    file_dict = parse_chr_directory(directory)
    snp_dict = {}
    for ext in file_dict:
        file_name = path.join(directory, file_dict[ext])
        parse_dict = parse_file(file_name)
        snp_dict[ext] = parse_dict
    if "haps" in snp_dict:
        assert "bim" in snp_dict
        for key in snp_dict["bim"]:
            if key == "ext": continue
            minor, major = [(snp_dict["haps"][key])[m] for m in ["minor", "major"]]
            allele_1, allele_2 = [(snp_dict["bim"][key])[m] for m in ["allele_1", "allele_2"]]
            minor_allele = allele_1 if minor == 1 else allele_2
            major_allele = allele_2 if minor == 1 else allele_1
            snp_dict["haps"][key]["minor_allele"] = minor_allele
            snp_dict["haps"][key]["major_allele"] = major_allele
            if key in snp_dict["tped"]:
                snp_dict["tped"][key]["minor_allele"] = minor_allele
                snp_dict["tped"][key]["major_allele"] = major_allele
    return snp_dict

def parse_chr_directory(directory):
    """
    Parses SNP processing files from a SNP directory.
    Parses out haps, bim, ped, tped, and gen files.

    Parameters
    ----------
    directory: str
        SNP directory to parse.

    Returns
    -------
    file_dict: dict
        Dictionary of extension keys and file path values.
    """
    # We need to ignore these files for now.
    ignore_string = "HAPGENinput"
    files = [f for f in listdir(directory) if path.isfile(path.join(directory,f))]
    files = [f for f in files if ignore_string not in f]
    file_dict = {}

    def insert(ext, elem):
        if ext == "gen":
            if "cases" in elem:
                ext = "cases"
            elif "control" in elem:
                ext = "controls"
            else:
                raise ValueError("Cannot parse gen file %s" % elem)
        if ext in file_dict:
            raise ValueError("Multiple %s files found in %s" %(ext, directory))
        file_dict[ext] = path.join(directory, elem)

    logger.info("Found files %r in %s" % (files, directory))

    for f_name in files:
        ext = f_name.split(".")[-1]
        if ext == "haps":
            # Gen directories will have 2 haps files...
            if not "cases" in f_name:
                insert(ext, f_name)
        elif ext in ["bim", "ped", "tped", "gen"]:
            insert(ext, f_name)
        else:
            logger.warn("Unknown file type %s" % ext)
#            raise ValueError("Unknown file type %s" % ext)

    if "cases" in file_dict and "controls" in file_dict:
        # Only cases and controls needed for gen files.
        file_dict = dict((f, file_dict[f]) for f in ["cases", "controls"])
    else:
        for key in ["bim", "haps", "ped"]:
            if key not in file_dict:
                raise ValueError("%s not found in %s" % (key, directory))
        if "tped" not in file_dict:
            logger.warning("tped file not found in %s, process only with .haps" % directory)
        
    logger.info("Parsed %s to %r" % (directory, file_dict))
    return file_dict  

def parse_dataset_directory(directory, chromosomes=22):
    directory = path.abspath(directory)
    subdirs = [d for d in listdir(directory) if path.isdir(path.join(directory,d))]
    dir_dict = {}
    for c in range(1, chromosomes + 1):
        chr_dir = next((d for d in subdirs if d == "chr%d" % c or "chr%d_" % c in d), None)
        if chr_dir is None:
            raise ValueError("Chromosome %d not found in %s" % (c, directory))
        dir_dict[c] = path.join(directory, chr_dir)
    if path.isfile(path.join(directory, "diagnosis_ref.txt")):
        dir_dict["labels"] = path.join(directory, "diagnosis_ref.txt")

    logger.info("Directory dictionary: %r" % dir_dict)
    return dir_dict

def read_dataset_directory(directory, chromosomes=22):
    logger.info("Reading %d chromosomes from directory %s" % (chromosomes, directory))
    dir_dict = parse_dataset_directory(directory, chromosomes=chromosomes)
    dataset_dict = {}
    for key in dir_dict:
        if key == "labels":
            dataset_dict["labels"] = parse_labels_file(dir_dict[key])
        else:
            chr_dict = read_chr_directory(dir_dict[key])
            dataset_dict[key] = chr_dict

    if "labels" not in dataset_dict:
        for c in range(1, chromosomes + 1):
            assert "cases" in dataset_dict[c]
            assert "controls" in dataset_dict[c]
            assert have_same_SNPs(dataset_dict[c]["cases"], dataset_dict[c]["controls"])

    return dataset_dict

def pull_dataset(dataset_dict, chromosomes=22):
    if "labels" not in dataset_dict:
        num_cases = None
        num_controls = None
        data = None
        for c in range(1, chromosomes + 1):
            try:
                cases = dataset_dict[c]["cases"]
                controls = dataset_dict[c]["controls"]
                assert have_same_SNPs(cases, controls), "Cases and controls have different SNPs."
                
                cases_data = pull_gen_data(cases)
                controls_data = pull_gen_data(controls)

                if num_cases is None:
                    num_cases = cases_data.shape[0]
                if num_controls is None:
                    num_controls = controls_data.shape[0]
                assert cases_data.shape[0] == num_cases,\
                    "Cases data has inconsistent subjects (%d vs %d)" % (cases_data.shape[0], num_cases)
                assert controls_data.shape[0] == num_controls,\
                    "Control data has inconsistent subjects (%d vs %d)" % (control_data.shape[0], num_controls)
                assert cases_data.shape[1] == controls_data.shape[1],\
                    "Cases and controls have difference number of columns (%d vs %d)."\
                    % (cases_data.shape[1], controls_data.shape[1])
                if data is None:
                    data = np.concatenate((controls_data, cases_data), axis=0)
                else:
                    data = np.concatenate((data, np.concatenate(
                                (controls_data, cases_data), axis=0)), axis=1)
            except AssertionError as e:
                raise ValueError("Pulling data from dataset chromosome %d failed (%s)" % (c, e))
        labels = [0] * num_controls + [1] * num_cases
    else:
        data = None
        for c in range(1, chromosomes + 1):
            chr_data = pull_haps_data(dataset_dict[c]["haps"])
            if data is None:
                data = chr_data
            else:
                data = np.concatenate((data, chr_data), axis=1)
        labels = dataset_dict["labels"]

    return data, labels

def pull_haps_data(haps_dict, reference_names=None):
    logger.info("Getting haps data.")
    samples = haps_dict[haps_dict.keys()[0]]["values"].shape[0]

    if reference_names == None:
        reference_names = haps_dict.keys()

    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    for i, SNP_name in enumerate(reference_names):
        if SNP_name == "ext":
            continue
        data[:, i] = haps_dict[SNP_name]["values"]

    return data

def pull_tped_data(tped_dict, reference_names=None):
    logger.info("Getting tped data.")
    samples = len(tped_dict[tped_dict.keys()[0]]["values"])

    if reference_names == None:
        reference_names = tped_dict.keys()

    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    for i, SNP_name in enumerate(reference_names):
        if SNP_name == "ext":
            continue
        minor, major = [tped_dict[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        values = tped_dict[SNP_name]["values"]

        for j, value in enumerate(values):
            x, y = tuple(0 if v == minor else 1 for v in value)
            data[j, i] = x + y

    return data

def pull_gen_data(gen_dict, reference_names=None):
    logger.info("Getting gen data")

    if reference_names is None:
        reference_names = gen_dict.keys()
    samples = len(gen_dict[reference_names[0]]["values"])

    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    for i, SNP_name in enumerate(reference_names):
        data[:, i] = gen_dict[SNP_name]["values"]

    return data

def pull_data(file_dict, reference_names=None):
    ext = file_dict["ext"]
    if ext == "gen":
        return pull_gen_data(file_dict, reference_names)
    elif ext == "haps":
        return pull_haps_data(file_dict, reference_names)
    elif ext == "tped":
        return pull_tped_data(file_dict, reference_names)
    else:
        raise ValueError("Cannot pull data from extension %s" % ext)

def compare_SNPs(file_A, file_B):
    if isinstance(file_A, str):
        file_A = path.abspath(file_A)
        dict_A = parse_file(file_A)
    else:
        assert isinstance(file_A, dict)
        dict_A = file_A
    if isinstance(file_B, str):
        file_B = path.abspath(file_B)
        dict_B = parse_file(file_B)
    else:
        assert isinstance(file_B, dict)
        dict_B = file_B
    SNPs_A = set(dict_A.keys())
    SNPs_B = set(dict_B.keys())
    a_not_in_b = SNPs_A.difference(SNPs_B)
    b_not_in_a = SNPs_B.difference(SNPs_A)
    neg_intercept = SNPs_A.symmetric_difference(SNPs_B)
    print "Comparing A and B:"
    print "%d SNPs in A not in B" % len(a_not_in_b)
    if len(a_not_in_b) < 10 and len(a_not_in_b) != 0:
        print a_not_in_b
    print "%d SNPs in B not in A" % len(b_not_in_a)
    if len(b_not_in_a) < 10 and len(b_not_in_a) != 0:
        print b_not_in_a
    print "%d SNPs symmetric difference A and B" % len(neg_intercept)

def A_is_compatible_reference_for_B(file_A, file_B):
    if isinstance(file_A, str):
        file_A = path.abspath(file_A)
        dict_A = parse_file(file_A)
    else:
        assert isinstance(file_A, dict)
        dict_A = file_A
    if isinstance(file_B, str):
        file_B = path.abspath(file_B)
        dict_B = parse_file(file_B)
    else:
        assert isinstance(file_B, dict)
        dict_B = file_B
    SNPs_A = set(dict_A.keys())
    SNPs_B = set(dict_B.keys())
    A_not_in_B = SNPs_A.difference(SNPs_B)
    return len(A_not_in_B) == 0

def A_has_similar_priors_to_B(file_A, file_B):
    if isinstance(file_A, str):
        file_A = path.abspath(file_A)
        dict_A = parse_file(file_A)
    else:
        assert isinstance(file_A, dict)
        dict_A = file_A
    if isinstance(file_B, str):
        file_B = path.abspath(file_B)
        dict_B = parse_file(file_B)
    else:
        assert isinstance(file_B, dict)
        dict_B = file_B
    if A_is_compatible_reference_for_B(dict_A, dict_B):
        logger.debug("A is the reference.")
        reference = dict_A
    elif A_is_compatible_reference_for_B(dict_B, dict_A):
        logger.info("B is the reference.")
        reference = dict_B
    else:
        logger.info("Prior check failed due to no compatible reference.")
        return False
    data_A = pull_data(dict_A, reference_names=[k for k in reference.keys() if k != "ext"])
    data_B = pull_data(dict_B, reference_names=[k for k in reference.keys() if k != "ext"])
    priors_A = [(data_A == i).sum(0) * 1. / data_A.shape[0] for i in range(3)]
    priors_B = [(data_B == i).sum(0) * 1. / data_B.shape[0] for i in range(3)]
    similar = True
    for prior_A, prior_B in zip(priors_A, priors_B):
        percent_off = (len(np.where(prior_A - prior_B > 0.15)[0].tolist()) * 1. / prior_A.shape[0])
        if percent_off > .05:
            logger.warn("Priors not close: %.2f%% off by 15%% or more" % (percent_off * 100))
            similar = False
    return similar

def A_is_aligned_to_B(file_A, file_B):
    if isinstance(file_A, str):
        file_A = path.abspath(file_A)
        dict_A = parse_file(file_A)
    else:
        assert isinstance(file_A, dict)
        dict_A = file_A
    if isinstance(file_B, str):
        file_B = path.abspath(file_B)
        dict_B = parse_file(file_B)
    else:
        assert isinstance(file_B, dict)
        dict_B = file_B
    if A_is_compatible_reference_for_B(dict_A, dict_B):
        reference = dict_A
    elif A_is_compatible_reference_for_B(dict_B, dict_A):
        reference = dict_B
    else:
        logger.info("Alignment check failed due to no compatible reference.")
        return False
    for key in reference.keys():
        if key == "ext":
            continue
        if dict_A[key]["minor_allele"] != dict_B[key]["minor_allele"]:
            logger.info("Alleles for %s not aligned (%s, %s) vs (%s, %s)"
                        % (key, dict_A[key]["minor_allele"], dict_A[key]["major_allele"],
                           dict_B[key]["minor_allele"], dict_B[key]["major_allele"]))
            return False
    return True

def align_A_to_B(file_A, file_B):
    logger.info("Aligning files")
    if isinstance(file_A, str):
        file_A = path.abspath(file_A)
        dict_A = parse_file(file_A)
    else:
        assert isinstance(file_A, dict)
        dict_A = file_A
    if isinstance(file_B, str):
        file_B = path.abspath(file_B)
        dict_B = parse_file(file_B)
    else:
        assert isinstance(file_B, dict)
        dict_B = file_B
    assert A_is_compatible_reference_for_B(dict_B, dict_A)
    reference_names = [k for k in dict_B.keys() if k != "ext"]

    for SNP_name in reference_names:
        minor_A, major_A = [dict_A[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        minor_B, major_B = [dict_B[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        if minor_A == minor_B and major_A == major_B:
            pass
        elif minor_A == major_B and major_A == minor_B:
            dict_A[SNP_name]["values"] = (-(dict_A[SNP_name]["values"] - 1)) + 1
            dict_A[SNP_name]["minor_allele"] = dict_B[SNP_name]["minor_allele"]
            dict_A[SNP_name]["major_allele"] = dict_B[SNP_name]["major_allele"]
            assert dict_A[SNP_name]["minor_allele"] == dict_B[SNP_name]["minor_allele"]
        else:
            raise ValueError()
    assert A_is_aligned_to_B(dict_A, dict_B)
    logger.info("Alignment finished.")
    return dict_A

def have_same_SNPs(dict_A, dict_B):
    neg_intercept = set(dict_A.keys()).symmetric_difference(set(dict_B.keys()))
    return len(neg_intercept) == 0

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="snp")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show more verbosity!")

    subparsers = parser.add_subparsers(help="sub-command help")
    subparsers.required = True

    compare_parser = subparsers.add_parser("compare", help="Compare 2 chromosome directories")
    compare_parser.set_defaults(which="compare")
    compare_parser.add_argument("dir_1")
    compare_parser.add_argument("dir_2")

    extract_parser = subparsers.add_parser("extract")
    extract_parser.set_defaults(which="extract")
    extract_parser.add_argument("directory", help="SNP dataset directory.")
    extract_parser.add_argument("-c", "--chromosomes", default=22,
                                type=int, help="Number of chromosomes to process.")

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.which == "compare":
        dir_dict_1 = read_chr_directory(args.dir_1)
        dir_dict_2 = read_chr_directory(args.dir_2)
        def get_dict(dir_dict):
            if "tped" in dir_dict:
                return dir_dict["tped"]
            elif "cases" in dir_dict:
                return dir_dict["cases"]
            else:
                raise ValueError("No haps or cases in %r." % dir_dict.keys())
        dict_1 = get_dict(dir_dict_1)
        dict_2 = get_dict(dir_dict_2)

        compare_SNPs(dict_1, dict_2)
        A_lessthan_B = A_is_compatible_reference_for_B(dict_1, dict_2)
        print "%s can%s be used as a reference for %s"\
            % (args.dir_1, "" if A_lessthan_B else " not", args.dir_2)
        B_lessthan_A = A_is_compatible_reference_for_B(dict_2, dict_1)
        print "%s can%s be used as a reference for %s"\
            % (args.dir_2, "" if B_lessthan_A else " not", args.dir_1)
        if A_is_aligned_to_B(dict_1, dict_2):
            print "A and B are aligned."
        else:
            print "A and B are not aligned."
        if A_lessthan_B:
            dict_2 = align_A_to_B(dict_2, dict_1)
        elif B_lessthan_A:
            dict_1 = align_A_to_B(dict_1, dict_2)
        else:
            raise ValueError()
        if A_has_similar_priors_to_B(dict_1, dict_2):
            print "A and B have similar priors."
        else:
            print "A and B do not have similar priors."



    elif args.which == "extract":
        data_dict = read_dataset_directory(args.directory, chromosomes=args.chromosomes)
        pull_dataset(data_dict, chromosomes=args.chromosomes)
