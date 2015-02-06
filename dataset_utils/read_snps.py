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
import copy
import logging
import gzip
import numpy as np
from os import listdir
from os import path
import random
import re
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
    if len(elems) == 1:
        elems = line.translate(None, "\n").split(" ")
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
    elems = [e for e in elems if e != ""]

    try:
        assert (len(elems) - 5) % 2 == 0, "Line length error (%s)" % (elems, )
        chromosome = int(elems[0])
        SNP_name = elems[1]
        location = int(elems[2])
        minor = int(elems[3])
        major = int(elems[4])
        assert (minor, major) in [(1, 2), (2, 1)], "Minor major error (%s)" % ((minor, major),)
        values = np.zeros((len(elems) - 5) // 2, dtype=np.int8)
        for i in range(5, len(elems), 2):
            x = int(elems[i])
            y = int(elems[i+1])
            assert (x, y) in [(0, 0), (0, 1), (1, 0), (1, 1)], "Value error (%s)" % ((x, y),)
            values[(i - 5) // 2] = x + y
    except AssertionError as e:
        raise ValueError("Could not parse haps line \"%s\" (%s)" % (line, e))

    return (SNP_name, {"location": location, "chromosome": chromosome,
                       "minor": minor, "major": major, "values": values,
                       "raw_values": elems[5:]})

def write_haps_line(SNP_name, entry, omit_info=False, info_only=False):
    if omit_info:
        assert not info_only

    if not omit_info:
        line = "%(chromosome)d" % entry
        line += " %s " % SNP_name
        line += "%(location)d %(minor)s %(major)s " % entry
    else:
        line = ""
    if not info_only:
        line += " ".join(entry["raw_values"])
    return line

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
        SNP_name, location, allele_1, allele_2 = elems[1:5]
        assert allele_1 in "TCAG", "Allele_1 not in TCAG"
        assert allele_2 in "TCAG", "Allele_2 not in TCAG"
        assert allele_1 != allele_2, "Alleles are equal."
        location = int(location)

        values = np.zeros((len(elems) - 5) // 3, dtype=np.int8)
        for j in range(5, len(elems), 3):
            for i in range(j, j+3):
                elems[i] = int(elems[i])
            assert (sum(elems[j:j+3]) == 1), "Line segment value does not add to 1 (%d,%d,%d)" % elems[j:j+3]
            values[(j - 5) / 3] = elems[j:j+3].index(1)
    except AssertionError as e:
        raise ValueError("Could not parse gen line \"%s\"(%s)" % (line, e))

    return (SNP_name, {"location": location, "values": values,
                      "allele_1": allele_1, "allele_2": allele_2})

def parse_dat_file(dat_file):
    """
    Parse a complete dat file.
    dat files are transposed wrt the rest of the data formats here. In addition, they only contain integer fields,
    so we can use np.loadtxt.
    First 6 columns are ignored.
    Note: must have a bims and info file to process completely.

    Parameters
    ----------
    dat_file: str
        Path for dat file to process.

    Returns
    -------
    data: array-like
    """

    data = np.loadtxt(dat_file)
    data = data[:, 6:].T
    return data

def convert_dat_to_haps(data, info_dict):
    """
    Converts dat to haps.

    Parameters
    ----------
    data: array-like
        Data to be converted
    info_dict: dict
        Haps dictionary with empty info_dict["values"]

    Returns
    -------
    new_haps_dict: dict
        New haps dictionary from data
    """

    assert info_dict["ext"] == "info"
    assert (len(info_dict) - 1) == (data.shape[0] // 2), (len(info_dict), data.shape)

    new_haps_dict = copy.deepcopy(info_dict)
    keys = [k for k in info_dict.keys() if k != "ext"]
    data_idx = [info_dict[k]["line_number"] for k in keys]
    for j, SNP_name in enumerate(keys):
        if SNP_name == "rsdummy":
            continue
        i = 2 * data_idx[j]
        assert i < data.shape[0], (i, data.shape[0])
        data_entry = data[i:i+2]
        assert data_entry.shape[0] == 2,\
            "data entry shape on SNP %s is %s (idx %d out of %d)" % (SNP_name, data_entry.shape, i, data.shape[0])
        value_entry = data_entry.sum(axis=0) - 2

        assert SNP_name in new_haps_dict.keys(), SNP_name
        new_haps_dict[SNP_name]["values"] = value_entry
        assert "minor" in new_haps_dict[SNP_name], SNP_name

    new_haps_dict.pop("rsdummy", None)

    return new_haps_dict

def parse_labels_file(label_file):
    """
    Parses a labels file.
    Lables are single line with pairwise designations of controls vs cases. Space delimited.
    e.g., 0 0 1 1 1 1 translates to [1, 0, 0], where 0 is for conrols and 1 is for cases.

    Parameters
    ----------
    label_file: str
        Path for label file to process.

    Returns
    -------
    labels: list of ints
        The labels from the file.
    """
    label_file = path.abspath(label_file)
    logger.info("Parsing label file %s" % label_file)
    def read_line(line):
        elems = line.translate(None, "\n").split(" ")
        try:
            elems = [int(e) for e in elems]
            assert len(elems) % 2 == 0, "Label line must have even elements."
            labels = []
            for i in range(0, len(elems), 2):
                assert elems[i] == elems[i+1], "Can only read 1 1 (healthies) or 0 0 (schizophrenia)."
                labels.append(elems[i])
        except AssertionError:
            raise ValueError("Could not parse label line \"%s\"(%s)" % (line, e))
        except ValueError:
            raise ValueError("Error with line %s, maybe there's an extra newline?" % line)
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
    exts = ["bim", "haps", "tped", "gen", "info"]
    ext = file_name.split(".")[-1]
    if ext == "gzip":
        open_method = gzip.open
        ext = file_name.split(".")[-2]
    else:
        open_method = open

    if ext == "ped":
        return
    if ext not in exts:
        raise NotImplementedError("Extension not supported (%s), must be in %s" % (ext, exts))


    method_dict = {
        "bim": parse_bim_line,
        "haps": parse_haps_line,
        "info": parse_haps_line,
        "tped": parse_tped_line,
        "gen": parse_gen_line,
        }

    parse_dict = {}
    parse_dict["ext"] = ext
    with open_method(file_name, "r") as f:
        for i, line in enumerate(f.readlines()):
            entry = method_dict[ext](line)
            entry[1]["line_number"] = i
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
    snp_dict: dict with extension keys and dictionary values.
        Dictionaries depend on the extension.
    """

    directory = path.abspath(directory)
    file_dict = parse_chr_directory(directory)
    snp_dict = {"directory": directory}
    for ext in file_dict:
        file_name = path.join(directory, file_dict[ext])
        if ext == "dat":
            continue
        parse_dict = parse_file(file_name)
        snp_dict[ext] = parse_dict

    if "dat" in file_dict:
        info_dict = snp_dict["info"]
        bim_dict = snp_dict["bim"]
        info_keys = [k for k in info_dict.keys() if k != "ext"]
        bim_keys = [k for k in bim_dict.keys() if k != "ext"]
        if len(set(info_keys) - set(bim_keys)) != 0:
            logger.warning("Fixing info %d -> %d. This is a hack" % (len(info_keys), len(bim_keys)))
            assert len(set(bim_keys) - set(info_keys)) == 1, set(bim_keys) - set(info_keys)
            new_info = {"ext": "info"}
            for k in bim_keys:
                if k == "rsdummy":
                    new_info[k] = bim_dict[k]
                    continue
                new_info[k] = info_dict[k]
                new_info[k]["line_number"] = bim_dict[k]["line_number"]
            snp_dict["info"] = new_info

        file_name = path.join(directory, file_dict["dat"])
        data = parse_dat_file(file_name)
        parse_dict = convert_dat_to_haps(data, snp_dict["info"])
        snp_dict["haps"] = parse_dict

    if "tped" in snp_dict:
        snp_dict["haps"] = dict((k, snp_dict["haps"][k])
                                for k in snp_dict["haps"].keys()
                                if k in snp_dict["tped"].keys())
        snp_dict["bim"] = dict((k, snp_dict["bim"][k])
                               for k in snp_dict["bim"].keys()
                               if k in snp_dict["tped"].keys())

    if "haps" in snp_dict:
        assert "bim" in snp_dict
        for key in snp_dict["bim"]:
            if key == "rsdummy": continue
            if key == "ext": continue

            minor, major = [(snp_dict["haps"][key])[m] for m in ["minor", "major"]]
            minor_allele, major_allele = [(snp_dict["bim"][key])[m]
                                          for m in ["allele_1", "allele_2"]]

            if (minor, major) == (2, 1):
                minor_allele, major_allele = major_allele, minor_allele

            snp_dict["haps"][key]["minor_allele"] = minor_allele
            snp_dict["haps"][key]["major_allele"] = major_allele

            if "tped" in snp_dict:
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
    ignore_strings = ["HAPGENinput", "input"]
    files = [f for f in listdir(directory) if path.isfile(path.join(directory,f))]
    files = [f for f in files if not any([ignore_string in f for ignore_string in ignore_strings])]
    file_dict = {}

    def insert(ext, elem):
        if ext == "gen":
            if "cases" in elem:
                ext = "cases"
            elif "controls" in elem:
                ext = "controls"
            else:
                raise ValueError("Cannot parse gen file %s" % elem)
        if ext in file_dict:
            raise ValueError("Multiple %s files found in %s" %(ext, directory))
        file_dict[ext] = path.join(directory, elem)

    logger.info("Found files %r in %s" % (files, directory))

    for f_name in files:
        ext = f_name.split(".")[-1]
        if ext == "gz":
            ext = f_name.split(".")[-2]
        if ext == "haps":
            # Gen directories will have 2 haps files...
            if not "cases" in f_name:
                insert(ext, f_name)
        elif ext in ["bim", "ped", "gen", "tped", "info", "dat"]:
            insert(ext, f_name)
        else:
            logger.warn("Unknown file type %s" % ext)
#            raise ValueError("Unknown file type %s" % ext)

    if "cases" in file_dict and "controls" in file_dict:
        # Only cases and controls needed for gen files.
        file_dict = dict((f, file_dict[f]) for f in ["cases", "controls"])
    else:
        for key in ["bim", "haps"]:
            if key not in file_dict and ("dat" not in file_dict):
                raise ValueError("%s not found in %s (%s)" % (key, directory, file_dict))
        if "tped" not in file_dict:
            logger.warning("tped file not found in %s, process only with .haps" % directory)

    if "dat" in file_dict:
        assert "info" in file_dict
        assert "bim" in file_dict

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

def read_dataset_directory(directory, chromosomes=22,
                           snps_reference=None, align_reference=None, nofill=False):
    """
    Reads a SNP dataset directory with multiple chromosomes.
    Note: Directory must contrain subdirectories with names "chr%d" or chr%d_synthetic
    which fit the chromosome directory specification in parse_chromosome_directory.

    Parameters
    ----------
    directory: str
        Directory to read dataset from.
    chromosomes: int, optional
        Number of chromosomes to process.
    reference_directory: str, optional
        Reference directory to align allelesa and SNPs.

    Returns
    -------
    dataset_dict: dict
        Dictionary of chromosome or "labels" keys and file dictionary or labels values.
    """
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
            assert have_same_SNP_order(dataset_dict[c]["cases"], dataset_dict[c]["controls"])

    if snps_reference is not None:
        logger.info("Setting to snp reference")
        snps_ref_dataset_dict, _, _ = read_dataset_directory(snps_reference,
                                                             chromosomes=chromosomes)

        for key in snps_ref_dataset_dict:
            if key == "labels":
                continue

            assert isinstance(key, int)
            assert key in dataset_dict
            chr_dict = dataset_dict[key]
            def get_dict(d):
                if "tped" in d:
                    return d["tped"]
                if "haps" in d:
                    return d["haps"]
                if "cases" in d:
                    return d["cases"]
                raise ValueError()
            snps_ref_chr_dict = get_dict(snps_ref_dataset_dict[key])

            for ext in ["tped", "haps", "cases", "controls"]:
                if ext not in chr_dict:
                    continue
                data_dict = set_A_with_B(chr_dict[ext], snps_ref_chr_dict, nofill=nofill)
                dataset_dict[key][ext] = data_dict
    else:
        snps_ref_dataset_dict = None

    if align_reference is not None:
        logger.info("Aligning")
        align_ref_dataset_dict, _, _ = read_dataset_directory(align_reference,
                                                           chromosomes=chromosomes)
        for key in align_ref_dataset_dict:
            if key == "labels":
                continue
            assert key in dataset_dict
            chr_dict = dataset_dict[key]
            align_ref_chr_dict = align_ref_dataset_dict[key]
            if "controls" in chr_dict:
                continue
            elif "tped" in chr_dict:
                assert "tped" in align_ref_chr_dict
                ext = "tped"
            elif "haps" in chr_dict:
                continue
            else:
                raise ValueError()
            data_dict = align_A_to_B(chr_dict[ext], align_ref_chr_dict[ext])
            dataset_dict[key][ext] = data_dict
    else:
        align_ref_dataset_dict = None

    return dataset_dict, snps_ref_dataset_dict, align_ref_dataset_dict

def pull_dataset(dataset_dict, chromosomes=22, shuffle=True):
    """
    Pull complete dataset from a dataset directory.
    TODO: currently concatenates the data. Needs to save each chromosome dataset to a different
    numpy file instead. Or return lists or dicts of array-likes

    Parameters
    ----------
    dataset_dict: dict
        Dictionary of chromosome or "labels" keys and file dictionary or labels values.

    Returns
    -------
    data, labels: array-like, list
        The data and labels. TODO: return list or dict.
    """
    if "labels" not in dataset_dict:
        num_cases = None
        num_controls = None
        data_dict = {}
        for c in range(1, chromosomes + 1):
            try:
                cases = dataset_dict[c]["cases"]
                controls = dataset_dict[c]["controls"]
                assert have_same_SNP_order(cases, controls),\
                    "Cases and controls have different SNPs."

                cases_data = pull_gen_data(cases)
                controls_data = pull_gen_data(controls)

                if num_cases is None:
                    num_cases = cases_data.shape[0]
                if num_controls is None:
                    num_controls = controls_data.shape[0]
                assert cases_data.shape[0] == num_cases,\
                    "Cases data has inconsistent subjects (%d vs %d)"\
                    % (cases_data.shape[0], num_cases)
                assert controls_data.shape[0] == num_controls,\
                    "Control data has inconsistent subjects (%d vs %d)"\
                    % (control_data.shape[0], num_controls)
                assert cases_data.shape[1] == controls_data.shape[1],\
                    "Cases and controls have difference number of columns (%d vs %d)."\
                    % (cases_data.shape[1], controls_data.shape[1])
                data_dict[c] = np.concatenate((controls_data, cases_data), axis=0)

            except AssertionError as e:
                raise ValueError("Pulling data from dataset chromosome %d failed (%s)" % (c, e))
        labels = [0] * num_controls + [1] * num_cases
    else:
        data_dict = {}
        for c in range(1, chromosomes + 1):
            if "tped" in dataset_dict[c]:
                chr_data = pull_tped_data(dataset_dict[c]["tped"])
            else:
                chr_data = pull_haps_data(dataset_dict[c]["haps"])
            data_dict[c] = chr_data
        labels = dataset_dict["labels"]

    if shuffle:
        logger.info("Shuffling data")
        idx = range(len(labels))
        random.shuffle(idx)
        labels = [labels[i] for i in idx]
        for key in data_dict:
            data_dict[key] = data_dict[key][idx]

    return data_dict, labels

def pull_haps_data(haps_dict, reference_names=None):
    """
    Pull data from a haps dictionary.

    Parameters
    ----------
    haps_dict: dict
        A haps dictionary (see read_haps_line)
    reference_names: list, optional
        List of SNP names to use as reference.

    Returns
    -------
    data: array-like
    """
    logger.info("Getting haps data.")
    samples = haps_dict[haps_dict.keys()[0]]["values"].shape[0]

    if reference_names == None:
        reference_names = [k for k in haps_dict.keys() if k != "ext"]

    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    reference_names = sorted(list(reference_names))
    for i, SNP_name in enumerate(reference_names):
        assert SNP_name != "ext"
        data[:, i] = haps_dict[SNP_name]["values"]

    return data

def pull_tped_data(tped_dict, reference_names=None):
    """
    Pull data from a tped dictionary.

    Parameters
    ----------
    tped_dict: dict
        A tped dictionary (see read_tped_line)
    reference_names: list, optional
        List of SNP names to use as reference.

    Returns
    -------
    data: array-like
    """
    logger.info("Getting tped data.")
    samples = len(tped_dict[tped_dict.keys()[0]]["values"])

    if reference_names == None:
        reference_names = [k for k in tped_dict.keys() if k != "ext"]

    reference_names = sorted(list(reference_names))
    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    for i, SNP_name in enumerate(reference_names):
        assert SNP_name != "ext"
        minor, major = [tped_dict[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        values = tped_dict[SNP_name]["values"]

        for j, value in enumerate(values):
            x, y = tuple(0 if v == minor else 1 for v in value)
            data[j, i] = x + y

    return data

def pull_gen_data(gen_dict, reference_names=None):
    """
    Pull data from a gen dictionary.

    Parameters
    ----------
    gen_dict: dict
        A gen dictionary (see read_gen_line)
    reference_names: list, optional
        List of SNP names to use as reference.

    Returns
    -------
    data: array-like
    """
    logger.info("Getting gen data")

    if reference_names is None:
        reference_names = [k for k in gen_dict.keys() if k != "ext"]
    samples = len(gen_dict[reference_names[0]]["values"])

    reference_names = sorted(list(reference_names))
    data = np.zeros((samples, len(reference_names)), dtype=np.int8)
    for i, SNP_name in enumerate(reference_names):
        assert SNP_name != "ext"
        data[:, i] = gen_dict[SNP_name]["values"]

    return data

def pull_data(file_dict, reference_names=None):
    """
    Pull data from a arbitrary file dictionary.

    Parameters
    ----------
    file_dict: dict
        A file dictionary. Must be "haps", "tped", or "gen".
    reference_names: list, optional
        List of SNP names to use as reference.

    Returns
    -------
    data: array-like
    """
    if reference_names is not None:
        assert "ext" not in reference_names
    ext = file_dict["ext"]
    if ext == "gen":
        return pull_gen_data(file_dict, reference_names)
    elif ext == "haps":
        return pull_haps_data(file_dict, reference_names)
    elif ext == "tped":
        return pull_tped_data(file_dict, reference_names)
    else:
        raise ValueError("Cannot pull data from extension %s" % ext)

def check_directory(dir_dict):
    """
    Directory checker.
    Makes sure directory is consistent with expectations.

    Paramters
    ---------
    dir_dict: dict
        Directory dictionary to be checked.
    """
    if "haps" in dir_dict:
        assert "tped" in dir_dict
        reference_names = [k for k in dir_dict["tped"].keys() if k != "ext"]
        haps_data = pull_data(dir_dict["haps"], reference_names=reference_names)
        tped_data = pull_data(dir_dict["tped"], reference_names=reference_names)
        assert np.all(haps_data == tped_data), "%r\n%r" % (haps_data.shape, tped_data.shape)
        logger.info("haps and tped have the same data.")

def compare_SNPs(file_A, file_B):
    """
    Compares the SNPs from two files or dictionaries.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.
    """
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
    """
    Checks if file_A is a compatible reference for file_B.
    Compatible references have SNP names which are a strict subset of the other.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.

    Returns
    -------
    compatible: bool
    """
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
    if dict_A["ext"] == "gen":
        #Gen files are never a reference.
        return False
    SNPs_A = set(dict_A.keys())
    SNPs_B = set(dict_B.keys())
    A_not_in_B = SNPs_A.difference(SNPs_B)
    compatible = len(A_not_in_B) == 0
    return compatible

def set_A_with_B(file_A, file_B, nofill=False):
    """
    Sets A with SNPs from B and fills missing SNPs from B using priors from B.
    Uses B priors to randomly set.
    Note: tped dict_A will raise a ValueError.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.

    Returns
    -------
    dict_A, dict
        Dicitonary of A with filled SNPs.
    """
    logger.info("Setting A with SNPs from B.")
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

    SNPs_A = set([k for k in dict_A.keys() if k != "ext"])
    SNPs_B = set([k for k in dict_B.keys() if k != "ext"])
    b_not_in_a = SNPs_B.difference(SNPs_A)

    if dict_A["ext"] in ["haps", "gen"]:
        value_shape = dict_A[list(SNPs_A)[0]]["values"].shape
    else:
        value_shape = len(dict_A[list(SNPs_A)[0]]["values"])

    new_dict_A = {}
    new_dict_A["ext"] = dict_A["ext"]
    for SNP_name in list(SNPs_A):
        if SNP_name in SNPs_B:
            new_dict_A[SNP_name] = dict_A[SNP_name]

    if nofill:
        SNPs_A = set([k for k in new_dict_A.keys() if k != "ext"])
        a_not_in_b = SNPs_A.difference(SNPs_B)
        assert len(a_not_in_b) == 0
        return new_dict_A

    logger.info("Filling with %d SNPs" % len(b_not_in_a))

    if dict_A["ext"] == "gen":
        #Gen files can't be filled right now.
        return dict_A

    for SNP_name in list(b_not_in_a):
        assert SNP_name not in new_dict_A
        B_values = dict_B[SNP_name]["values"]
        if dict_B["ext"] in ["haps", "gen"]:
            B_priors = [(B_values == i).sum(0) * 1. / B_values.shape[0] for i in range(3)]
        elif dict_B["ext"] == "tped":
            minor_allele, major_allele = (dict_B[SNP_name]["minor_allele"],
                                          dict_B[SNP_name]["major_allele"])
            allele_pairs = [[(minor_allele, minor_allele)],
                            [(minor_allele, major_allele), (major_allele, minor_allele)],
                            [(major_allele, major_allele)]]
            B_priors = np.array([sum([1 for b in B_values if b in pairs]) * 1. / len(B_values)
                                 for pairs in allele_pairs])
        else:
            raise ValueError("extension %s not supported" % B_dict["ext"])
        assert abs(B_priors.sum() - 1) < 10e-5, B_priors.sum()

        new_dict_A[SNP_name] = copy.copy(dict_B[SNP_name])
        if new_dict_A["ext"] in ["haps", "gen"]:
            new_dict_A[SNP_name]["values"] = np.random.choice(range(3), size=value_shape[0],
                                                              p=B_priors)
        elif new_dict_A["ext"] == "tped":
            minor_allele, major_allele = (dict_B[SNP_name]["minor_allele"],
                                          dict_B[SNP_name]["major_allele"])
            allele_pairs = [(minor_allele, minor_allele),
                            (minor_allele, major_allele),
                            (major_allele, major_allele)]
            values = np.random.choice(range(3),
                                      size=value_shape,
                                      p=B_priors)
            new_dict_A[SNP_name]["values"] = [allele_pairs[i] for i in values]

    SNPs_A = set([k for k in new_dict_A.keys() if k != "ext"])
    b_not_in_a = SNPs_B.difference(SNPs_A)
    a_not_in_b = SNPs_A.difference(SNPs_B)
    assert len(b_not_in_a) == 0
    assert len(a_not_in_b) == 0
    logger.info("Setting completed.")
    return new_dict_A

def A_has_similar_priors_to_B(file_A, file_B):
    """
    Checks if file_A and file_B have similar priors.
    Priors are similar if for P(i)_j for i in range(3) are within 15% for 95% of the SNPs j.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.

    Returns
    -------
    similar: bool
    """
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

    reference_names = [k for k in dict_A.keys() if (k != "ext") and (k in dict_B.keys())]

    data_A = pull_data(dict_A, reference_names=reference_names)
    data_B = pull_data(dict_B, reference_names=reference_names)
    priors_A = [(data_A == i).sum(0) * 1. / data_A.shape[0] for i in range(3)]
    priors_B = [(data_B == i).sum(0) * 1. / data_B.shape[0] for i in range(3)]
    assert np.allclose(priors_A[0] + priors_A[1] + priors_A[2], np.zeros(priors_A[0].shape) + 1)
    assert np.allclose(priors_B[0] + priors_B[1] + priors_B[2], np.zeros(priors_B[0].shape) + 1)
    similar = True
    for i, (prior_A, prior_B) in enumerate(zip(priors_A, priors_B)):
        percent_off = (
            len(np.where(prior_A - prior_B > 0.15)[0].tolist()) * 1. / prior_A.shape[0])
        percent_reversed = (len(np.where(
                    np.logical_and(
                        prior_A - priors_B[-(i - 1) + 1] <= 0.15,
                        prior_A - prior_B > 0.15))[0].tolist()) * 1. /prior_A.shape[0])
        if percent_off > .05:
            logger.warn("Priors P(%d) not close: %.2f%% off by 15%% or more"
                        % (i, percent_off * 100))
            logger.warn("Priors P(%d) reversed for %.2f%% of SNPs"
                        % (i, percent_reversed * 100))
            similar = False
        else:
            logger.info("Priors P(%s) close: %.2f%% off by 15%% or more"
                        % (i, percent_off * 100))
    return similar

def A_is_aligned_to_B(file_A, file_B):
    """
    Checks if file_A and file_B are aligned.
    Files or dictionaries are aligned if one is a compatible
    reference for the other and the major / minor specifications are the same.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.

    Returns
    -------
    aligned: bool
    """
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

    reference_keys = [k for k in dict_A.keys() if (k != "ext") and (k in dict_B.keys())]

    not_aligned = 0
    for key in reference_keys:
        if key == "ext":
            continue
        if dict_A[key]["minor_allele"] != dict_B[key]["minor_allele"]:
            #logger.info("Alleles for %s not aligned (%s, %s) vs (%s, %s)"
            #            % (key, dict_A[key]["minor_allele"], dict_A[key]["major_allele"],
            #               dict_B[key]["minor_allele"], dict_B[key]["major_allele"]))
            not_aligned += 1
    logger.info("%d alleles for A and B are not aligned" % not_aligned)
    aligned = not_aligned == 0
    return aligned

def align_A_to_B(file_A, file_B):
    """
    Align two files or dictionaries.
    B must be a compatible reference of A.
    TODO: remove this restriction and add filling possibly.

    Parameters
    ----------
    file_A: str or dict
        File path or dictionary.
    file_B: str or dict
        File path or dictionary.

    Returns
    -------
    dict_A: dict
        An aligned dictionary for A.
    """
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
    if dict_A["ext"] == "gen":
        #gen files can't be aligned, as they have no minor/major reference.
        return dict_A

    reference_names = [k for k in dict_A.keys() if (k != "ext") and (k in dict_B.keys())]

    for SNP_name in reference_names:
        minor_A, major_A = [dict_A[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        minor_B, major_B = [dict_B[SNP_name][m] for m in ["minor_allele", "major_allele"]]
        if minor_A == minor_B and major_A == major_B:
            pass
        elif minor_A == major_B and major_A == minor_B:
            if dict_A["ext"] in ["haps", "gen"]:
                dict_A[SNP_name]["values"] = (-(dict_A[SNP_name]["values"] - 1)) + 1
            dict_A[SNP_name]["minor_allele"] = dict_B[SNP_name]["minor_allele"]
            dict_A[SNP_name]["major_allele"] = dict_B[SNP_name]["major_allele"]
            assert dict_A[SNP_name]["minor_allele"] == dict_B[SNP_name]["minor_allele"]
        else:
            raise ValueError()
    assert A_is_aligned_to_B(dict_A, dict_B)
    logger.info("Alignment finished.")
    return dict_A

def check_flipping(haps_dict, gen_dict):
    """
    Checks if haps a gen files flip consistently.
    Right now they don't, so do not use.
    """
    reference_names = [k for k in gen_dict.keys() if k != "ext"]
    for SNP_name in reference_names:
        minor, major = haps_dict[SNP_name]["minor"], haps_dict[SNP_name]["major"]
        assert (minor, major) in [(1, 2), (2, 1)]
        if (minor, major) == (2, 1):
            assert haps_dict[SNP_name]["minor_allele"] == gen_dict[SNP_name]["major_allele"]
            assert haps_dict[SNP_name]["major_allele"] == gen_dict[SNP_name]["minor_allele"]

def have_same_SNP_order(dict_A, dict_B):
    """
    Checks if two dictionaries have the same SNP order.
    """
    have_same_order = [k for k in dict_A.keys() if k != "ext"] == [k for k in dict_B.keys() if k != "ext"]
    return have_same_order

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
    extract_parser.add_argument("out_dir", help="Output directory for Pylearn2 data.")
    extract_parser.add_argument("-c", "--chromosomes", default=22,
                                type=int, help="Number of chromosomes to process.")
    extract_parser.add_argument("-u", "--use_snps", default=None,
                                help="Reference dataset to use SNPs from.")
    extract_parser.add_argument("-a", "--align_to", default=None)
    extract_parser.add_argument("--nofill", action="store_true")

    separate_parser = subparsers.add_parser("separate", help="Separate haps for GWA sim")
    separate_parser.set_defaults(which="separate")
    separate_parser.add_argument("chr_dir")
    separate_parser.add_argument("labels")
    separate_parser.add_argument("-s", "--separate_info", action="store_true")
    separate_parser.add_argument("-t", "--transposed", action="store_true")

    return parser

def save_snp_names(out_file, snp_list):
    with open(out_file, "w") as f:
        for snp in snp_list:
            f.write("%s\n" % snp)

def write_haps_file(haps, out_file, omit_info=False, info_only=False, transposed=False):
    with open(out_file, "w") as f:
        lines = []
        for SNP_name in sorted(haps.keys()):
            if SNP_name == "ext":
                continue
            lines.append(write_haps_line(SNP_name, haps[SNP_name],
                                         omit_info=omit_info,
                                         info_only=info_only))
        if transposed:
            logger.info("Transposing file")
            lines = [l.split(" ") for l in lines]
            lines = map(list, zip(*lines))
            lines = [" ".join(l) for l in lines]
        logger.info("Writing file")
        for line in lines:
            f.write(line + "\n")

def split_haps(chr_dict, labels, separate_info=False, transposed=False):
    """
    Splits haps files into 2 and saves them.
    """
    if not "haps" in chr_dict:
        raise ValueError()
    controls_idx = [i for i, j in enumerate(labels) if j == 0]
    cases_idx = [i for i, j in enumerate(labels) if j == 1]

    def split(haps, idx):
        new_haps = copy.deepcopy(chr_dict["haps"])
        for SNP_name in new_haps:
            if SNP_name == "ext":
                continue
            new_haps[SNP_name]["values"] = [new_haps[SNP_name]["values"][i] for i in idx]
            new_haps[SNP_name]["raw_values"] = [new_haps[SNP_name]["raw_values"][i]
                                                for i in idx for _ in range(2)]
        return new_haps

    controls_haps = split(chr_dict["haps"], controls_idx)
    cases_haps = split(chr_dict["haps"], cases_idx)

    try:
        prefix = re.findall(r'chr\d+', chr_dict["directory"])[0] + "_"
    except IndexError:
        prefix = ""
    if transposed:
        transposed_prefix = "transposed_"
    else:
        transposed_prefix = ""

    write_haps_file(controls_haps,
                    path.join(chr_dict["directory"],
                              prefix + transposed_prefix + "input_controls.haps"),
                    omit_info=separate_info, transposed=transposed)
    write_haps_file(cases_haps,
                    path.join(chr_dict["directory"],
                              prefix + transposed_prefix + "input_cases.haps"),
                    omit_info=separate_info, transposed=transposed)
    write_haps_file(chr_dict["haps"], path.join(chr_dict["directory"],
                                                prefix.translate(None, "_") + ".info"),
                    info_only=separate_info)

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    def get_dict(dir_dict):
        if "tped" in dir_dict:
            return dir_dict["tped"]
        elif "haps" in dir_dict:
            return dir_dict["haps"]
        elif "cases" in dir_dict:
            return dir_dict["cases"]
        else:
            raise ValueError("No haps or cases in %r." % dir_dict.keys())

    if args.which == "separate":
        chr_dict = read_chr_directory(args.chr_dir)
        labels = parse_labels_file(args.labels)
        split_haps(chr_dict, labels, args.separate_info, args.transposed)

    elif args.which == "compare":
        dir_dict_1 = read_chr_directory(args.dir_1)
        dir_dict_2 = read_chr_directory(args.dir_2)

        dict_1 = get_dict(dir_dict_1)
        dict_2 = get_dict(dir_dict_2)

        if have_same_SNP_order(dict_1, dict_2):
            print "files have the same SNP order."
        else:
            print "files do not have the same SNP order."

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
            dict_1 = set_A_with_B(dict_1, dict_2)
            dict_1 = align_A_to_B(dict_1, dict_2)

        if A_has_similar_priors_to_B(dict_1, dict_2):
            print "A and B have similar priors."
        else:
            print "A and B do not have similar priors."

    elif args.which == "extract":
        data_dict, snp_ref_data_dict, ref_data_dict = read_dataset_directory(
            args.directory,
            chromosomes=args.chromosomes,
            snps_reference=args.use_snps,
            align_reference=args.align_to,
            nofill=args.nofill)

        if snp_ref_data_dict is not None:
            for key in data_dict:
                if key == "labels":
                    continue
                data_chr_dict = get_dict(data_dict[key])
                snp_ref_chr_dict = get_dict(snp_ref_data_dict[key])

                if args.nofill:
                    data_snps = set([k for k in data_chr_dict if k != "ext"])
                    snp_ref_snps = set([k for k in snp_ref_chr_dict if k != "ext"])
                    in_data_not_in_ref = data_snps - snp_ref_snps
                    assert len(in_data_not_in_ref) == 0, len(in_data_not_in_ref)
                    logger.info("Data now a strict subset of reference with %d SNPs" % len(data_snps))

                else:
                    assert have_same_SNP_order(data_chr_dict, snp_ref_chr_dict)
                    logger.info("Data now the same set of SNPs as reference")

        def get_ext(d):
            if "controls" in d:
                return "controls"
            elif "tped" in d:
                return "tped"
            elif "haps" in d:
                return "haps"
            else:
                raise ValueError(d.keys())

        if ref_data_dict is not None:
            for key in data_dict:
                if key == "labels":
                    continue
                ext1 = get_ext(data_dict[key])
                ext2 = get_ext(ref_data_dict[key])

                assert A_has_similar_priors_to_B(data_dict[key][ext1],
                                                 ref_data_dict[key][ext2])
        data, labels = pull_dataset(data_dict, chromosomes=args.chromosomes)
        out_dir = serial.preprocess("${PYLEARN2_NI_PATH}/" + args.out_dir)
        assert path.isdir(out_dir), out_dir
        logger.info("Saving labels to %s" % out_dir)
        np.save(path.join(out_dir, "labels.npy"), labels)
        for c in data_dict:
            if c == "labels":
                continue
            logger.info("Saving chromosome %d to %s" %
                        (c, path.join(out_dir, "chr%d.npy" % c)))
            np.save(path.join(out_dir, "chr%d.npy" % c), data[c])

            save_snp_names(path.join(out_dir, "chr%d.snps" % c),
                           [k for k in data_dict[c][get_ext(data_dict[c])]
                            if k != "ext"])
