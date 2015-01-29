"""
Module for managing experiments.
"""

from jobman.tools import expand
from jobman.tools import flatten

import copy
import datetime
import json
import Levenshtein
import logging
import numpy as np
import os
from os import path

from pl2mind.datasets import MRI

import psutil

from pylearn2.config import yaml_parse
from pylearn2.scripts.jobman.experiment import ydict
from pylearn2 import monitor
from pylearn2.utils import serial

import socket
import time


class MRIInputHandler(object):
    """
    Input handler for MRI data.
    This is an object in the case of loading multiple experiments with the same
    data parameters.
    """
    def __init__(self):
        self.d = {}

    def get_input_params(self, hyperparams):
        """
        Get the input parameters given data hyperparameters.

        Parameters
        ----------
        hyperparams: dict
            Hyperparameters.

        Returns
        -------
        input_dim, variance_map_file: int, str
            Input dimensionality and the location of the variance map file.
        """

        dataset_name = hyperparams["dataset_name"]
        data_class = hyperparams["data_class"]
        variance_normalize = hyperparams.get("variance_normalize", False)
        unit_normalize = hyperparams.get("unit_normalize", False)
        demean = hyperparams.get("demean", False)
        assert not (variance_normalize and unit_normalize)

        data_path = serial.preprocess("${PYLEARN2_NI_PATH}/" + dataset_name)

        h = hash((data_class, variance_normalize, unit_normalize, demean))

        if self.d.get(h, False):
            return self.d[h]
        else:
            if data_class == "MRI_Transposed":
                assert not variance_normalize
                mri = MRI.MRI_Transposed(dataset_name=dataset_name,
                                         unit_normalize=unit_normalize,
                                         demean=demean,
                                         even_input=True,
                                         apply_mask=True)
                input_dim = mri.X.shape[1]
                variance_file_name = ("variance_map_transposed%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_dm" if demean else ""))

            elif data_class == "MRI_Standard":
                assert not demean
                mask_file = path.join(data_path, "mask.npy")
                mask = np.load(mask_file)
                input_dim = (mask == 1).sum()
                if input_dim % 2 == 1:
                    input_dim -= 1
                mri = MRI.MRI_Standard(which_set="full",
                                       dataset_name=dataset_name,
                                       unit_normalize=unit_normalize,
                                       variance_normalize=variance_normalize,
                                       even_input=True,
                                       apply_mask=True)
                variance_file_name = ("variance_map%s%s.npy"
                                      % ("_un" if unit_normalize else "",
                                         "_vn" if variance_normalize else ""))

        variance_map_file = path.join(data_path, variance_file_name)
        if not path.isfile(variance_map_file):
            mri_nifti.save_variance_map(mri, variance_map_file)
        self.d[h] = (input_dim, variance_map_file)
        return self.d[h]


class LogHandler(object):
    """
    Class that hijacks the Pylearn2 monitor logs and outputs json.
    Saves to out_path/model.json.

    Parameters
    ----------
    experiment: module
        Experiment module
    hyperparams: dict
        Dictionary of hyperparameters
    out_path: str
        Output path directory for the json.
    pid: int
        Process id
    """

    def __init__(self, experiment, hyperparams, out_path, pid):
        self.__dict__.update(locals())
        self.on = False
        self.channels = []
        self.collect_channels = False
        self.channel_groups = {}

        p = psutil.Process(self.pid)
        self.d = {
            "name": experiment.name,
            "results_of_interest": experiment.results_of_interest +\
                ["cpu", "mem"],
            "stats": {
                "status": "STARTING",
                "pid": self.pid,
                "user": p.username,
                "host": socket.gethostname(),
                "create_time": datetime.datetime.fromtimestamp(
                    p.create_time).strftime("%Y-%m-%d %H:%M:%S")
                },
            "hyperparams": hyperparams,
            "logs": {
                "cpu": {"cpu": []},
                "mem": {"mem": []}
                }
            }

    def finish(self, status):
        """
        Method to call on quit.
        """
        self.d["stats"]["status"] = status
        self.d["stats"]["stopped_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        self.write_json()

    def get_stats(self):
        """
        Get process statistics.
        """
        self.d["stats"]["status"] = "RUNNING (afaik)"
        p = psutil.Process(self.pid)
        stats = {
            "cpu": p.get_cpu_percent(),
            "mem": p.get_memory_percent(),
            "user": p.username
        }

        for key, value in stats.iteritems():
            if key in self.d["logs"].keys():
                self.d["logs"][key][key].append(value)

    def compile_channels(self):
        """
        Compiles the list of channels found.
        This will attempt to group channels by edit distance.
        """
        group_name_omits = ["train_", "valid_", "test_"]
        edit_thresh = 8
        for channel in self.channels:
            edit_distances = dict((c, Levenshtein.distance(channel, c))
                              for c in self.channel_groups.keys())
            if len(edit_distances) == 0:
                group_name = channel
                for omit in group_name_omits:
                    group_name = group_name.replace(omit, "")
                self.channel_groups[group_name] = [channel]
            else:
                group = None
                min_ed = len(channel)
                for c, d in edit_distances.iteritems():
                    if d <= min_ed:
                        min_ed = d
                        group = c
                if min_ed > edit_thresh or group is None:
                    group_name = channel
                    for omit in group_name_omits:
                        group_name = group_name.replace(omit, "")
                    self.channel_groups[group_name] = [channel]
                else:
                    # Now we reduce the group to the minimum shared string
                    # mb = matching blocks (see Levenshtein docs).
                    mb =\
                        Levenshtein.matching_blocks(
                        Levenshtein.editops(channel, group), channel, group)
                    new_group = "".join([group[x[1]:x[1]+x[2]] for x in mb])
                    if new_group != group:
                        self.channel_groups[new_group] =\
                            copy.deepcopy(self.channel_groups[group])
                        self.channel_groups.pop(group)
                    self.channel_groups[new_group].append(channel)
        for group, channels in self.channel_groups.iteritems():
            self.d["logs"][group] = {}
            for channel in channels:
                self.d["logs"][group][channel] = []
        print self.d["logs"]

    def add_value(self, channel, value):
        """
        Adds a value to a channel.
        """
        group = next((group
                      for group, channels in self.channel_groups.iteritems()
                      if channel in channels), None)
        if group is None:
            return
        self.d["logs"][group][channel].append(value)

    def write_json(self):
        """
        Write the json file.
        """
        with open(path.join(self.out_path, "model.json"), "w") as f:
            json.dump(self.d, f)

    def write(self, message):
        """
        SteamHandle call.
        Necessary as a StreamHandle object. Where hijack occures.
        """
        if "Monitored channels" in message:
            self.collect_channels = True
            return
        if "Compiling accum..." in message:
            self.compile_channels()
            self.collect_channels = False
            return
        if "Monitoring step" in message:
            self.on = True
            self.write_json()
            self.get_stats()
            return
        if "Saving to" in message:
            return
        if "Examples seen" in message:
            return
        if "Batches seen" in message:
            return

        if self.collect_channels:
            channel = message.translate(None, "\t\n")
            self.channels.append(channel)
        elif self.on:
            parsed = message.split(":")
            channel = parsed[0].translate(None, "\t\n")
            value = float(parsed[1].translate(None, "\n "))
            self.add_value(channel, value)


def set_hyper_parameters(hyper_parameters, **kwargs):
    """
    Sets hyper-parameters from kwargs.
    """

    if kwargs is not None:
        for key, value in kwargs.iteritems():
            entry = hyper_parameters
            split_keys = key.split(".")
            for k in split_keys[:-1]:
                entry = entry[k]
            if split_keys[-1] in entry:
                entry[split_keys[-1]] = value

def run_experiment(experiment, **kwargs):
    """
    Experiment function.
    Used by jobman to run jobs. Must be loaded externally.
    TODO: add sigint handling.

    Parameters
    ----------
    experiment: module
        Experiment module.
    kwargs: dict
        Typically hyperparameters.
    """

    hyper_parameters = experiment.default_hyperparams()
    set_hyper_parameters(hyper_parameters, **kwargs)
    file_parameters = experiment.fileparams
    set_hyper_parameters(file_parameters, **kwargs)
    hyper_parameters.update(file_parameters)

    ih = MRIInputHandler()
    input_dim, variance_map_file = ih.get_input_params(hyper_parameters)
    hyper_parameters["nvis"] = input_dim
    hyper_parameters["variance_map_file"] = variance_map_file

    pid = os.getpid()
    out_path = serial.preprocess(
        hyper_parameters.get("out_path", "${PYLEARN2_OUTS}"))
    if not path.isdir(out_path):
        os.mkdir(out_path)
    if not path.isdir(path.join(out_path, "logs")):
        os.mkdir(path.join(out_path, "logs"))

    hyper_parameters = expand(flatten(hyper_parameters), dict_type=ydict)

    lh = LogHandler(experiment, hyper_parameters, out_path, pid)
    h = logging.StreamHandler(lh)
    monitor.log.addHandler(h)

    yaml_template = open(experiment.yaml_file).read()
    yaml = yaml_template % hyper_parameters
    train_object = yaml_parse.load(yaml)
    try:
        train_object.main_loop()
        lh.finish("COMPLETED")
    except KeyboardInterrupt:
        print("Quitting...")
        lh.finish("KILLED")
