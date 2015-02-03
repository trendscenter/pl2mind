"""
Module for managing experiments.
"""

import matplotlib
matplotlib.use("Agg")

import copy
import datetime
import glob
from jobman.tools import expand
from jobman.tools import flatten
import json
import Levenshtein
import logging
import multiprocessing as mp
import numpy as np
import os
from os import path
from pl2mind.datasets import MRI
import psutil

from pylearn2.config import yaml_parse
from pylearn2.scripts.jobman.experiment import ydict
from pylearn2 import monitor
from pylearn2.utils import serial

import signal
import socket
import time
import zmq


class MRIInputHandler(object):
    """
    Input handler for MRI data.
    This is an object in the case of loading multiple experiments with the same
    data parameters.
    TODO: clean up.
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
    processing_flag: mp.Value
        Shared boolean indicating whether the processing worker is active.
    mem: mp.Value
        Shared float indicating moving average of memory percentage.
    cpu: mp.Value
        Shared float of cpu percentage for this process.
    port: int
        Port the socket is listening on.
    last_processed: mp.Value
        Shared float when model was last processed
    """

    def __init__(self, experiment, hyperparams,
                 out_path, pid, processing_flag,
                 mem, cpu, port, last_processed):
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
            "outputs": dict((o, "%s.pdf" % o)
                for o in experiment.outputs),
            "stats": {
                "status": "STARTING",
                "pid": self.pid,
                "user": p.username,
                "host": socket.gethostname(),
                "create_time": datetime.datetime.fromtimestamp(
                    p.create_time).strftime("%Y-%m-%d %H:%M:%S"),
                "last_heard": datetime.datetime.fromtimestamp(
                    p.create_time).strftime("%Y-%m-%d %H:%M:%S"),
                "port": port
                },
            "processing": False,
            "last_processed": "Never",
            "hyperparams": hyperparams,
            "logs": {
                "cpu": {"cpu": []},
                "mem": {"mem": []}
                }
            }

    def finish(self, status):
        """
        Method to call on quit.

        Parameters
        ----------
        status: str
            Status at which model terminated.
        """
        self.d["stats"]["status"] = status
        self.d["stats"]["stopped_time"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")
        self.write_json()

    def get_stats(self):
        """
        Get process statistics.
        Process stats are handled by a StatProcessor worker.
        """
        self.d["stats"]["status"] = "RUNNING (afaik)"
        p = psutil.Process(self.pid)
        self.d["processing"] = bool(self.processing_flag.value)
        self.d["last_processed"] = self.last_processed.valie
        stats = {
            "cpu": self.cpu.value,
            "mem": self.mem.value,
            "user": p.username
        }
        self.d["stats"]["last_heard"] = datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")

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

        Parameters
        ----------
        channel: str
            The channel
        value: float
            The value at the channel
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

        Parameters
        ----------
        message: str
            The message passed into the Monitor.log object.
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


class ModelProcessor(mp.Process):
    """
    Processor for model analysis.

    Parameters
    ----------
    experiment: pl2mind.experiment module
        Experiment
    checkpoint: str
        Location of the checkpoint file
    ep: mp.Pipe endpoint
        Endpoint of pipe to socket listener.
        This should be a signal originating from web server.
    flag: mp.Value
        Shared boolean to indicate to log handler that processor is running.
    """
    def __init__(self, experiment, checkpoint, ep, flag, last_processed):
        self.__dict__.update(locals())
        self.socket = None

        self.out_path = "/".join(checkpoint.split("/")[:-1])
        self.best_checkpoint = path.join(self.out_path,
                                         checkpoint.split(
                                            "/")[-1].split(
                                            ".")[0] + "_best.pkl")
        self.persistent = False
        super(ModelProcessor, self).__init__()

    def run(self):
        while True:
            message = self.ep.recv()
            if message is None:
                return
            print("Processing")
            self.flag.value = True
            try:
                self.experiment.analyze_fn(self.best_checkpoint, self.out_path)
                self.ep.send("SUCCESS")
                self.last_processed.value = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            except IOError:
                self.experiment.analyze_fn(self.checkpoint, self.out_path)
                self.ep.send("SUCCESS")
                self.last_processed.value = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                print (e)
                self.ep.send("FAILED")
            self.flag.value = False


class StatProcessor(mp.Process):
    """
    Worker for getting process stats.

    Parameters
    ----------
    pid: int
        Process id
    mem: mp.Value
        Shared integer with log handler.
    cpu: mp.Value
        Shared integer with log handler.
    """
    def __init__(self, pid, mem, cpu):
        self.__dict__.update(locals())
        self.cpus = []
        self.mems = []
        super(StatProcessor, self).__init__()

    def run(self):
        p = psutil.Process(self.pid);
        while True:
            time.sleep(1);
            if len(self.cpus) == 100:
                self.cpus.pop(0)
            elif len(self.cpus) > 100:
                raise ValueError()
            self.cpus.append(p.get_cpu_percent())

            self.cpu.value = np.mean(self.cpus)

            if len(self.mems) == 100:
                self.mems.pop(0)
            elif len(self.mems) > 100:
                raise ValueError()
            self.mems.append(p.get_memory_percent())
            self.mem.value = np.mean(self.mems)


def server(pid, ep):
    """
    Worker for listening socket for communication with web server.

    Parameters
    ----------
    pid: int
        Process id. Used to send SIGINT.
    ep: mp.Pipe endpoint.
        Used to communicate with model processor.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = socket.bind_to_random_port("tcp://0.0.0.0")
    ep.send(port)
    while True:
        message = socket.recv()
        if message == "KILL":
            socket.send("OK")
            socket.close()
            ep.send(None)
            p = psutil.Process(pid)
            p.send_signal(signal.SIGINT)
            return
        elif message == "PROCESS":
            ep.send(True)
            message = ep.recv()
            socket.send(message)

def set_hyper_parameters(hyper_parameters, **kwargs):
    """
    Sets hyper-parameters from kwargs.

    Parameters
    ----------
    hyper_parameters: dict
        Dictionary of hyperparameters
    **kwargs: dict
        Dictionary of fill values. Keys in the form <key_1>.<key_2>...<key_n>
        fill value of hyper_parameters[key_1][key_2]...[key_n].
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

    # Fill the hyperparameter values.
    hyper_parameters = experiment.default_hyperparams
    set_hyper_parameters(hyper_parameters, **kwargs)
    file_parameters = experiment.fileparams
    set_hyper_parameters(file_parameters, **kwargs)
    hyper_parameters.update(file_parameters)

    # Use the input hander to get input information.
    ih = MRIInputHandler()
    input_dim, variance_map_file = ih.get_input_params(hyper_parameters)
    if hyper_parameters["nvis"] is None:
        hyper_parameters["nvis"] = input_dim

    # Corruptor is a special case of hyper parameters that depends on input
    # file: variance_map. So we hack it in here.
    if "corruptor" in hyper_parameters.keys():
        if "variance_map" in hyper_parameters["corruptor"].keys():
            hyper_parameters["corruptor"]["variance_map"] =\
            "!pkl: %s" % variance_map_file
    else:
        hyper_parameters["variance_map_file"] = variance_map_file

    # The Process id
    pid = os.getpid()

    # Set the output path, default from environment variable $PYLEARN2_OUTS
    out_path = serial.preprocess(
        hyper_parameters.get("out_path", "${PYLEARN2_OUTS}"))
    if not path.isdir(out_path):
        os.mkdir(out_path)

    # If any pdfs are in out_path, kill or quit
    if len(glob.glob(path.join(out_path, "*.pdf"))) > 0:
        print ("Results found in %s "
               "Proceeding will erase." % out_path)
        command = None
        while not command in ["yes", "no", "y", "n"]:
            command = raw_input("%s: " % "Proceed?")
            if command in ["yes", "y"]:
                break
            elif command in ["no", "n"]:
                exit()
            else:
                print ("Please enter yes(y) or no(n)")
        for pdf in glob.glob(path.join(out_path, "*.pdf")):
            print "Removing %s" % pdf
            os.remove(pdf)

    # Get the train object
    hyper_parameters = expand(flatten(hyper_parameters), dict_type=ydict)
    yaml_template = open(experiment.yaml_file).read()
    yaml = yaml_template % hyper_parameters
    train_object = yaml_parse.load(yaml)

    # Set up subprocesses and log handler.
    mp_ep, s_ep = mp.Pipe()
    p = mp.Process(target=server, args=(pid, s_ep))
    p.start()
    port = mp_ep.recv()
    print "Listening on port %d" % port

    processing_flag = mp.Value("b", False)
    mem = mp.Value("f", 0.0)
    cpu = mp.Value("f", 0.0)
    last_processed = mp.Value("s", "Never")

    lh = LogHandler(experiment, hyper_parameters, out_path,
                    pid, processing_flag, mem, cpu, port, last_processed)
    h = logging.StreamHandler(lh)
    monitor.log.addHandler(h)

    model_processor = ModelProcessor(experiment, train_object.save_path,
                                     mp_ep, processing_flag, last_processed)
    model_processor.start()
    stat_processor = StatProcessor(pid, mem, cpu)
    stat_processor.start()

    # A signal handler so processes kill cleanly.
    def signal_handler(signum, frame):
        print("Quitting...")
        lh.finish("KILLED")
        p.terminate()
        stat_processor.terminate()
        model_processor.terminate()
        exit()

    signal.signal(signal.SIGINT, signal_handler)

    # Main loop.
    train_object.main_loop()
    lh.finish("COMPLETED")
    p.terminate()
    model_processor.terminate()
    stat_processor.terminate()

    # After complete, process model.
    try:
        self.experiment.analyze_fn(model_processor.best_checkpoint,
                                   model_processor.out_path)
        os.remove(model_processor.best_checkpoint)
    except IOError:
        experiment.analyze_fn(model_processor.checkpoint,
                              model_processor.out_path)
    except Exception as e:
        print (e)

    # Clean checkpoints.
    # TODO(dhjelm): give option to keep checkpoints somewhere.
    try:
        os.remove(model_processor.best_checkpoint)
    except:
        pass
    try:
        os.remove(model_processor.checkpoint)
    except:
        pass
