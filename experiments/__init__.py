"""
Module for managing experiments.
"""

import matplotlib
matplotlib.use("Agg")

import ast
import copy
from cStringIO import StringIO
import datetime
import glob
import imp

from jobman import api0
from jobman.dbi.dbi import DBILocal
from jobman import sql
from jobman.sql_runner import runner_sql
from jobman.tools import DD
from jobman.tools import expand
from jobman.tools import flatten

import json
import Levenshtein
import logging
import multiprocessing as mp
import numpy as np
import optparse
import os
from os import path
from pl2mind.experiments import input_handler
import psutil

from pylearn2.config import yaml_parse
from pylearn2.scripts.jobman.experiment import ydict
from pylearn2 import monitor
from pylearn2.utils import serial

import signal
import socket
import sys
import time
import zmq


class MetaLogHandler(object):
    """
    Handler for module logs.
    Adds logs to LogHandler dictionary d.
    TODO: add stdout and/or stderr.
    """
    def __init__(self, d):
        self.__dict__.update(locals())

    def write(self, message):
        self.d["log_stream"] += message


class LogHandler(object):
    """
    Class that hijacks the Pylearn2 monitor logs and outputs json.
    Saves to out_path/model.json.
    TODO: kwarg all of these arguments.

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
    last_processed: mp.Manager.dict
        Shared float when model was last processed
    dbdescr: str, optional
        Database descritiption string.
    job_id: int, optional
        Job id.
    """

    def __init__(self, experiment, hyperparams,
                 out_path, yaml, pid, processing_flag,
                 mem, cpu, port, last_processed, dbdescr=None, job_id=None):
        self.__dict__.update(locals())
        self.on = False
        self.channels = []
        self.collect_channels = False
        self.channel_groups = {}

        p = psutil.Process(self.pid)
        self.d = {
            "name": experiment.name,
            "yaml": yaml,
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
                },
            "log_stream": ""
            }


        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        h = logging.StreamHandler(MetaLogHandler(self.d))
        self.logger.addHandler(h)

    def update_db(self):
        """
        Update the postgresql database with results.
        """

        if self.dbdescr == None:
            return
        assert self.job_id is not None
        db = api0.open_db(self.dbdescr)
        j = db.get(self.job_id)
        for key, value in self.d["logs"].iteritems():
            if key in ["cpu", "mem"]:
                j[translate_string("stats.%s" % key, "knex")] = value[key][-1]
            else:
                for channel, channel_value in value.iteritems():
                    if len(channel_value) > 0:
                        j[translate_string("results.%s" % channel, "knex")] =\
                            channel_value[-1]
        for stat in self.d["stats"]:
            j[translate_string("stats.%s" % stat, "knex")] =\
                self.d["stats"][stat]
        status = self.d["stats"]["status"]

        if status in ["STARTING", "RUNNING (afaik)"]:
            j["jobman.status"] = 1
        elif status == "KILLED":
            j["jobman.status"] = -1
        elif status == "COMPLETED":
            j["jobman.status"] = 2
        table = self.dbdescr.split("table=",1)[1]
        db.createView("%s" % table)

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
        self.d["logs"]["cpu"]["cpu"].append(0)
        self.d["logs"]["mem"]["mem"].append(0)
        self.write_json()
        self.update_db()

    def get_stats(self):
        """
        Get process statistics.
        Process stats are handled by a StatProcessor worker.
        """
        self.d["stats"]["status"] = "RUNNING (afaik)"

        p = psutil.Process(self.pid)
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

        self.d["processing"] = bool(self.processing_flag.value)
        if self.d["processing"]:
            self.d["last_processed"] = self.last_processed["value"] +\
                " (epoch %d)" % len(self.d["logs"]["cpu"]["cpu"])
        self.update_db()

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
        self.logger.info(self.d["logs"])

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
    def __init__(self, experiment, checkpoint, ep,
                 flag, last_processed, logger):
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
            self.logger.info("Processing")
            self.flag.value = True
            try:
                self.experiment.analyze_fn(self.best_checkpoint, self.out_path)
                self.logger.info("Processing successful")
                self.ep.send("SUCCESS")
                self.last_processed["value"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            except IOError:
                self.experiment.analyze_fn(self.checkpoint, self.out_path)
                self.logger.info("Processing successful")
                self.ep.send("SUCCESS")
                self.last_processed["value"] = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
            except Exception as e:
                self.logger.exception(e)
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
    def __init__(self, pid, mem, cpu, logger, length=100):
        self.__dict__.update(locals())
        self.cpus = []
        self.mems = []
        self.br = False
        super(StatProcessor, self).__init__()

    def run(self):
        p = psutil.Process(self.pid);

        def sig_handler(signum, frame):
            self.br = True
            return

        signal.signal(signal.SIGTERM, sig_handler)

        while not self.br:
            time.sleep(1);
            if len(self.cpus) == self.length:
                self.cpus.pop(0)
            elif len(self.cpus) > self.length:
                raise ValueError()
            self.cpus.append(p.get_cpu_percent())

            self.cpu.value = np.mean(self.cpus)

            if len(self.mems) == self.length:
                self.mems.pop(0)
            elif len(self.mems) > self.length:
                raise ValueError()
            self.mems.append(p.get_memory_percent())
            self.mem.value = np.mean(self.mems)
        self.logger.info("Server is dying (This is good)")


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

    def sig_handler(signum, frame):
        socket.close()
        return

    signal.signal(signal.SIGTERM, sig_handler)

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

def run_experiment(experiment, hyper_parameters=None, ask=True, keep=False,
                   dbdescr=None, job_id=None, **kwargs):
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
    if hyper_parameters is None:
        hyper_parameters = experiment.default_hyperparams

    set_hyper_parameters(hyper_parameters, **kwargs)
    file_parameters = experiment.fileparams
    set_hyper_parameters(file_parameters, **kwargs)
    hyper_parameters.update(file_parameters)

    # HACK TODO: fix this. For some reason knex formatted strings are sometimes
    # Getting in.
    hyper_parameters = translate(hyper_parameters, "pylearn2")

    # Use the input hander to get input information.
    ih = input_handler.MRIInputHandler()
    input_dim, variance_map_file = ih.get_input_params(hyper_parameters)
    if hyper_parameters["nvis"] is None:
        hyper_parameters["nvis"] = input_dim
    # Hack for NICE. Need to rethink inner-dependencies of some model params.
    if ("encoder" in hyper_parameters.keys()
        and "nvis" in hyper_parameters["encoder"].keys()
        and hyper_parameters["encoder"]["nvis"] is None):
        hyper_parameters["encoder"]["nvis"] = input_dim

    # If there's min_lr, make it 1/10 learning_rate
    if "min_lr" in hyper_parameters.keys():
        hyper_parameters["min_lr"] = hyper_parameters["learning_rate"] / 10

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
        os.makedirs(out_path)

    # If any pdfs are in out_path, kill or quit
    if ask and len(glob.glob(path.join(out_path, "*.pdf"))) > 0:
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

    processing_flag = mp.Value("b", False)
    mem = mp.Value("f", 0.0)
    cpu = mp.Value("f", 0.0)
    last_processed = mp.Manager().dict()
    last_processed["value"] = "Never"

    lh = LogHandler(experiment, hyper_parameters, out_path, yaml,
                    pid, processing_flag, mem, cpu, port, last_processed,
                    dbdescr, job_id)
    h = logging.StreamHandler(lh)
    monitor.log.addHandler(h)

    lh.logger.info("Listening on port %d" % port)

    model_processor = ModelProcessor(experiment, train_object.save_path,
                                     mp_ep, processing_flag, last_processed,
                                     lh.logger)
    model_processor.start()
    stat_processor = StatProcessor(pid, mem, cpu, lh.logger)
    stat_processor.start()

    # Clean the model after running
    def clean():
        p.terminate()
        lh.logger.info("waiting for server...")
        p.join()
        model_processor.terminate()
        lh.logger.info("waiting for model processor...")
        model_processor.join()
        stat_processor.terminate()
        lh.logger.info("waiting for stat processor...")
        stat_processor.join()

        if keep:
            lh.logger.info("Keeping checkpoints")
        else:
            lh.logger.info("Cleaning checkpoints")
            try:
                os.remove(model_processor.best_checkpoint)
            except:
                pass
            try:
                os.remove(model_processor.checkpoint)
            except:
                pass

    # A signal handler so processes kill cleanly.
    def signal_handler(signum, frame):
        lh.logger.info("Forced quitting...")
        clean()
        lh.finish("KILLED")
        if dbdescr is None:
            exit()
        else:
            raise ValueError("KILLED")

    signal.signal(signal.SIGINT, signal_handler)

    # Main loop.
    try:
        train_object.main_loop()
    except Exception as e:
        lh.logger.exception(e)
        clean()
        lh.finish("FAILED")
        raise(e)

    # After complete, process model.
    lh.logger.info("Processing...")
    try:
        experiment.analyze_fn(model_processor.best_checkpoint,
                                   model_processor.out_path)
    except IOError:
        experiment.analyze_fn(model_processor.checkpoint,
                              model_processor.out_path)
    except Exception as e:
        lh.logger.error(e)

    # Clean checkpoints.
    clean()
    lh.logger.info("Finished experiment.")
    lh.finish("COMPLETED")
    return

def run_jobman_from_sql(jobargs):
    """
    Runs muliple jobs on postgresql database table.
    """
    dbdescr = ("postgres://%(user)s@%(host)s:"
               "%(port)d/%(database)s?table=%(table)s"
               % {"user": jobargs.user,
                  "host": jobargs.host,
                  "port": jobargs.port,
                  "database": jobargs.database,
                  "table": jobargs.table,
                  })

    command = "/na/homes/dhjelm/Code/Jobman/bin/jobman sql %s ." % dbdescr
    dbi = DBILocal([command] * jobargs.n_proc,
        **dict(log_dir="/export/mialab/users/dhjelm/Experiments/LOGS"))
    dbi.nb_proc = jobargs.n_proc
    dbi.run()

def run_one_jobman(jobargs):
    """
    Run a single job from postgresql database table.
    """
    dbdescr = ("postgres://%(user)s@%(host)s:"
               "%(port)d/%(database)s?table=%(table)s"
               % {"user": jobargs.user,
                  "host": jobargs.host,
                  "port": jobargs.port,
                  "database": jobargs.database,
                  "table": jobargs.table,
                  })
    options = optparse.Values(dict(modules=None,
                                   n=1,
                                   workdir="/na/homes/dhjelm/tmp/jobman/",
                                   finish_up_after=None,
                                   save_every=None))
    expdir = "/export/mialab/users/dhjelm/Experiments/"
    runner_sql(options, dbdescr, expdir)

def jobman_status(jobargs):
    """
    Print status of jobs in postgres database table.
    """
    db = api0.open_db("postgres://%(user)s@%(host)s:"
                      "%(port)d/%(database)s?table=%(table)s"
                      % {"user": jobargs.user,
                         "host": jobargs.host,
                         "port": jobargs.port,
                         "database": jobargs.database,
                         "table": jobargs.table,
                         })
    for job in db.__iter__():
        print "-------------------"
        print "Job: %d" % job.id
        print "Status: %d" % job.status
        for k in job.keys():
            print "\t%s\t%r" % (k, job[k])

def open_db(jobargs):
    dbdescr = ("postgres://%(user)s@%(host)s:"
               "%(port)d/%(database)s?table=%(table)s"
               % {"user": jobargs.user,
                  "host": jobargs.host,
                  "port": jobargs.port,
                  "database": jobargs.database,
                  "table": jobargs.table,
                  })
    db = api0.open_db(dbdescr)
    return db

def set_status_jobman(jobargs):
    """
    Sets status of jobs in postgres database table.
    """

    db = open_db(jobargs)
    job_id = jobargs.job_id
    if job_id.isdigit():
        job = db.get(job_id)
        job["jobman.status"] = 0
    elif job_id == "ALL":
        for job in db.__iter__():
            job["jobman.status"] = 0

def clear_jobman(jobargs):
    """
    Clear jobs in postgresql database table.
    """
    db = api0.open_db("postgres://%(user)s@%(host)s:"
                      "%(port)d/%(database)s?table=%(table)s"
                      % {"user": jobargs.user,
                         "host": jobargs.host,
                         "port": jobargs.port,
                         "database": jobargs.database,
                         "table": jobargs.table,
                      })
    for job in db.__iter__():
        job.delete()

def translate_string(s, to):
    """
    Translate strings from and to format that wont break in knex.
    Knex interprets "." as "_" and removes "_" from keys.
    """
    if to == "knex":
        rep = ["_", "&"]
    elif to == "pylearn2":
        rep = ["&", "_"]
    else:
        raise ValueError()
    return s.replace(*rep)

def translate(d, to):
    """
    Helper method to translate dict keys to and from knex format.
    Knex interprets "." as "_" and removes "_" from keys.
    """

    def recursive_translate(rd, rep):
        if not isinstance(rd, dict):
            return
        for k, v in rd.iteritems():
            new_k = k.replace(*rep)
            if new_k != k:
                assert new_k not in rd
                rd[new_k] = v
                rd.pop(k)
                k = new_k

            recursive_translate(v, rep)

    new_d = copy.deepcopy(d)
    assert to in ["knex", "pylearn2"]
    if to == "knex":
        rep = ["_", "&"]
    elif to == "pylearn2":
        rep = ["&", "_"]
    else:
        raise ValueError()
    recursive_translate(new_d, rep)
    return new_d

def load_experiments_jobman(experiment_module, jobargs):
    """
    Load jobs from experiment onto postgresql database table.
    """
    dbdescr = ("postgres://%(user)s@%(host)s:"
               "%(port)d/%(database)s?table=%(table)s"
               % {"user": jobargs.user,
                  "host": jobargs.host,
                  "port": jobargs.port,
                  "database": jobargs.database,
                  "table": jobargs.table,
                  })
    db = api0.open_db(dbdescr)

    experiment = imp.load_source("module.name", experiment_module)
    for i, items in enumerate(experiment.generator):
        hyperparams = experiment.default_hyperparams
        state = DD()
        set_hyper_parameters(hyperparams, **dict((k, v) for k, v in items))
        state.hyperparams = translate(hyperparams, "knex")
        state["out&path"] = path.abspath(jobargs.out_path)
        state["experiment&module"] = path.abspath(experiment_module)
        state["dbdescr"] = dbdescr

        sql.insert_job(run_experiment_jobman,
                       flatten(state),
                       db)
    db.createView("%s" % jobargs.table)

def run_experiment_jobman(state, channel):
    """
    Main jobman experiment function, called by all jobs.
    """
    experiment_module = state["experiment&module"]
    experiment = imp.load_source("module.name", experiment_module)

    yaml_template = open(experiment.yaml_file).read()
    hyperparams = expand(flatten(translate(state.hyperparams, "pylearn2")),
                         dict_type=ydict)

    if not state["out&path"].endswith("job_%d" % state["jobman.id"]):
        state["out&path"] = path.join(state["out&path"],
                                      "job_%d" % state["jobman.id"])
    channel.save()
    out_path = path.join(state["out&path"])
    try:
        run_experiment(experiment, hyperparams, ask=False, out_path=out_path,
                       dbdescr=state["dbdescr"], job_id=state["jobman.id"])
    except ValueError as e:
        if str(e) == "KILLED":
            return channel.CANCELED
        else:
            return channel.ERR_RUN
    except:
        return channel.ERR_RUN

    print "Ending experiment"
    return channel.COMPLETE
