"""
Module to run analysis on jobman tables.
"""

import argparse
import copy
import datetime
import imp
from jobman.tools import flatten
from jobman import sql
import logging
import multiprocessing as mp
from multiprocessing.managers import BaseManager, SyncManager, NamespaceProxy
import os
from os import path
import cPickle as pickle
import psutil
from pl2mind.tools import mri_analysis
from pl2mind.tools.write_table import HTMLPage
from pylearn2.utils import serial
from sys import stdout
import time


experiment = None

logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)


class CommandLine(mp.Process):
    """
    Class process for command line.
    """
    def __init__(self, alerter, table):
        self.__dict__.update(locals())
        self.make_argument_parser()

    def make_argument_parser(self):
        self.parser = argparse.ArgumentParser()
        subparsers = self.parser.add_subparsers()
        subparsers.required = True

        analyze_parser = subparsers.add_parser("analyze")
        analyze_parser.set_defaults(which="analyze")
        analyze_parser.add_argument("job_id", type=int)

        show_parser = subparsers.add_parser("show")
        show_parser.set_defaults(which="show")
        show_parser.add_argument("what", help="What to show")

        quit_parser = subparsers.add_parser("quit")
        quit_parser.set_defaults(which="quit")

        kill_parser = subparsers.add_parser("kill")
        kill_parser.set_defaults(which="kill")
        kill_parser.add_argument("job_id", type=int)

    def run(self):
        while True:
            command = raw_input("%s: " % self.alerter.prompt)
            try:
                a = self.parser.parse_args(command.split())
                if a.which == "quit":
                    self.alerter(None)
                    self.table.stop()
                    return
                elif a.which == "show":
                    what = a.what
                    try:
                        job_id = int(what)
                        if job_id in self.table.jobs.keys():
                            print self.table[job_id]
                        else:
                            raise AttributeError("Job %d not found in table"
                                             % job_id)
                    except ValueError:
                        if what == "table":
                            print self.table
                        elif what == "updater":
                            print self.table.updater
                        elif what == "analyzer":
                            print self.table.analyzer
                        elif what == "running":
                            print [k for k in self.table.jobs.keys()
                                   if (self.table.jobs[k].stats.get(
                                    "status", None) == 1)]
                        else:
                            raise ValueError("Cannot show object %s" % what)

                elif a.which == "analyze":
                    job_id = a.job_id
                    if not job_id in self.table:
                        self.alert_queue.put("Job id %d not found." % job_id)
                    else:
                        p = self.table.analyzer.put(prefix)
                        if p:
                            self.alerter(self.table.analyzer)
                        else:
                            self.alerter("Model %s already in queue." % prefix)
                elif a.which == "kill":
                    job_id = a.job_id
                    if not job_id in self.table:
                        self.alerter("Job id %d not found." % job_id)
                    else:
                        self.table[job_id].kill()

            except Exception as e:
                print e
                pass

        self.alerter.join()


class Alerter(mp.Process):
    """
    Class for sending alerts to stdout.
    """
    def __init__(self, prompt="$"):
        self.__dict__.update(locals())
        self.alert_queue = mp.Queue()
        self.verbose = False
        super(Alerter, self).__init__()

    def run(self):
        while True:
            message = self.alert_queue.get()
            if message is None:
                stdout.write("Quitting....\n")
                stdout.flush()
                return

            stdout.write("\r<alert>: %s\n" % message)
            stdout.write("%s: " % self.prompt)
            stdout.flush()

    def __call__(self, message):
        self.alert_queue.put(message)


class Updater(mp.Process):
    """
    Class for updating job information.
    """
    def __init__(self, manager, alerter, jobs, db, analyze=False):
        self.__dict__.update(locals())
        self.queue = mp.Queue()
        self.in_queue = manager.list()
        super(Updater, self).__init__()

    def run(self):
        self.alerter("Starting updater")
        while True:
            job_id = self.queue.get()
            job = self.jobs[job_id]
            if job is None:
                self.alerter("Stopping updater")
                return
            assert isinstance(job, Job)
            assert job.job_id in self.in_queue
            job = update(job, self.db, self.analyze)
            self.jobs[job_id] = job
            self.in_queue.remove(job.job_id)

    def put(self, job_id):
        if not job_id in self.in_queue:
            self.in_queue.append(job_id)
            self.queue.put(job_id)
            return 1
        return None

    def __str__(self):
        return "Update queue: %r" % list(self.in_queue)


class Analyzer(mp.Process):
    """
    Class for analyzing job results.
    """
    def __init__(self, manager, alerter, jobs, db):
        self.__dict__.update(locals())
        self.queue = mp.Queue()
        self.in_queue = manager.list()
        super(Analyzer, self).__init__()

    def run(self):
        self.alerter("Starting analyzer")
        while True:
            job_id = self.queue.get()
            job = self.jobs[job_id]
            if job is None:
                self.alerter("Quitting analyzer")
                return
            assert isinstance(job, Job)
            assert job.job_id in self.in_queue
            model = job.get_checkpoint()
            analyze_model(job, model)
            self.in_queue.remove(job.job_id)

    def put(self, job_id):
        if not job_id in self.in_queue:
            self.in_queue.append(job_id)
            self.queue.put(job_id)
            return 1
        return None

    def __str__(self):
        return "Update queue: %r" % list(self.in_queue)


class JobManager(SyncManager): pass


class JobProxy(NamespaceProxy):
    _exposed_ = ("job_id", "stats", "hyperparams",
                 "results", "__getattribute__", "__setattr__")


def set_hyperparams(job, db):
    dbjob = db.get(job.job_id)
    job.file_prefix = dbjob["file_parameters.save_path"]
    job.out_dir = path.join(job.table_dir,
                             job.file_prefix.split("/")[-1])
    hyperparams = experiment.default_hyperparams()
    model_keys = flatten(hyperparams).keys()
    job.hyperparams = dict(
        (" ".join(k.replace(".__builder__", "").split(".")),
         dbjob.get("hyper_parameters." + k, None))
        for k in model_keys)

def update(job, db, analyze=False):
    dbjob = db.get(job.job_id)
    job = update_stats(job, dbjob)
    if job.stats["status"] in [0, 3, 4, 5]:
        return job
    elif job.pid is not None and job.stats["status"] == 1 and job.is_running():
        model = job.get_checkpoint()
        job.update_results_from_checkpoint(model)
        job.make_plots(model)
    elif job.pid is not None and job.stats["status"] != 1 and job.is_running():
        job.kill()
        job = update_results_from_db(job, dbjob)
    else:
        job = update_results_from_db(job, dbjob)

    if job.stats["status"] == 2 and analyze:
        if path.isfile(path.join(job.out_dir, "info.txt")):
            with open(path.join(job.out_dir, "info.txt"), "r") as f:
                lines = f.readlines()
                if len(lines) == 2:
                    status = int(lines[1])
                else:
                    status = 0
                if status != 2:
                    model = job.get_checkpoint()
                    job.analyze_model(model)
                    job.make_plots(model)
        else:
            model = job.get_checkpoint()
            job.analyze_model(model)
            job.make_plots(model)

    return job

def update_stats(job, dbjob):
    job.pid = dbjob.get("pid", None)
    job.stats.update(id=dbjob.id,
                     status=dbjob.status,
                     priority=dbjob.priority,
                     pid=job.pid,
                     host = dbjob.get("jobman.sql.host_name", None))
    if job.pid is not None and job.is_running():
        p = psutil.Process(job.pid)
        job.stats.update(cpu = p.get_cpu_percent(),
                         mem = p.get_memory_percent(),
                         user = p.username)

    return job


def update_results_from_db(job, dbjob):
    result_keys = [k.split(".")[-1] for k in dbjob.keys()
                   if "results." in k]
    job.results.update(dict((k, dbjob["results." + k])
        for k in result_keys))

    return job


class Job(object):
    """
    Single job class.
    """

    def __init__(self,
                 job_id,
                 table_dir,
                 e=None):

        if e is not None:
            logger.warning("Setting experiment in constructor should only be "
                           "done in debugging.")
            global experiment
            experiment = e

        self.__dict__.update(locals())
        if experiment is None:
            raise ValueError("Experiment has not been loaded yet.\n"
                             "Please load using `imp` before loading jobs.")

        self.stats = {}
        self.results = {}

    def update_results_from_checkpoint(self, model):
        self.results = experiment.extract_results(model)

    def get_checkpoint(self):
        try:
            checkpoint = self.file_prefix + ".pkl"
            model = serial.load(checkpoint)
        except IOError:
            checkpoint = self.file_prefix + "_best.pkl"
            model = serial.load(checkpoint)
        except IOError:
            return None

        return model

    def make_plots(self, model):
        experiment.make_plots(model, self.out_dir)

    def analyze_model(self, model):
        experiment.analysis_fn(model, self.out_dir)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path.join(self.out_dir, "info.txt"), "w") as f:
            f.write(now + "\n")
            f.write(self.stats["status"])

    def is_running(self):
        try:
            p = psutil.Process(self.pid)
        except psutil.NoSuchProcess:
            return False
        return p.is_running()

    def kill(self):
        p = psutil.Process(self.pid)
        if p.is_running():
            p.kill()

    def clean_files(self):
        os.remove(self.file_prefix + "_best.pkl")
        os.remove(self.file_prefix + ".pkl")

    def __repr__(self):
        return ("%r(at %r)(job_id=%d, hyperparams=%r, stats=%r, results=%r)"
                % (self.__class__,
                   hex(id(self)),
                   self.job_id,
                   self.hyperparams,
                   self.stats,
                   self.results))

    def __str__(self):
        return self.__repr__()


class Table(object):
    def __init__(self, jobs, db, name, updater, analyzer, alerter, reload=False):
        self.__dict__.update(locals())

        self.table_dir = serial.preprocess(path.join(args.out_dir,
                                                     self.name))
        self.html = HTMLPage(self.name + " results")

        self.analyzer.start()
        self.updater.start()

    def stop(self):
        self.analyzer.terminate()
        self.updater.terminate()

    def save(self):
        bk_table = path.join(self.table_dir, "table.bk.pkl")
        pickle.dump(dict(self.__dict__), open(bk_table, "wb"))

    def reload(self):
        backup_table_file = path.join(self.table_dir, "table.bk.pkl")
        if path.isfile(backup_table_file):
            with open(backup_table_file, "rb") as f:
                bk_table = pickle.load(f)
                self.__dict__.update(bk_table.__dict__)

    def get(self, key, default=None):
        return self.jobs.get(key, default)

    def write_html(self):
        self.html.clear()
        results_of_interest = experiment.results_of_interest

        status_dict = {
            "waiting": [0],
            "running": [1],
            "done": [2],
            "failed": [3, 4, 5]
            }

        for status in ["waiting", "running", "done", "failed"]:
            row_keys = [k for k in self.jobs.keys()
                        if self.jobs[k].stats.get("status", 0)
                        in status_dict[status]]
            row_keys.sort()

            for group in ["stats", "hyperparams", "results"]:
                table_dict = {}
                column_keys = set()
                for job_id in row_keys:
                    job = self.jobs[job_id]
                    table_dict[job_id] = getattr(job, group)
                    column_keys = column_keys.union(
                        set(table_dict[job_id].keys()))
                    if group == "stats":
                        table_dict[job_id]["file_prefix"] =\
                        job.file_prefix.split("/")[-1]
                        column_keys.add("file_prefix")

                column_keys = list(column_keys)
                column_keys.sort()

                if group == "results":
                    roi = results_of_interest
                else:
                    roi = column_keys

                self.html.write_table(table_dict, row_keys, column_keys,
                                 path.join(table_dir, "%(status)s_%(group)s.txt"
                                           % {"status": status, "group": group}),
                                 roi)
                self.html.write(path.join(self.table_dir, "table.html"))

    def __getitem__(self, key):
        return self.jobs[key]

    def __repr__(self):
        return ("%r(table_name=%r, jobs=%r)" %
                (self.__class__,
                self.name,
                dict(self.jobs)))

    def __str__(self):
        return self.__repr__()

def auto_update(table, secs=30):
    table.alerter("Starting automatic updates")
    table.updater.analyze = False
    while True:
        table.alerter("Updating table")
        for db_job in table.db.__iter__():
            job_id = db_job.id
            job = table.jobs.get(job_id, None)
            if job is not None:
                table.updater.put(job_id)
            else:
                job = Job(job_id, table.table_dir)
                #job = manager.job(job_id, table_dir)
                set_hyperparams(job, table.db)
                table.jobs[job_id] = job
                table.updater.put(job_id)

        table.write_html()
        time.sleep(secs)

JobManager.register("job", Job, JobProxy)

def main(args):
    if not args.experiment:
        raise ValueError("Must include experiment source file")
    global experiment
    logger.info("Loading module %s" % args.experiment)
    experiment = imp.load_source("module.name", args.experiment)

    db = sql.db("postgres://%(user)s@%(host)s:"
                "%(port)d/%(database)s?table=%(table)s"
                % {"user": args.user,
                   "host": args.host,
                   "port": args.port,
                   "database": args.database,
                   "table": args.table,
                   })

    alerter = Alerter(prompt=args.table)
    alerter.start()

    manager = JobManager()
    manager.start()

    jobs = manager.dict()

    updater = Updater(manager, alerter, jobs, db)
    analyzer = Analyzer(manager, alerter, jobs, db)

    table = Table(jobs, db, args.table, updater, analyzer, alerter, args.reload)

    auto_updater = mp.Process(target=auto_update,
                              args=(table,))
    auto_updater.start()

    command_line = CommandLine(alerter, table)
    command_line.run()

    updater.join()
    analyzer.join()

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("user")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("database")
    parser.add_argument("table")
    parser.add_argument("experiment")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--out_dir", default="${PYLEARN2_OUTS}")
    parser.add_argument("-r", "--reload", action="store_true")
    parser.add_argument("-f", "--finished_only", action="store_true")
    parser.add_argument("-l", "--log_dir", default=None)
    parser.add_argument("--process", action="store_true")

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()

    if not path.isdir(serial.preprocess(args.out_dir)):
        raise IOError("Directory %s not found." % serial.preprocess(args.out_dir))

    table_dir = serial.preprocess(path.join(args.out_dir, args.table))

    if not path.isdir(table_dir):
        os.mkdir(table_dir)

    if args.log_dir is not None:
        try:
            os.symlink(path.abspath(path.join(args.log_dir, args.table)),
                       path.join(table_dir, "log_files"))
        except OSError:
            pass

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    main(args)
