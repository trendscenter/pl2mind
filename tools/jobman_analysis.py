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
import os
from os import path
import cPickle as pickle
import psutil
from pylearn2.neuroimaging_utils.tools import mri_analysis
from pylearn2.neuroimaging_utils.tools.write_table import HTMLPage
from pylearn2.utils import serial
from sys import stdout
import time


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

class JobDict(dict):
    def __init__(self, args):
        self.experiment_module = imp.load_source("module.name", args.experiment)
        self.table_dir = serial.preprocess(path.join(args.out_dir, args.table))
        self.manager = mp.Manager()

        self.in_queue = manager.list()
        running_process_in_queue = manager.list()
        
        self.message_queue = mp.Queue()
        self.model_queue = mp.Queue()
        self.running_results_queue = mp.Queue()        

        message_process = mp.Process(target=message_worker,
                                     args=(args.table, message_queue, ))
        message_process.start()
        
        
        model_process = mp.Process(target=model_worker,
                                   args=(args, model_queue,
                                         message_queue, in_queue))
        model_process.start()

        
        
        running_process = mp.Process(target=running_model_results,
                                     args=(job_dict, table_dir,
                                           running_results_queue, running_process_in_queue,
                                           message_queue, experiment_module))
        running_process.start()

        update_event = mp.Event()
        table_process = mp.Process(target=table_worker,
                                   args=(experiment_module, args, job_dict,
                                         update_event, message_queue,
                                         model_queue, in_queue,
                                         running_results_queue, running_process_in_queue))
        table_process.start()

        backup_jobdict_file = path.join(table_dir, "jobdict.pkl")
        if path.isfile(backup_jobdict_file) and not args.reload:
            message_queue.put("Loading backup jobdict")
            old_job_dict = pickle.load(open(backup_jobdict_file, "rb"))
            job_dict.update(old_job_dict)
            message_queue.put("Loading complete")
        else:
            message_queue.put("Backup jobdict not found, creating from scratch")

        update_event.set()
        
        update_process = mp.Process(target=table_trigger,
                                    args=(update_event, ))
        update_process.start()

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        subparsers.required = True

        process_parser = subparsers.add_parser("process")
        process_parser.set_defaults(which="process")
        process_parser.add_argument("job_id", type=int)

        show_parser = subparsers.add_parser("show")
        show_parser.set_defaults(which="show")

        quit_parser = subparsers.add_parser("quit")
        quit_parser.set_defaults(which="quit")

        kill_parser = subparsers.add_parser("kill")
        kill_parser.set_defaults(which="kill")
        kill_parser.add_argument("job_id", type=int)



def update_jobdict(job_dict, db, experiment_module, table_dir,
                   running_results_queue, running_process_in_queue):
    """
    Updates the job dict with the database.
    """

    if not hasattr(experiment_module, "default_hyperparams"):
        raise AttributeError("%s does not implement %s"
                         % (experiment_module, "default_hyperparams()"))

    hyper_params = experiment_module.default_hyperparams()
    model_keys = flatten(hyper_params).keys()

    old_jobdict = copy.deepcopy(job_dict)

    for job in db.__iter__():
        if not job.id in job_dict.keys():
            params = get_param_dict(job, model_keys)
            process = get_process_dict(job)
            d = {}
            d["params"] = params
            d.update(process)
            d["results"] = {}
            job_dict[job.id] = d
        else:
            process = get_process_dict(job)
            j = job_dict[job.id]
            j.update(process)
            job_dict[job.id] = j

        if job.status == 1:
            running_results_queue.put(job.id)
            running_process_in_queue.append(job.id)
        elif job.status == 2:
            plot_results = [k + ".pdf" for k in experiment_module.plot_results.keys()]
            all_there = True
            file_prefix = job_dict[job.id]["file_prefix"]
            for plot_result in plot_results:
                if not path.isfile(path.join(table_dir, file_prefix, plot_result)):
                    all_there = False
                    break
            if all_there:
                j = job_dict[job.id]
                results_keys = [k.split(".")[-1] for k in job.keys() if "results." in k]
                j["results"].update(dict((k, job["results." + k]) for k in results_keys))
                job_dict[job.id] = j
            else:
                running_results_queue.put(job.id)
                running_process_in_queue.append(job.id)

    return old_jobdict

def get_param_dict(job, model_keys):
    """
    Get model parameters + hyperparams as dictionary.
    """

    params = dict((" ".join(k.replace(".__builder__", "").split(".")),
                   job.get("hyper_parameters." + k, None))
                  for k in model_keys)

    return params

def load_results_from_model(checkpoint, experiment_module, out_dir=None):
    results_dict = {}
    try:
        model = serial.load(checkpoint)
    except IOError:
        raise IOError("Checkpoint %s not found" % checkpoint)

    if not hasattr(experiment_module, "extract_results"):
        raise AttributeError("%s does not implement %s" % 
                             (experiment_module,
                              "extract_results(<model>, <file_prefix>)"))

    results = experiment_module.extract_results(model, out_dir)
    return results

def get_process_dict(job):
    """
    Loads running and other miscellaneous data into a dict.
    """

    md_dict = {"status": job.status,
               "id": job.id,
               "file_prefix": job["file_parameters.save_path"]}

    if job.status == 0:
        return md_dict
    try:
        md_dict["host"] = job["jobman.sql.host_name"]
#        md_dict["run_time"] = job["jobman.run_time"]
        md_dict["start_time"] = job["jobman.start_time"]
#        md_dict["stop_time"] = job["jobman.stop_time"]
        md_dict["priority"] = job["jobman.sql.priority"]
        md_dict["pid"] = job["pid"]
    except KeyError as e:
        pass

    try:
        p = psutil.Process(md_dict["pid"])
        md_dict["cpu"] = p.get_cpu_percent()
        md_dict["mem"] = p.get_memory_percent()
        md_dict["user"] = p.username
    except:
        pass

    return md_dict

def check_results_dir(table_dir, file_prefix):
    return path.isdir(path.join(table_dir, file_prefix.split("/")[-1]))

def table_worker(experiment_module, args,
                 job_dict, update_event,
                 message_queue, model_queue,
                 in_queue, running_results_queue, running_process_in_queue):

    assert isinstance(args, argparse.Namespace), type(args)
    results_of_interest = experiment_module.results_of_interest
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))
    table_name = args.table
    html = HTMLPage(table_name + " results")
    
    while True:
        db = sql.db("postgres://%(user)s@%(host)s:%(port)d/%(database)s?table=%(table)s"
                    % {"user": args.user,
                       "host": args.host,
                       "port": args.port,
                       "database": args.database,
                       "table": args.table,
                       })
    
        update_event.wait()

        old_jobdict = update_jobdict(job_dict, db, experiment_module, table_dir,
                                     running_results_queue, running_process_in_queue)

        unprocessed_finished = [j_id
                                for j_id in job_dict.keys()
                                if not check_results_dir(
                table_dir, job_dict[j_id]["file_prefix"])
                                and job_dict[j_id]["status"] == 2]
        
        new_finished_jobs = [j_id
                             for j_id in job_dict.keys() 
                             if (job_dict[j_id]["status"] == 2 and
                                 (j_id in old_jobdict and old_jobdict[j_id]["status"] == 1))]

        for job_id in unprocessed_finished + new_finished_jobs:
            prefix = job_dict[job_id]["file_prefix"]
            if not prefix in in_queue:
                message_queue.put("Adding job %d to processing queue (%s)" % (job_id, prefix))
                in_queue.append(prefix)
                model_queue.put(prefix)

        backup_jobdict_file = path.join(table_dir, "jobdict.pkl")
        pickle.dump(dict(job_dict), open(backup_jobdict_file, "wb"))

        html.clear()

        status_dict = {
            "waiting": [0],
            "running": [1],
            "done": [2],
            "failed": [3, 4, 5]
            }

        for status in ["waiting", "running", "done", "failed"]:
            row_keys = [k for k in job_dict.keys() if job_dict[k]["status"] in status_dict[status]]
            row_keys.sort()

            for group in ["process", "params", "results"]:
                table_dict = {}
                column_keys = set()
                for job_id in row_keys:
                    if group == "process":
                        # Bit of a hack to get here. REDO
                        table_dict[job_id] = dict((k, job_dict[job_id][k]) for k in job_dict[job_id].keys()
                                                      if k not in ["params", "results"])
                        table_dict[job_id]["file_prefix"] = table_dict[job_id]["file_prefix"].split("/")[-1]
                    else:
                        table_dict[job_id] = job_dict[job_id][group]
                    column_keys = column_keys.union(set(table_dict[job_id].keys()))

                column_keys = list(column_keys)
                column_keys.sort()
            
                if group == "results":
                    roi = results_of_interest
                else:
                    roi = column_keys

                html.write_table(table_dict, row_keys, column_keys,
                                 path.join(table_dir, "%(status)s_%(group)s.txt"
                                           % {"status": status, "group": group}),
                                 roi)
                html.write(path.join(table_dir, "table.html"))
                
        update_event.clear()

def model_worker(args, model_queue, message_queue, in_queue):
    mri_analysis.logger.level = 40
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))
    message_queue.put("Model worker ready.")
    q = model_queue.get()
    while q:
        file_prefix = q
        #probably a better way to do this
        message_queue.put("Processing model %s" % file_prefix)
        try:
            mri_analysis.main(file_prefix + "_best.pkl",
                              table_dir,
                              "",
                              prefix=file_prefix.split("/")[-1])
        except IOError:
            mri_analysis.main(file_prefix + "_best.pkl",
                              table_dir,
                              "",
                              prefix=file_prefix.split("/")[-1])
        except IOError:
            message_queue.put("Checkpoint for %s not found" % file_prefix)
        in_queue.remove(file_prefix)
        message_queue.put("Finished processing model %s" % file_prefix)
        time.sleep(5)
        q = model_queue.get()

def message_worker(prompt, message_queue):
    while True:
        message = message_queue.get()
        if message == "quit":
            stdout.write("Quitting....\n")
            return

        stdout.write("\r<alert>: %s\n" % message)
        stdout.write("%s: " % prompt)
        stdout.flush()

def table_trigger(update_event, secs=30):
    while True:
        update_event.set()
        time.sleep(secs)

def running_model_results(job_dict, table_dir, running_results_queue, running_process_in_queue, message_queue, experiment_module):
    q = running_results_queue.get()
    while True:
        job_id = q
        file_prefix = job_dict[job_id]["file_prefix"]
        message_queue.put("Getting results for running model: %s" % file_prefix)
        job = job_dict[job_id]
        job_results = job["results"]
        out_dir = path.join(table_dir, file_prefix.split("/")[-1])
        try:
            checkpoint = job_dict[job_id]["file_prefix"] + ".pkl"
            results = load_results_from_model(checkpoint, experiment_module, out_dir)
            job_results.update(results)
            job_dict[job_id] = job
            message_queue.put("Finished results for running model: %s" % file_prefix)
        except IOError:
            checkpoint = job_dict[job_id]["file_prefix"] + "_best.pkl"
            results = load_results_from_model(checkpoint, experiment_module, out_dir)
            job["results"] = results
            message_queue.put("Finished results for running model: %s best()" % file_prefix)
        except IOError:
            message_queue.put("Model %s not found" % file_prefix)
        running_process_in_queue.remove(job_id)
        time.sleep(5)
        q = running_results_queue.get()
        
def main(args):
    if not args.experiment:
        raise ValueError("Must include experiment source file")
    logger.info("Loading module %s" % args.experiment)
    experiment_module = imp.load_source("module.name", args.experiment)
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))

    manager = mp.Manager()
    job_dict = manager.dict()
    in_queue = manager.list()

    message_queue = mp.Queue()
    message_process = mp.Process(target=message_worker,
                                 args=(args.table, message_queue, ))
    message_process.start()

    model_queue = mp.Queue()
    model_process = mp.Process(target=model_worker,
                               args=(args, model_queue,
                                     message_queue, in_queue))
    model_process.start()

    running_results_queue = mp.Queue()
    running_process_in_queue = manager.list()
    running_process = mp.Process(target=running_model_results,
                                 args=(job_dict, table_dir, running_results_queue, running_process_in_queue,
                                       message_queue, experiment_module))
    running_process.start()

    update_event = mp.Event()
    table_process = mp.Process(target=table_worker,
                               args=(experiment_module, args, job_dict, update_event, message_queue,
                                     model_queue, in_queue, running_results_queue, running_process_in_queue))
    table_process.start()

    backup_jobdict_file = path.join(table_dir, "jobdict.pkl")
    if path.isfile(backup_jobdict_file) and not args.reload:
        message_queue.put("Loading backup jobdict")
        old_job_dict = pickle.load(open(backup_jobdict_file, "rb"))
        job_dict.update(old_job_dict)
        message_queue.put("Loading complete")
    else:
        message_queue.put("Backup jobdict not found, creating from scratch")

    update_event.set()

    update_process = mp.Process(target=table_trigger,
                                args=(update_event, ))
    update_process.start()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    process_parser = subparsers.add_parser("process")
    process_parser.set_defaults(which="process")
    process_parser.add_argument("job_id", type=int)

    show_parser = subparsers.add_parser("show")
    show_parser.set_defaults(which="show")

    quit_parser = subparsers.add_parser("quit")
    quit_parser.set_defaults(which="quit")

    kill_parser = subparsers.add_parser("kill")
    kill_parser.set_defaults(which="kill")
    kill_parser.add_argument("job_id", type=int)

    while True:
        command = raw_input("%s: " % args.table)
        try:
            a = parser.parse_args(command.split())
            if a.which == "quit":
                message_queue.put("quit")
                table_process.terminate()
                update_process.terminate()
                model_process.terminate()
                running_process.terminate()
                message_process.join()
                message_process.terminate()
                return
            elif a.which == "show":
                message_queue.put(job_dict)
            elif a.which == "process":
                job_id = a.job_id
                prefix = job_dict[job_id]["file_prefix"]
                if not prefix in in_queue:
                    in_queue.append(prefix)
                    model_queue.put(prefix)
                    message_queue.put("Models in process queue:\n %s" % "\n".join(in_queue))
                else:
                    message_queue.put("Model %s already in queue." % prefix)
            elif a.which == "kill":
                job_id = a.job_id
                now = time.time()
                process = psutil.Process(job_dict[job_id]["pid"])
                if process.is_running():
                    process.kill()
                    while process.is_running():
                        if time.time() - now > 20:
                            break
                    if process.is_running():
                        message_queue.put("Job not stopped after 20 secs")
                    else:
                        message_queue.put("Job %d killed" % job_id)
                else:
                    message_queue.put("No such job running.")

        except Exception as e:
            print e
            pass
            
    table_process.join()
    model_process.join()

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
