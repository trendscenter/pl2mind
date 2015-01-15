"""
Module to run analysis on jobman tables.
"""

import argparse
import imp
from jobman.tools import flatten
from jobman import sql
import logging
import multiprocessing as mp
import os
from os import path
import cPickle as pickle
from pylearn2.neuroimaging_utils.tools import mri_analysis
from pylearn2.neuroimaging_utils.tools.write_table import HTMLPage
from pylearn2.utils import serial
from sys import stdout
import time


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def get_job_dict(experiment_module, args):

    db = sql.db("postgres://%(user)s@%(host)s:%(port)d/%(database)s?table=%(table)s"
                % {"user": args.user,
                   "host": args.host,
                   "port": args.port,
                   "database": args.database,
                   "table": args.table,
                   })

    logging.warning("Models are loaded twice (TODO).")

    try:
        hyper_params = experiment_module.default_hyperparams()
    except AttributeError:
        raise ValueError("%s does not implement %s"
                         % (experiment_module, "default_hyperparams()"))
    model_keys = flatten(hyper_params).keys()
    logger.info("Model keys: %s" % model_keys)

    job_dict = {}
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))

    for job in db.__iter__():
        file_prefix = job["file_parameters.save_path"]
        params = dict(("\n".join(k.replace(".__builder__", "").split(".")),
                       job.get("hyper_parameters." + k, None))
                      for k in model_keys)
        params["status"] = job.status
        params["id"] = job.id
        params["file_prefix"] = file_prefix
        if job.status in [0, 3, 4, 5]:
            results = {}
        elif job.status in [1, 2]:
            logger.info("Analyzing job %(id)d, with status %(status)d, "
                        "and file_prefix %(file_prefix)s"
                        % params)

            if job.status == 1 and args.finished_only:
                results = {}
            elif job.status == 1 or args.reload:
                try:
                    model = serial.load(file_prefix + ".pkl")
                    try:
                        results = experiment_module.extract_results(model)
                    except AttributeError:
                        raise ValueError("%s does not implement %s" % 
                                         (experiment_module,
                                          "extract_results(<model>, <file_prefix>)"))
                except IOError:
                    logger.info("File not found")
                    results = {}
            else:
                results_keys = [k.split(".")[-1] for k in job.keys() if "results." in k]
                results = dict((k, job["results." + k]) for k in results_keys)
                
        job_dict[job.id] = {
            "status": job.status,
            "params": params,
            "results": results
            }

    return job_dict

def check_results_dir(table_dir, file_prefix):
    return path.isdir(path.join(table_dir, file_prefix.split("/")[-1]))

def table_worker(experiment_module, args, job_dict,
                 update_event, message_queue, model_queue):
    assert isinstance(args, argparse.Namespace), type(args)
    
    results_of_interest = experiment_module.results_of_interest
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))
    table_name = args.table
    html = HTMLPage(table_name + " results")
    
    while True:
        update_event.wait()
        message_queue.put("Processing table")
        new_job_dict = get_job_dict(experiment_module, args)
        unprocessed_finished = [j_id
                                for j_id in new_job_dict.keys()
                                if not check_results_dir(
                table_dir, new_job_dict[j_id]["params"]["file_prefix"])
                                and new_job_dict[j_id]["status"] == 2]
        
        new_finished_jobs = [new_job_dict[j]["params"]["file_prefix"]
                             for j in job_dict.keys() 
                             if (new_job_dict[j]["status"] == 2 
                                 and job_dict[j]["status"] == 1)]

        for job_id in unprocessed_finished + new_finished_jobs:
            message_queue.put("Adding job %d to processing queue" %job_id)
            model_queue.put(new_job_dict[job_id]["params"]["file_prefix"])

        job_dict.update(new_job_dict)
        backup_table_file = path.join(table_dir, "table.pkl")
        pickle.dump(dict(job_dict), open(backup_table_file, "wb"))

        message_queue.put("Finishing processing table")

        message_queue.put("Making HTML table")
        for status in [2, 1, 0, 3]:
            jobs = [(v["params"], v["results"]) for v in job_dict.values()
                    if ((v["status"] == status) if status != 3
                        else (v["status"] in [3, 4, 5]))]
            if len(jobs) > 0:
                html.add_table(status, 
                               [p for p, _ in jobs],
                               [r for _, r in jobs],
                               results_of_interest)
        html.write(path.join(table_dir, "table.html"))
        message_queue.put("Finished HTML table")

        update_event.clear()

def model_worker(args, model_queue, message_queue):
    mri_analysis.logger.level = 40
    table_dir = serial.preprocess(path.join(args.out_dir, args.table))
    q = model_queue.get()
    while q:
        file_prefix = q
        message_queue.put("Processing model %s" % file_prefix)
        mri_analysis.main(file_prefix + ".pkl",
                          table_dir,
                          "",
                          prefix=file_prefix.split("/")[-1])
        message_queue.put("Finished processing model %s" % file_prefix)
        time.sleep(5)
        q = model_queue.get()

def message_worker(message_queue):
    while True:
        message = message_queue.get()
        stdout.write("\r<alert>: %s\n" % message)
        stdout.write("$: ")
        stdout.flush()

def table_trigger(update_event, secs):
    assert time >= 60 * 5
    while True:
        time.sleep(secs)
        update_event.set()

def main(args):
    if not args.experiment:
        raise ValueError("Must include experiment source file")
    logger.info("Loading module %s" % args.experiment)
    experiment_module = imp.load_source("module.name", args.experiment)

    manager = mp.Manager()
    job_dict = manager.dict()
    message_queue = mp.Queue()
    message_process = mp.Process(target=message_worker,
                                 args=(message_queue, ))
    message_process.start()

    model_queue = mp.Queue()
    model_process = mp.Process(target=model_worker,
                               args=(args, model_queue,
                                     message_queue))
    model_process.start()

    update_event = mp.Event()
    table_process = mp.Process(target=table_worker,
                               args=(experiment_module,
                                     args, job_dict, 
                                     update_event,
                                     message_queue,
                                     model_queue))
    table_process.start()

    table_dir = serial.preprocess(path.join(args.out_dir, args.table))
    backup_table_file = path.join(table_dir, "table.pkl")
    if path.isfile(backup_table_file):
        message_queue.put("Loading backup table")
        old_job_dict = pickle.load(open(backup_table_file, "rb"))
        job_dict.update(old_job_dict)
        message_queue.put("Loading complete")
    else:
        message_queue.put("Backup table not found, creating from scratch")

    update_event.set()

    update_process = mp.Process(target=table_trigger,
                                args=(update_event,
                                      args.time))
    update_process.start()

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    subparsers.required = True

    process_parser = subparsers.add_parser("process")
    process_parser.set_defaults(which="process")
    process_parser.add_argument("job_id", type=int)

    while True:
        command = raw_input("$: ")
        try:
            a = parser.parse_args(command.split())
            if a.which == "process":
                model_queue.put(job_dict[a.job_id]["params"]["file_prefix"])
        except:
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
    parser.add_argument("-t", "--time", type=int, default=300)

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
