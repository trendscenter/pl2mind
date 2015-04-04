"""
Module for handling jobman experiments
"""

import argparse
import copy
from pl2mind import experiments


default_server = dict(
    user="dlserver",
    host="pleiades",
    port=5431,
    database="jobman"
)

def make_argument_parser():
    """
    Make argument parser for jobmind.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("table")
    parser.add_argument("-d", "--server", default=None,
                        help="comma separated list of args in form: user,host"
                        ",port,database")
    parser.add_argument("-v", "--verbose", action="store_true")

    subparsers = parser.add_subparsers(help="sub-command help")

    load_parser = subparsers.add_parser("load")
    load_parser.set_defaults(which="load")
    load_parser.add_argument("experiment", help="Experiment module.")
    load_parser.add_argument("out_path", help="Out path for the models.")

    status_parser = subparsers.add_parser("status")
    status_parser.set_defaults(which="status")

    run_parser = subparsers.add_parser("run")
    run_parser.set_defaults(which="run")
    run_parser.add_argument("--n_proc", type=int, default=1,
                            help="Number of processors.")

    clear_parser = subparsers.add_parser("clear")
    clear_parser.set_defaults(which="clear", help="Clear the table.")

    set_status = subparsers.add_parser("set_status")
    set_status.set_defaults(which="set_status")
    set_status.add_argument("job_id")

    return parser

def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    jobargs = copy.deepcopy(vars(args))
    jobargs.pop("which")
    jobargs.pop("verbose")

    if jobargs["server"] is None:
        jobargs.update(**default_server)
    else:
        user, host, port, database = jobargs["database"].split(",")
        assert isinstance(user, str)
        assert isinstance(port, int)
        assert isinstance(host, str)
        assert isinstance(database, str)
        jobargs.update(
            user=user,
            host=host,
            port=port,
            database=database
        )

    print jobargs

    if args.which == "load":
        jobargs.pop("experiment")
        print "Loading experiments from %s" % args.experiment
        experiments.load_experiments_jobman(args.experiment, jobargs)

    elif args.which == "status":
        experiments.jobman_status(jobargs)

    elif args.which == "run":
        if args.n_proc > 1:
            experiments.run_jobman_from_sql(jobargs)
        else:
            experiments.run_one_jobman(jobargs)

    elif args.which == "clear":
        experiments.clear_jobman(jobargs)

    elif args.which == "set_status":
        experiments.set_status_jobman(jobargs)

if __name__ == "__main__":
    main()