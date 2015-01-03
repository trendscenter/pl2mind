"""
Module to run analysis on jobman tables.
"""

import argparse
import imp
from jobman.tools import flatten
from jobman import sql
import logging
from os import path
from pylearn2.utils import serial


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def html_header():
    header_string = ("<html>\n"
                     "<head>\n"
                     "<link rel=\"stylesheet\" href=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/blue/style.css\" type=\"text/css\" media=\"print, projection, screen\" />\n"
                     "<script type=\"text/javascript\" "
                     "src=\"http://code.jquery.com/jquery-1.11.2.min.js\"></script>\n"
                     "<script type=\"text/javascript\" "
                     "src=\"http://code.jquery.com/jquery-migrate-1.2.1.min.js\"></script>\n"
                     "<script type=\"text/javascript\" "
                     "src=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/tablesorter/js/jquery.tablesorter.js\"></script>\n"
                     "<script type=\"text/javascript\">\n"
                     "$(document).ready(function() {\n"
                     "$(\"table\").tablesorter({\n"
                     "debug: true\n"
                     "});\n" 
                     "});\n" 
                     "</script>\n" 
                     "\t<script type=\"text/javascript\" "
                     "src=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/tablesorter/jquery-latest.js\"></script>\n"
                     "\t<script type=\"text/javascript\" "
                     "src=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/tablesorter/jquery.tablesorter.js\">"
                     "</script>\n"
                     "<style media=\"screen\" type=\"text/css\">\n"
                     "table.tablesorter thead tr .header {\n"
                     "background-position: left center;\n"
                     "padding-left: 20px;\n"
                     "}\n"
                     "</style>\n"
                     "<title>%(title)s</title>\n"
                     "</head>\n")
    return header_string

def html_column_names(model_dict, results_dict):
    column_name_string = (
        "<table id=\"ResultsTable\" class=\"tablesorter\" border=\"0\" cellpadding=\"0\" cellspacing=\"1\">\n"
        "<thead>\n<tr>")
    for k in model_dict.keys() + results_dict.keys():
        column_name_string += "\n\t<th>%s\t\n</th>" % k
    column_name_string +=  "\n</tr>\n</thead>\n<tbody>\n"
    return column_name_string

def html_rows(model_dict, results_dict):
    row_string = "<tr>"
    for k in model_dict.keys():
        row_string += "\n\t<td>%s</td>" % model_dict[k]
    row_string += "\n"
    for k in results_dict.keys():
        row_string += "<td>%s</td>" % results_dict[k]
    row_string += "\n</tr>\n"
    return row_string

def html_footer():
    footer_string = ("</tbody>\n"
                     "</table>\n"
                     "</html>")
    return footer_string

def make_html_table(model_dicts, results_dicts, title):
    table_string = html_header()
    assert len(model_dicts) == len(results_dicts)
    table_string += html_column_names(model_dicts[0], results_dicts[0])
    for md, rd in zip(model_dicts, results_dicts):
        table_string += html_rows(md, rd)
    table_string += html_footer()
    user = path.expandvars("$USER")
    table_string = table_string % {"user": user, "title": title}
    return table_string

def analyze(args):
    if not args.experiment:
        raise ValueError("Must include experiment source file")
    logger.info("Loading module %s" % args.experiment)
    experiment_module = imp.load_source("module.name", args.experiment)

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

    model_dicts = []
    results_dicts = []

    for job in db.__iter__():
        if job.status in [1, 2]:
            file_prefix = job["file_parameters.save_path"]

            model_dict = dict((k.split(".")[-1], job["hyper_parameters." + k])
                              for k in model_keys if "__builder__" not in k)
            model_dict["status"] = job.status
            model_dict["id"] = job.id
            model_dict["file_prefix"] = file_prefix
            logger.info("Analyzing job %(id)d, with status %(status)d, "
                        "and file_prefix %(file_prefix)s"
                        % model_dict)
            model_dicts.append(model_dict)

            if job.status == 1:
                logger.info("Model not complete. Loading from checkpoint.")
                model = serial.load(file_prefix + "_best.pkl")
                try:
                    results_dicts.append(experiment_module.extract_results(model))
                except AttributeError:
                    raise ValueError("%s does not implement %s" % 
                                     (experiment_module,
                                      "extract_results(<model>, <file_prefix>)"))

            else:
                results_keys = [k.split(".")[-1] for k in job.keys() if "results." in k]
                results_dicts.append(dict((k, job["results." + k]) for k in results_keys))

    if args.html:
        html_table_string = make_html_table(model_dicts, results_dicts, args.table + " results")
        with open(args.html, "w") as f:
            f.write(html_table_string)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("user")
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("database")
    parser.add_argument("table")
    parser.add_argument("experiment")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--html", default=None)

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    analyze(args)
