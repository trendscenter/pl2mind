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

def html_header(title):
    header_string = (
        "<html>\n"
        "<head>\n"             
        "<link rel=\"stylesheet\" href=\"html_css/blue/theme.blue.css\" type=\"text/css\" "
        "media=\"print, projection, screen\" />\n"
        "<link rel=\"stylesheet\" href=\"html_css/popup_css/style.css\" type=\"text/css\">\n"
        "<link rel=\"stylesheet\" href=\"html_css/column_selector.css\" type=\"text/css\">\n"
        "<link rel=\"stylesheet\" href=\"html_css/bootstrap.min.css\">\n"
        "<script "
        "src=\"http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js\">"
        "</script>\n"
        "<script type=\"text/javascript\" "
        "src=\"html_css/tablesorter/js/jquery.tablesorter.js\">"
        "</script>\n"
        "<script src=\"html_css/js/bootstrap.min.js\"></script>\n"
        "<script src=\"html_css/tablesorter/js/widgets/widget-columnSelector.js\">"
        "</script>\n" 
        "<script src=\"html_css/js/popup.js\"></script>\n"
        "<script src=\"html_css/js/table_sorter.js\"></script>\n"
        "<title>%(title)s</title>\n"
        "</head>\n"
        "<div><h1>%(title_capped)s</h1></div>"% {"title": title, "title_capped": title.upper()})
    return header_string

def html_elements():
    elements_string = (
        "<div class=\"cd-popup\" role=\"alert\">\n"
	"<div class=\"cd-popup-container\">\n"
        "<p id=\"bashscript\"></p>\n"
        "<a href=\"#0\" class=\"cd-popup-close img-replace\">Close</a>\n"
	"</div> <!-- cd-popup-container -->\n"
        "</div> <!-- cd-popup -->\n"
        )
    return elements_string

def html_column_names(model_keys, results_keys, 
                      non_priority_keys, priority_keys):
    column_name_string = "\n\t<th>Process\t\nscript</th>"
    for k in model_keys:
        if k in non_priority_keys:
            column_name_string += "\n\t<th class=\"columnSelector-false\">%s\t\n</th>" % k
        else:
            column_name_string += "\n\t<th>%s\t\n</th>" % k
    for k in results_keys:
        if k in priority_keys:
            column_name_string += "\n\t<th>%s\t\n</th>" % k
        else:
            column_name_string += "\n\t<th class=\"columnSelector-false\">%s\t\n</th>" % k
    return column_name_string

def html_rows(model_keys, results_keys, model_dict, results_dict):
    row_string = "<tr>"
    process_string = "\"python $PYLEARN2_DIR/neuroimaging_utils/tools/mri_analysis.py %s.pkl\"" % model_dict["file_prefix"]
    row_string += "\n\t<td><a class=\"cd-popup-trigger\" href=\"#\" id=%s>Process</a></td>" % process_string
    for k in model_keys:
        if k == "file_prefix" and model_dict.get(k, False):
            row_string += "\n\t<td class=\"links\" id=\"%(path)s\"></td>" % {"path": model_dict[k].split("/")[-1]}
        else:
            row_string += "\n\t<td>%s</td>" % model_dict.get(k, "-")
    row_string += "\n"
    for k in results_keys:
        row_string += "<td>%s</td>" % results_dict.get(k, "-")
    row_string += "\n</tr>\n"
    return row_string

def html_footer():
    footer_string = ("<script>\n"
#                     "$(document).ready(function(){\n"
                     "var els = document.getElementsByClassName(\"links\");\n"
                     "for (var i=0; i < els.length; i++ ){\n"
                     "var id = els[i].id;\n"
                     "els[i].innerHTML = checkDir(id);\n"
                     "}\n"
#                     "})\n"
                     "</script>\n"
                     "</html>")
    return footer_string

class HTMLPage(object):
    title_dict = {0: "Waiting",
                  1: "Running",
                  2: "Finished",
                  3: "Failed",
                  4: "Failed",
                  5: "Failed"}

    def __init__(self, title):
        self.header_string = html_header(title)
        self.footer_string = html_footer()
        self.table_strings = []

    def add_table(self, key, model_dicts, results_dicts, results_of_interest):
        assert len(model_dicts) == len(results_dicts)
        model_keys = sorted(list(
                set(k for keys in [md.keys() 
                                   for md in model_dicts] for k in keys)))
        result_keys = sorted(list(
                set(k for keys in [rd.keys() 
                                   for rd in results_dicts] for k in keys)))
        repeated_value_keys = [k for k in model_keys
                               if all(md[k] == model_dicts[0][k] for md in model_dicts)]
        table_title = HTMLPage.title_dict[key]
        table_string = (
            "<div>\n"
            "<div style=\"width: 100%%;\">\n"
            "<div style=\"margin-left: 10px; width: 60px; float: left;\"><h2>\t%(table_title)s</h2></div>\n"
            "<div style=\"margin-left: 75px; float: bottom; \" class=\"columnSelectorWrapper\">\n"
            "<button id=\"popover%(key)d\" type=\"button\" class=\"btn btn-default\">\n"
            "Select\n"
            "</button>\n"
            "</div>\n"
            "</div>\n"
            "<table class=\"tablesorter bootstrap-popup%(key)d\">\n"
            "<thead>\n"
            "<tr>\n" % {"table_title": table_title, "key": key})

        table_string += html_column_names(model_keys, result_keys,
                                          non_priority_keys=repeated_value_keys,
                                          priority_keys=results_of_interest)
        table_string +=  (
            "</tr>\n"
            "</thead>\n"
            "<tbody>\n"
            )
        for md, rd in zip(model_dicts, results_dicts):
            table_string += html_rows(model_keys, result_keys, md, rd)
        table_string += ("</tbody>\n"
                         "</table>\n"
                         "</div>\n"
                         )
        self.table_strings.append(table_string)
    
    def write(self, filename):
        with open(args.html, "w") as f:
            f.write(self.header_string)
            f.write(html_elements())
            for table_string in self.table_strings:
                f.write(table_string)
            f.write(self.footer_string)

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

    model_dicts = {0: [], 1: [], 2: [], 3: []}

    for job in db.__iter__():
        file_prefix = job["file_parameters.save_path"]
        model_dict = dict(("\n".join(k.split(".")),
                           job.get("hyper_parameters." + k, None))
                          for k in model_keys if "__builder__" not in k)
        model_dict["status"] = job.status
        model_dict["id"] = job.id
        model_dict["file_prefix"] = file_prefix
        if job.status in [1, 2]:
            if job.status == 1 and args.finished_only:
                continue
            logger.info("Analyzing job %(id)d, with status %(status)d, "
                        "and file_prefix %(file_prefix)s"
                        % model_dict)

            if job.status == 1 or args.reload:
                logger.info("Model not complete. Loading from checkpoint.")
                try:
                    model = serial.load(file_prefix + "_best.pkl")
                except IOError:
                    logger.info("File not found")
                    continue
                try:
                    results_dict = experiment_module.extract_results(model)
                except AttributeError:
                    raise ValueError("%s does not implement %s" % 
                                     (experiment_module,
                                      "extract_results(<model>, <file_prefix>)"))

            else:
                results_keys = [k.split(".")[-1] for k in job.keys() if "results." in k]
                results_dict = dict((k, job["results." + k]) for k in results_keys)
            model_dicts[job.status].append((model_dict, results_dict))
        elif job.status in [3, 4, 5]:
            model_dicts[3].append((model_dict, {}))
        elif job.status == 0:
            model_dicts[0].append((model_dict, {}))

    try:
        results_of_interest = experiment_module.results_of_interest
    except AttributeError:
        results_of_interest = []

    if args.html:
        html = HTMLPage(args.table + " results")
        for k in [2, 1, 0, 3]:
#            if k != 2: continue
            if len(model_dicts[k]) > 0:
                html.add_table(k, 
                               [md for md, _ in model_dicts[k]],
                               [rd for _, rd in model_dicts[k]],
                               results_of_interest)
        html.write(args.html)

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
    parser.add_argument("-r", "--reload", action="store_true")
    parser.add_argument("-f", "--finished_only", action="store_true")

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    print args
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    analyze(args)
