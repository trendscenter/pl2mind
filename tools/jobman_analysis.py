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
"""<style id="css">/*** custom css only popup ***/
.columnSelectorWrapper {
  position: relative;
  margin: 10px 0;
  display: inline-block;
}
.columnSelector, .hidden {
  display: none;
}
.columnSelectorButton {
  background: #99bfe6;
  border: #888 1px solid;
  color: #111;
  border-radius: 5px;
  padding: 5px;
}
#colSelect1:checked + label {
  background: #5797d7;
  border-color: #555;
}
#colSelect1:checked ~ #columnSelector {
  display: block;
}
.columnSelector {
  width: 120px;
  position: absolute;
  top: 30px;
  padding: 10px;
  background: #fff;
  border: #99bfe6 1px solid;
  border-radius: 5px;
}
.columnSelector label {
  display: block;
}
.columnSelector label:nth-child(1) {
  border-bottom: #99bfe6 solid 1px;
  margin-bottom: 5px;
}
.columnSelector input {
  margin-right: 5px;
}
.columnSelector .disabled {
  color: #ddd;
}

/*** Bootstrap popover ***/
#popover-target label {
  margin: 0 5px;
  display: block;
}
#popover-target input {
  margin-right: 5px;
}
.popover {
	margin-top: -65px; /* adjust popover position */
}
</style>"""
                     "<link rel=\"stylesheet\" href=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/blue/style.css\" type=\"text/css\" media=\"print, projection, screen\" />\n"
                     "<script type=\"text/javascript\" src=\"http://code.jquery.com/jquery-1.11.2.min.js\"></script>\n"
                     "<script type=\"text/javascript\" src=\"http://code.jquery.com/jquery-migrate-1.2.1.min.js\"></script>\n"
                     "<script type=\"text/javascript\" "
                     "src=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/tablesorter/js/jquery.tablesorter.js\"></script>\n"
                     "<script type=\"text/javascript\" "
                     "src=\"/na/homes/%(user)s/Code/pylearn2/pylearn2/neuroimaging_utils/tools/tablesorter/js/widgets/widget-columnSelector.js\"></script>\n"
                     "<script type=\"text/javascript\">\n"
                     "$(document).ready(function() {\n"
                     "$(\"table\").tablesorter({\n"
                     "theme: 'blue',\n"
                     "widgets: ['zebra', 'columnSelector', 'stickyHeaders'],\n"
                     "debug: true,\n"
                     "widgetOptions : {\n"
                     "columnSelector_container : $('#columnSelector'),\n"
                     "columnSelector_columns : {\n"
                     "0: 'disable'\n"
                     "},\n"
                     "columnSelector_saveColumns: true,\n"
                     "columnSelector_layout : '<label><input type=\"checkbox\">{name}</label>',\n"
                     "columnSelector_name  : 'data-selector-name',\n"
                     "columnSelector_mediaquery: true,\n"
                     "columnSelector_mediaqueryName: 'Auto: ',\n"
                     "columnSelector_mediaqueryState: true,\n"
                     "columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],\n"
                     "columnSelector_priority : 'data-priority',\n"
                     "columnSelector_cssChecked : 'checked'\n"
                     "}\n"

                     "});\n" 
                     "$('#popover')\n"
                     ".popover({\n"
                     "placement: 'right',\n"
                     "html: true,\n"
                     "content: '<div id=\"popover-target\"></div>'\n"
                     "})\n"
                     ".on('shown.bs.popover', function () {\n"
                     "$.tablesorter.columnSelector.attachTo( $('.bootstrap-popup'), '#popover-target');\n"
                     "});\n"
                     "$(\".bootstrap-popup\").tablesorter({\n"
                     "theme: 'blue',\n"
                     "widgets: ['zebra', 'columnSelector', 'stickyHeaders']\n"
                     "});\n"
                     "});\n"
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

def html_column_names(model_keys, results_keys):
    column_name_string = (
        "<div class=\"columnSelectorWrapper\">\n"
	"<input id=\"colSelect1\" type=\"checkbox\" class=\"hidden\">\n"
	"<label class=\"columnSelectorButton\" for=\"colSelect1\">Column</label>\n"
	"<div id=\"columnSelector\" class=\"columnSelector\">\n"
	"</div>\n"
        "</div> (Select columns needed for analysis using this button)\n"
        "<table id=\"ResultsTable\" class=\"tablesorter\" border=\"0\" cellpadding=\"0\" cellspacing=\"1\">\n"
        "<thead>\n<tr>")
    for k in model_keys + results_keys:
        column_name_string += "\n\t<th>%s\t\n</th>" % k
    column_name_string +=  "\n</tr>\n</thead>\n<tbody>\n"
    return column_name_string

def html_rows(model_keys, results_keys, model_dict, results_dict):
    row_string = "<tr>"
    for k in model_keys:
        row_string += "\n\t<td>%s</td>" % model_dict.get(k, "-")
    row_string += "\n"
    for k in results_keys:
        row_string += "<td>%s</td>" % results_dict.get(k, "-")
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
    model_keys = sorted(list(set(k for keys in [md.keys() for md in model_dicts] for k in keys)))
    result_keys = sorted(list(set(k for keys in [rd.keys() for rd in results_dicts] for k in keys)))
    table_string += html_column_names(model_keys, result_keys)
    for md, rd in zip(model_dicts, results_dicts):
        table_string += html_rows(model_keys, result_keys, md, rd)
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
            model_dict = dict(("\n".join(k.split(".")), job.get("hyper_parameters." + k, None))
                              for k in model_keys if "__builder__" not in k)
            model_dict["status"] = job.status
            model_dict["id"] = job.id
            model_dict["file_prefix"] = file_prefix
            logger.info("Analyzing job %(id)d, with status %(status)d, "
                        "and file_prefix %(file_prefix)s"
                        % model_dict)
            model_dicts.append(model_dict)

            if job.status == 1 or args.reload:
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
    parser.add_argument("-r", "--reload", action="store_true")

    return parser

if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    print args
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    analyze(args)
