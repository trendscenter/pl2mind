"""
Module to write html table.
"""

import logging


logging.basicConfig(format="[%(module)s:%(levelname)s]:%(message)s")
logger = logging.getLogger(__name__)

def html_header(title):
    header_string = (
        "<html>\n"
        "<head>\n"             
        "<link rel=\"stylesheet\" href=\"../html_css/blue/theme.blue.css\" type=\"text/css\" "
        "media=\"print, projection, screen\" />\n"
        "<link rel=\"stylesheet\" href=\"../html_css/popup_css/style.css\" type=\"text/css\">\n"
        "<link rel=\"stylesheet\" href=\"../html_css/column_selector.css\" type=\"text/css\">\n"
        "<link rel=\"stylesheet\" href=\"../html_css/bootstrap.min.css\">\n"
        "<script "
        "src=\"http://ajax.googleapis.com/ajax/libs/jquery/1.11.1/jquery.min.js\">"
        "</script>\n"
        "<script type=\"text/javascript\" "
        "src=\"../html_css/tablesorter/js/jquery.tablesorter.js\">"
        "</script>\n"
        "<script src=\"../html_css/js/bootstrap.min.js\"></script>\n"
        "<script src=\"../html_css/tablesorter/js/widgets/widget-columnSelector.js\">"
        "</script>\n" 
        "<script src=\"../html_css/js/popup.js\"></script>\n"
        "<script src=\"../html_css/js/table_sorter.js\"></script>\n"
        "<title>%(title)s</title>\n"
        "</head>\n"
        "<body id=\"main\">\n"
        "<div id=\"title_div\"><h1>%(title_capped)s</h1></div>"% {"title": title, "title_capped": title.upper()})
    return header_string

def html_elements():
    elements_string = (
        "<div class=\"modal fade\" id=\"showModal\" tabindex=\"-1\" "
        "role=\"dialog\" aria-labelledby=\"showModalLabel\" aria-hidden=\"true\">\n"
        "<div class=\"modal-dialog\">\n"
        "<div class=\"modal-content\">\n"
        "<div class=\"modal-header\">\n"
        "<button type=\"button\" class=\"close\" data-dismiss=\"modal\" "
        "aria-label=\"Close\"><span aria-hidden=\"true\">&times;</span></button>\n"
        "<h4 class=\"modal-title\" id=\"myModalLabel\">Results Graphs</h4>\n"
        "</div>\n"
        "<div class=\"modal-body\">\n"
	"</div>\n"
        "<div class=\"modal-footer\">\n"
        "<button type=\"button\" class=\"btn btn-default\" "
        "data-dismiss=\"modal\">Close</button>\n"
        "</div>\n"
        "</div>\n"
        "</div>\n"
        "</div>\n"
        "<div class=\"modal fade\" id=\"pdfModal\" tabindex=\"-1\" "
        "role=\"dialog\" aria-labelledby=\"pdfModalLabel\" aria-hidden=\"true\">\n"
        "<div class=\"modal-dialog\">\n"
        "<div class=\"modal-content\">\n"
        "<div class=\"modal-header\">\n"
        "<button type=\"button\" class=\"close\" data-dismiss=\"modal\" "
        "aria-label=\"Close\"><span aria-hidden=\"true\">&times;</span></button>\n"
        "<h4 class=\"modal-title\" id=\"pdfModalLabel\">PDF Modal</h4>\n"
        "</div>\n"
        "<div class=\"modal-body\">No pdf found</div>\n"
        "</div>\n"
        "</div>\n"
        "</div>\n"
        "</div>\n"
        )

    return elements_string

def html_column_names(model_keys, results_keys, 
                      non_priority_keys, priority_keys):
    column_name_string = "\n\t<th></th>"
    for k in model_keys:
        if k in non_priority_keys:
            column_name_string += "\n\t<th class=\"columnSelector-false\">%s\t\n</th>" % k.replace("_", " ")
        else:
            column_name_string += "\n\t<th>%s\t\n</th>" % k.replace("_", " ") 
    for k in results_keys:
        if k in priority_keys:
            column_name_string += "\n\t<th>%s\t\n</th>" % k.replace("_", " ")
        else:
            column_name_string += "\n\t<th class=\"columnSelector-false\">%s\t\n</th>" % k.replace("_", " ")
    return column_name_string

def html_rows(status, model_keys, results_keys, model_dict, results_dict):
    row_string = "<tr>"
    if status == 0:
        row_string += (
            "\n\t<td>"
            "<a class=\"cd-popup-trigger off\" href=\"#\">View</a></td>"
            )
    elif status in [1, 2]:
        process_string = (
            "\"python $PYLEARN2_DIR/neuroimaging_utils/tools/mri_analysis.py %s.pkl\"" 
            % model_dict["file_prefix"])
        row_string += (
            "\n\t<td>"
            "<a class=\"cd-popup-trigger dir_btn off\" "
            "href=\"#\" "
            "data-toggle=\"modal\" "
            "data-target=\"#showModal\" "
            "dir=\"%(path)s\" "
            "id=\"%(path)s\" "
            "error_script=%(process_string)s>View</a></td>" 
            % {"process_string": process_string,
               "path": model_dict["file_prefix"].split("/")[-1]
               })
    elif status == 3:
        row_string += (
            "\n\t<td>"
            "<a class=\"cd-popup-trigger log_btn\" "
            "data-toggle=\"modal\" "
            "data-target=\"#showModal\" "
            "href=\"#\" "
            "link=\"log_files/%(job_id)s/stderr\">View</a></td>"
            % {"job_id": model_dict["id"]}
            )
    for k in model_keys:
        if k == "file_prefix":
            row_string += "\n\t<td>%s</td>" % model_dict.get(k, "-").split("/")[-1]
        else:
            row_string += "\n\t<td>%s</td>" % model_dict.get(k, "-")
    row_string += "\n"
    for k in results_keys:
        row_string += "<td>%s</td>" % results_dict.get(k, "-")
    row_string += "\n</tr>\n"
    return row_string

def html_footer():
    footer_string = (
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
            table_string += html_rows(key, model_keys, result_keys, md, rd)
        table_string += ("</tbody>\n"
                         "</table>\n"
                         "</div>\n"
                         )
        self.table_strings.append(table_string)
    
    def clear(self):
        self.table_strings = []

    def write(self, filename):
        with open(filename, "w") as f:
            f.write(self.header_string)
            f.write(html_elements())
            for table_string in self.table_strings:
                f.write(table_string)
            f.write("</body>\n")
            f.write(self.footer_string)
        assert f.closed

    def write_table(self, d, row_keys, column_keys, filename, results_of_interest=None):
        with open(filename, "w") as f:
            if len(column_keys) > 0:
                if results_of_interest is not None:
                    roi_string = "\t".join(["1" if c in results_of_interest else "0" for c in column_keys]) + "\n"
                    f.write(roi_string)
                keys_string = "\t".join([" ".join(c.split("_")) for c in column_keys]) + "\n"
                f.write(keys_string)
            for row in row_keys:
                row_string = "\t".join([str(d[row].get(c, "-")) for c in column_keys])
                f.write(row_string)
                if row != row_keys[-1]:
                    f.write("\n")
