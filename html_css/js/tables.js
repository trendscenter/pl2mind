$(document).ready(function () {

    var rois = [];

    function makeModal() {
        var body = document.getElementsByTagName("body")[0];
	var modal_fade = document.createElement("div");
        body.appendChild(modal_fade);
	modal_fade.id = "modelModal";
	modal_fade.className = "modal fade";
	modal_fade.setAttribute("tabindex", "-1");
	modal_fade.setAttribute("role", "dialog");
	modal_fade.setAttribute("aria-labelledby", "showModalLabel");
	modal_fade.setAttribute("aria-hidden", "true");

	var modal_dialog = document.createElement("div");
	modal_dialog.className = "modal-dialog";
	modal_fade.appendChild(modal_dialog);

	var modal_content = document.createElement("div");
	modal_content.className = "modal-content";
	modal_content.id = "model_modal_content";
	modal_dialog.appendChild(modal_content);

	var modal_header = document.createElement("div");
	modal_header.className = "modal-header";
	modal_content.appendChild(modal_header);

	var button = document.createElement("button");
	button.type = "button";
	button.className = "close";
	button.setAttribute("data-dismiss", "modal");
	button.setAttribute("aria-label", "Close");
	var span = document.createElement("span");
	span.setAttribute("aria-hidden", "true");
	span.innerHTML = "&times;";
	button.appendChild(span);
	modal_header.appendChild(button);

	var title = document.createElement("h4");
	modal_header.appendChild(title);
    }

    function resolve_column_names(json) {
        column_names = []
        var ignores = ["jobman.hash", "jobman.experiment", "dbdescr",
                       "jobman.sql.starttime", "jobman.starttime",
                       "jobman.endtime", "experiment_module", "jobman.runtime",
                       "out_path", "jobman.sql.hostworkdir"];
        for (var i in json) {
            for (var name in json[i]) {
                if (($.inArray(name, column_names) == -1) &&
                    ($.inArray(name, ignores) == -1)) {
                    column_names.push(name);
                }
            }
        }
        return column_names;
    }

    function makeColumns(dict, column_names, empties) {
        var table_head = document.createElement("thead");
        var head_tr = document.createElement("tr");
        table_head.appendChild(head_tr);

        for (var c = 0; c < empties; ++ c) {
            var th = document.createElement("th");
	    th.className = "columnSelector-disable";
            head_tr.appendChild(th);
        }

        for (var c = 0; c < column_names.length; ++c) {
            original_name = column_names[c];
            name = original_name.replace("hyperparams.", "");
            name = name.replace("jobman.", "");
            name = name.replace("results.", "");
            name = name.replace("stats.", "");
            name = name.replace(/_/g, " ");
            name = name.replace(/\./g, " ");
            name = name.replace(/__builder__/g, " ");

            var th = document.createElement("th");
            th.innerHTML = name;

            var all_repeats = true;
            var first = dict[0][column_names[c]];
            for (var i = 1; i < dict.length; ++i) {
                if (dict[i][column_names[c]] != first) {
                    all_repeats = false;
                    break;
                }
            }
            if (dict.length == 1) {
                all_repeats = false;
            }
            if (all_repeats) {
                th.className = "columnSelector-false";
            }
            var is_roi = false;
            for (var i = 0; i < rois.length; ++i) {
                var roi = rois[i];
                if (original_name.indexOf(roi) != -1) {
                    is_roi = true;
                    break;
                }
            }
            if ((original_name.indexOf("results") != -1) && (!is_roi)) {
                th.className = "columnSelector-false";
            }
            th.id = column_names[c];
            head_tr.appendChild(th);
        }
        return table_head;
    }

    function makeRows(json, column_names) {
        var table_body = document.createElement("tbody");
        for (var elem in json) {
            var tr = document.createElement("tr");
            table_body.appendChild(tr);
            var buttons = make_buttons(json[elem]);
            for (var i = 0; i < buttons.length; ++i) {
                var button_td = buttons[i];
                tr.appendChild(button_td);
            }

            for (var c = 0; c < column_names.length; ++c) {
                var td = document.createElement("td");
                var value = json[elem][column_names[c]];
                if ((!isNaN(value)) && (value % 1 !== 0)) {
                    value = value.toExponential(2);
                }
                td.innerHTML = value;
                tr.appendChild(td);
                tr.className = column_names[c];
            }
        }
        return table_body;
    }

    function make_buttons(elem) {
        buttons = [];

        var view_button = document.createElement("td");
        var a = document.createElement("a");
        view_button.appendChild(a);
        a.className = "cd-popup-trigger";
        a.href = "#";
        a.setAttribute("data-toggle", "modal");
        a.setAttribute("data-target", "#modelModal");
        a.innerHTML = "VIEW";

        $(a).on("click", function(){
            var modal_content = document.getElementById("model_modal_content");
            $.ajax({
                url: elem["out_path"],
                type: "GET",
                success: function (p) {
                    while (modal_content.firstChild) {
                        modal_content.removeChild(modal_content.firstChild);
                    }
		    console.log(p);
                    setLayout(modal_content, p);
                    periodicUpdate(p);
                },
                error: function() {
                    while (modal_content.firstChild) {
                        modal_content.removeChild(modal_content.firstChild);
                    }
                    modal_content.innerHTML = "No model found.";
                }
            });
        });
        buttons.push(view_button);

        var log_button = document.createElement("td");
        var b = document.createElement("a");
        log_button.appendChild(b);
        b.className = "cd-popup-trigger";
        b.href = "#";
        b.setAttribute("data-toggle", "modal");
        b.setAttribute("data-target", "#modelModal");
        b.innerHTML = "LOGS";

        $(b).on("click", function(){
            var modal_content = document.getElementById("model_modal_content");
            console.log(elem["out_path"] + "/model.json");
            $.ajax({
                url: elem["out_path"] + "/model.json",
                type: "GET",
                dataType: "json",
                success: function (json) {
                    while (modal_content.firstChild) {
                        modal_content.removeChild(modal_content.firstChild);
                    }
                    var modal_body = document.createElement("div");
                    modal_body.className = "modal-body";
                    modal_content.appendChild(modal_body);
		    var pre = document.createElement("pre");
                    pre.innerHTML = json["log_stream"];
		    modal_body.appendChild(pre);
                },
                error: function() {
                    while (modal_content.firstChild) {
                        modal_content.removeChild(modal_content.firstChild);
                    }
                    modal_content.innerHTML = "No logs found.";
                }
            });
        });
        buttons.push(log_button);

        return buttons;
    }

    function get_rois(json) {
	console.log(json);
        for (var job in json) {
            console.log("Getting " + json[job]["out_path"] + "/model.json");
            $.ajax({
                url: json[job]["out_path"] + "/model.json",
                type: "GET",
                dataType: "json",
                async: false,
                success: function (json) {
                    for (var i in json["results_of_interest"]) {
                        var roi = json["results_of_interest"][i];
                        if (rois.indexOf(roi) == -1) {
                            rois.push(roi);
                        }
                    }
                }
            });
        }
    }

    function split_tables(json) {
        var split_json = {
            waiting: [],
            running: [],
            completed: [],
            failed: [],
            killed: []
        };
        for (var job in json) {
            switch (parseInt(json[job]["jobman.status"])) {
                case 0:
                    split_json.waiting.push(json[job]);
                    break;
                case 1:
                    split_json.running.push(json[job]);
                    break;
                case 2:
                    split_json.completed.push(json[job]);
                    break;
                case -1:
                    split_json.killed.push(json[job]);
                    break;
                default:
                    split_json.failed.push(json[job]);
                    break;
            }
        }
        return split_json;
    }

    var make_table = function(json, title, popup_id) {
        var div1 = document.createElement("div");
        div1.className = "table_div";
        var div2 = document.createElement("div");
        div2.className = "table_header_div"
        var div3a = document.createElement("div");
        div3a.className = "table_title_div";
        div3a.innerHTML = title.charAt(0).toUpperCase() +
            title.slice(1);
        div2.appendChild(div3a);
        var div3b = document.createElement("div");
        div3b.className = "columnSelectorWrapper";
        button = document.createElement("button");
        button.id = "popover" + popup_id.toString();
        button.type = "button";
        button.className = "btn btn-default";
        button.innerHTML = "Select";
        div3b.appendChild(button);
        div2.appendChild(div3b);
        div1.appendChild(div2);

        var table = document.createElement("table");
        div1.appendChild(table);
        table.className = "tablesorter bootstrap-popup" + popup_id.toString();
        var column_names = resolve_column_names(json);
        table_head = makeColumns(json, column_names, 2);
        table.appendChild(table_head);
        table_body = makeRows(json, column_names);
        table.appendChild(table_body);

        var test = popup_id;
        var id = test.toString();
        $(button)
            .popover({
                placement: 'right',
                html: true,
                content: '<div id="popover-target' + id + '"></div>'
            })
            .on('shown.bs.popover', function () {
                // call this function to copy the column selection code into the popover
                $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup' + id), '#popover-target' + id);
            });

        $(table).tablesorter({
            theme: 'blue',
            widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
            widgetOptions : {
                columnSelector_columns : {
                    0: 'disable'
                },
                columnSelector_saveColumns: true,
                columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
                columnSelector_mediaquery: false,
                columnSelector_mediaqueryName: 'Auto: ',
                columnSelector_mediaqueryState: true,
                columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
                columnSelector_priority : 'data-priority',
                columnSelector_cssChecked : 'checked'
            }
        });

        return div1;
    }

    function make_tables(json) {
        var body = document.getElementsByTagName("body")[0];
        tables = split_tables(json);
        var i = 0;
        for (var item in tables) {
            table = make_table(tables[item], item, i);
            body.appendChild(table);
            i += 1;
        }
    }

    function get_data() {
        console.log("Getting tables");
        $.ajax({
	    url: "/get_table",
	    type: "GET",
	    dataType: "json",
	    success: function (json) {
                console.log("Got table");
                get_rois(json);
                make_tables(json);
            },
            error: function(xhr, status, error) {
                var err = eval("(" + xhr.responseText + ")");
                console.log(err.Message);
                trigger.innerHTML = "FAILED";
            }
	});
    }
    makeModal();
    get_data();
});