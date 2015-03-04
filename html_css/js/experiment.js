function makeModal(wd) {
    var modal_fade = document.createElement("div");
    modal_fade.id = "confirmModal";
    modal_fade.className = "modal fade";
    modal_fade.setAttribute("tabindex", "-1");
    modal_fade.setAttribute("role", "dialog");
    modal_fade.setAttribute("aria-labelledby", "showModalLabel");
    modal_fade.setAttribute("aria-hidden", "true");

    var modal_dialog = document.createElement("div");
    modal_dialog.className = "modal-dialog";
    modal_dialog.id = "modal_dialog";
    modal_fade.appendChild(modal_dialog);

    var modal_content = document.createElement("div");
    modal_content.className = "modal-content";
    modal_dialog.appendChild(modal_content);
    modal_content.id = "modal_content";

    var modal_header = document.createElement("div");
    modal_header.className = "modal-header";
    modal_content.appendChild(modal_header);
    modal_header.id = "modal_header";

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
    title.id = "modal_title";

    var modal_body = document.createElement("div");
    modal_body.className = "modal-body";
    modal_body.id = "modal_body";
    modal_content.appendChild(modal_body);

    return modal_fade;
}

function clearModalBody() {
    var modal_body = document.getElementById("modal_body");
    while (modal_body.firstChild) {
	modal_body.removeChild(modal_body.firstChild);
    }
    return modal_body;
}

function switchFeatureModal(obj) {
    console.log(obj);
    var modal_body = clearModalBody();
    modal_body.innerHTML = obj.index;
}

function switchInfoModal(title, info) {
    var modal_title = document.getElementById("modal_title");
    modal_title.innerHTML = title;

    var modal_body = clearModalBody();
    var pre = document.createElement("pre");
    pre.innerHTML = info;
    modal_body.appendChild(pre);

    $("#modal_dialog").addClass("large");

    if ($("#modal_footer")) {
	$("#modal_footer").remove();
    }
}

function switchDecisionModal(query, title) {

    function makeModalFooter() {
	$("#modal_dialog").removeClass("large");
	var modal_footer = document.createElement("div");
	modal_footer.className = "modal-footer";
	modal_footer.id = "modal_footer";

	var yes = document.createElement("button");
	yes.className = "btn btn-default";
	yes.setAttribute("data-dismiss", "modal");
	yes.innerHTML = "Yes";
	yes.id = "yes_button";
	modal_footer.appendChild(yes);

	var no = document.createElement("button");
	no.className = "btn btn-primary";
	no.innerHTML = "Cancel";
	no.setAttribute("data-dismiss", "modal");
	modal_footer.appendChild(no);
	return modal_footer;
    }

    var modal_title = document.getElementById("modal_title");
    modal_title.innerHTML = title;

    var modal_body = clearModalBody();
    modal_body.innerHTML = query;

    if (! $("#modal_footer")[0]) {
	var modal_footer = makeModalFooter();
	$("#modal_content").append($(modal_footer));
    }
    var yes = document.getElementById("yes_button");

    return yes;
}

function processModel(wd, button, a, port) {
    a.innerHTML = "Running";
    $.ajax({
	url: "/processme",
	type: "POST",
	dataType: "json",
	data: JSON.stringify({id: port}),
	contentType: "application/json",
	timeout: 1000000,
	complete: function() {
	    updateAll(wd);
	},
	success: function(data) {
	    console.log("Got " + data.response);
	    if (data.response == "SUCCESS") {
		a.innerHTML = "Process";
	    } else if (data.respond == "Already") {
		a.innerHTML = "Please wait";
	    } else {
		a.innerHTML = "Failed";
	    }
	    updateAll(wd);
	},
	error: function(xhr, status, error) {
	    a.innerHTML = "Failed";
	    var err = eval("(" + xhr.responseText + ")");
	    console.log(err.Message);
	}
    });
}

function killProcess(wd, button, a, port) {
    a.innerHTML = "STOPPING";
    $.ajax({
	url: "/killme",
	type: "POST",
	dataType: "json",
	timeout: 1000,
	data: JSON.stringify({id: port}),
	contentType: "application/json",
	success: function(data) {
	    console.log(data.response);
	    if (data.response == "OK") {
		$(button).addClass("is-hidden");
	    }
	    a.innerHTML = "STOP";
	    updateAll(wd);
	},
	error: function(xhr, status, error) {
	    a.innerHTML = "FAILED";
	    var err = eval("(" + xhr.responseText + ")");
	    console.log(err.Message);
	}
    });
}

function linkUp(str, prefix) {
    var re = new RegExp("\(" + prefix + "\)\(pylearn2\|pl2mind\|nice\)\(\\S+\)","g");
    // /(!obj:)(pylearn2|pl2mind|nice)(\S+)/g
    str = str.replace(
	re,
	function(a, b, c, d) {
	    if (c == "nice") {
		clink = "https://github.com/rdevon/nice/tree/master/";
	    } else if (c == "pl2mind") {
		clink = "https://github.com/MRN-Code/pl2mind/tree/master/";
	    } else if (c == "pylearn2") {
		clink = "https://github.com/rdevon/pylearn2/tree/nitools/pylearn2";
	    }
	    var dlink = d.replace(/\./g, "/");
	    dlink = dlink.replace(/\/[^\/]+$/g, ".py");
	    var s = b + "<a target=\"_blank\" href=" + clink + dlink + ">" + c + d + "</a>";
	    return s;
	});
    str = str.replace("termination_criteria.py", "termination_criteria/__init__.py");
    return str;
}

function makeButtons(wd, json, port) {

    function makeDecisionButton(text, title, query, func, id) {
	var button = document.createElement("div");
	button.className = "button_div";

	var a = document.createElement("a");
	a.href = "#";
	a.className = "trigger";
	a.id = id;
	a.setAttribute("data-toggle", "modal");
	a.setAttribute("data-target", "#confirmModal");
	a.innerHTML = text;

	button.appendChild(a);
	button.onclick = function() {
	    var yes = switchDecisionModal(query, title)
	    yes.onclick = function() {
		func(wd, this, a, port);
	    }
	}

	return button;
    }

    function makeInfoButton(text, title, info) {
	var button = document.createElement("div");
	button.className = "button_div";

	var a = document.createElement("a");
	a.href = "#";
	a.className = "trigger";
	a.setAttribute("data-toggle", "modal");
	a.setAttribute("data-target", "#confirmModal");
	a.innerHTML = text;

	button.appendChild(a);
	button.onclick = function() {
	    switchInfoModal(title, info);
	}

	return button;
    }

    function makeSwitchButton(on_text, off_text, func) {
	var button = document.createElement("div");
	button.className = "button_div";

	var a = document.createElement("a");
	a.href = "#";
	a.className = "trigger";
	a.innerHTML = off_text;

	$(button).on("click", function() {
	    $(this).toggleClass("on");
	    func();
	});
    }

    var stats = document.getElementById("stats");
    var kill = makeDecisionButton("STOP", "",
				  "Kill process? (Cannot be undone)",
				  killProcess, "kill_trigger");
    stats.firstChild.appendChild(kill);

    var yaml_str = json.yaml;
    yaml_str = linkUp(yaml_str, "!obj:");

    var yaml = makeInfoButton("YAML", "YAML", yaml_str);
    stats.firstChild.appendChild(yaml);

    var logs = makeInfoButton("Logs", "Logs", json.log_stream);
    stats.firstChild.appendChild(logs);

    var process = makeDecisionButton("Process", "", "Process model?",
			     processModel, "process_trigger");
    //var last_processed = document.createElement("div");
    //last_processed.innerHTML = "Last processed: No info";
    //process.appendChild(last_processed);
    var tabs = document.getElementById("myTab");
    tabs.appendChild(process);
/*
    var live = makeButton("live-trigger long-trigger", "live_button",
			  "GO-LIVE", false);
    var title = document.getElementById("title_div");
    var live_div = document.createElement("div");
    live_div.id = "last_updated";
    live.appendChild(live_div);
    title.appendChild(live);

    periodicUpdate(wd); */
}

function makePlots(group, id, title, start_idx, active) {
    // Make plot divs.
    var div = document.createElement("div");

    if (active) {
	div.className = "tab-pane fade in active";
    } else {
	div.className = "tab-pane fade";
    }

    div.role = "tabpanel";
    div.id = id;
    var h = document.createElement("h3");
    h.innerHTML = title;
    div.appendChild(h);

    function makePlotDiv(name, id) {
	// Make div for plot.
	var div1 = document.createElement("div");
	div1.className = "plot_content_div";
	var h1 = document.createElement("h3");
	h1.innerHTML = name.replace(/_/g, " ");
	div1.appendChild(h1);
	var div = document.createElement("div");
	div.id = "plot" + id.toString();
	div.className = "plot_div";
	div1.appendChild(div);
	return div1;
    }

    for (var g = 0; g < group.length; ++g) {
	var name = group[g];
	name = name.split("/").pop();
	name = name.split(".")[0];

	var plot_div = makePlotDiv(name, g + start_idx);
	div.appendChild(plot_div);
    }
    return div;
}

function makeTab(id, title, active) {
    var li = document.createElement("li");
    if (active) {
	li.className = "active";
    }
    var a = document.createElement("a");
    a.href = "#" + id;
    a.setAttribute("aria-controls", id);
    a.role = "tab";
    a.setAttribute("data-toggle", "tab");
    a.innerHTML = title
    li.appendChild(a);
    return li;
}

function makeLayout(body, json, wd) {
    // Makes the basic layout as defined by the JSON file.
    var stats = json.stats;
    var logs = json.logs;
    var rois = json.results_of_interest;
    var hyperparams = json.hyperparams;
    var name = json.name;
    var results = json.outputs;
    var port = stats.port;

    var extras = [];
    for (var log_group in logs) {
	if (logs.hasOwnProperty(log_group)
	    && ($.inArray(log_group, rois) == -1)) {
	    extras.push(log_group);
	}
    }

    var title_div = document.createElement("div");
    title_div.id = "title_div";
    var title = document.createElement("h1");
    title.innerHTML = name;
    title_div.appendChild(title);
    body.appendChild(title_div);

    var tab_div = document.createElement("div");
    tab_div.role = "tabpanel";
    tab_div.id = "tabs"
    body.appendChild(tab_div);
    ul = document.createElement("ul");
    ul.role = "tablist";
    ul.className = "nav nav-tabs";
    ul.id = "myTab";
    tab_div.appendChild(ul);

    var info_tab = makeTab("stats", "Model Info", true);
    ul.appendChild(info_tab);

    var hyperparams_tab = makeTab("hyperparams", "Hyperparams", false);
    ul.appendChild(hyperparams_tab);

    var primary_tab = makeTab("primary", "Primary Stats", false);
    ul.appendChild(primary_tab);

    var secondary_tab = makeTab("secondary", "Secondary Stats", false);
    ul.appendChild(secondary_tab);

    var analysis_tab = makeTab("analysis", "Analysis", false);
    ul.appendChild(analysis_tab);

    /*
    for (var result in results) {
	var result_tab = makeTab(result,
				 result.charAt(0).toUpperCase() +
				 result.slice(1), false);
	ul.appendChild(result_tab);
    }*/

    var tab_content = document.createElement("div");
    tab_content.className = "tab-content";
    tab_div.appendChild(tab_content);

    function makeInfoDiv(id, title, active) {
	// Makes a div element for info.
	var info_div = document.createElement("div");
	if (active) {
	info_div.className = "tab-pane fade in active";
	} else {
	    info_div.className = "tab-pane fade";
	}
	info_div.role = "tabpanel";
	info_div.id = id;
	//info_div.className = "plot_group_div";
	var info_title = document.createElement("div");
	info_title.className = "info_title";
	var hi = document.createElement("h3");
	hi.innerHTML = title;
	info_title.appendChild(hi);
	info_div.appendChild(info_title);
	var content_div = document.createElement("div");
	content_div.id = id + "_content";
	info_div.appendChild(content_div);

	return info_div;
    }

    var stat_div = makeInfoDiv("stats", "General Info", true);
    tab_content.appendChild(stat_div);

    var param_div = makeInfoDiv("hyperparams",
				"Model Hyperparameters", false);
    tab_content.appendChild(param_div);

    var primary_plots = makePlots(rois, "primary",
				  "Primary Stats", 0, false);
    tab_content.appendChild(primary_plots);
    var secondary_plots = makePlots(extras, "secondary", "Secondary Stats",
				    rois.length, false);
    tab_content.appendChild(secondary_plots);

    console.log("Making analysis div");
    var analysis = document.createElement("div");
    analysis.id = "analysis";
    analysis.className = "tab-pane fade";
    tab_content.appendChild(analysis);

    /*
    for (var result in results) {
	var result_div = document.createElement("div");
	result_div.role = "tabpanel";
	result_div.id = result;
	result_div.className = "tab-pane fade"
	var pdf_div = document.createElement("div");
	pdf_div.innerHTML = "Loading...";
	result_div.appendChild(pdf_div);
	tab_content.appendChild(result_div);
    }*/

    var modal = makeModal(wd, port);
    makeButtons(wd, json, port);
    body.appendChild(modal);
}

function updatePdf(result, location, wd) {
    var pdf_object = document.createElement("object");
    pdf_object.className = "pdf_object";

    pdf_object.data = wd + "/" + location;
    pdf_object.type = "application/pdf";
    var pdf_div = $("#" + result).children()[0];
    $.ajax({
	url: wd + "/" +  location,
	type: "GET",
	success: function(d) {
	    $(pdf_div).replaceWith(pdf_object);
	},
	error: function(d) {
	    $(pdf_div).replaceWith("<div>" +
				   result.charAt(0).toUpperCase() +
				   result.slice(1) +
				   " not found. Try Processing</div>");
	}
    });
}

function updateAnalysis(json, wd) {
    console.log("Updating analysis");
    var analysis = document.getElementById("analysis");
    while (analysis.firstChild) {
	analysis.removeChild(analysis.firstChild);
    }

    function plot_image(image, id) {
	var data = [[[image, 0, 0, 10, 10]]];
	var options = {
	    series: {
		images: {
		    margin: 0,
		    show: true
		}
	    },
	    xaxis: {
		show: false
	    },
	    yaxis: {
		show: false
	    },
	    grid: {
		borderWidth: 0,
		minBorderMargin: 5
	    }
	};

	$.plot.image.loadDataImages(data, options, function () {
	    $.plot("#" + id, data, options);
	});
    }

    for (var model in json) {
	var div = document.createElement("div");
	div.className = "features_div";
	analysis.appendChild(div);
	for (var feature in json[model].features) {
	    var fdiv = document.createElement("div");
	    fdiv.className = "feature_div";
	    fdiv.id = "feature_" + feature.toString();
	    fdiv.href = "#";
	    fdiv.setAttribute("data-toggle", "modal");
	    fdiv.setAttribute("data-target", "#confirmModal");
	    fdiv.setAttribute("num", feature);
	    fdiv.onclick = function() {
		switchFeatureModal(json[model].features[this.getAttribute("num")]);
	    }
	    plot_image(json[model].features[feature].image, fdiv.id);
	    //fdiv.innerHTML = json[model].features[feature].index;
	    div.appendChild(fdiv);
	}
    }
}

function updateFromJSON(json, wd) {
    console.log("Updating from json");
    // Update the elements from the JSON file.
    var stats = json.stats;
    var logs = json.logs;
    var rois = json.results_of_interest;
    var hyperparams = json.hyperparams;
    var name = json.name;
    var results = json.outputs;
    var processing = json.processing;
    var last_processed = json.last_processed;

    function updateInfo(info, info_div) {
	var str = JSON.stringify(info, undefined, 2);
	str = str.replace(/"/g,"");
	str = str.replace(/,/g,"");
	str = str.replace(/{/g,"");
	str = str.replace(/}/g,"");
	str = str.replace(/\\/g,"");
	str = str.replace(/\[/g,"");
	str = str.replace(/\]/g,"");
	str = str.replace(/^([^0-9]*:)(.*)$/g, "<b>$1</b>$2");
	str = str.replace(/__builder__/g,"constructor");
	str = linkUp(str, ": ");
	var pre = document.createElement("pre");
	pre.innerHTML = str;
	info_div.appendChild(pre);
    }

    var stat_div = document.getElementById("stats_content");
    updateInfo(stats, stat_div);

    var param_div = document.getElementById("hyperparams_content")
    updateInfo(hyperparams, param_div);

    var extras = [];
    for (var log_group in logs) {
	if (logs.hasOwnProperty(log_group)
	    && ($.inArray(log_group, rois) == -1)) {
	    extras.push(log_group);
	}
    }

    var options = {
	lines: { show: true },
	points: { show: true , radius: 1},
	grid: {
	    hoverable: true
	}
    };

    function plotGroup(group, id) {
	stats = [];
	for (var stat in group) {
	    if (group.hasOwnProperty(stat)) {
		stats.push(stat);
	    }
	}
	var plot_data = [];
	var min = 0;
	var max = 0;
	for (var s = 0; s < stats.length; ++s) {
	    stat_plot = [];
	    stat = stats[s];
	    values = group[stat];
	    var label;
	    label = stat;

	    for (var i = 0; i < values.length; ++i) {
		var value = parseFloat(values[i]);
		if (value > max) {
		max = value;
		} else if (value < min) {
		    min = value;
		}
		stat_plot.push([i, value]);
	    }
	    plot_data.push({
		label: label,
		data: stat_plot
	    });
	}
	options.min = min;
	options.max = max;
	options.series = {
            shadowSize: 0,
            downsample: { threshold: 100 }
        };
	var placeholder = $("#plot" + id.toString());
	console.log("Actual plot");
	$.plot(placeholder, plot_data, options);

	placeholder.bind("plothover", function (event, pos, item) {
	    if (item) {
		var x = item.datapoint[0].toFixed(2),
		y = item.datapoint[1].toFixed(2);

		$("#tooltip").html(item.series.label +
				   "(" + parseInt(x) + "): " + y)
		    .css({top: item.pageY+5, left: item.pageX+5})
		    .fadeIn(200);
	    } else {
	    $("#tooltip").hide();
	    }
	});
    }

    for (var g = 0; g < rois.length; ++g) {
	group = logs[rois[g]];
	plotGroup(group, g);
    }

    for (var g = 0; g < extras.length; ++g) {
	group = logs[extras[g]];
	plotGroup(group, g + rois.length);
    }
    /*
    for (var result in results) {
	updatePdf(result, results[result], wd);
    }*/

    $("<div id='tooltip'></div>").css({
	position: "absolute",
	display: "none",
	color: "#111",
	border: "1px solid #bbb",
	padding: "2px",
	"border-radius": "2px",
	"background-color": "#eee",
	opacity: .9
    }).appendTo("body");
}

function setLayout(body, wd) {
    $.ajax({
	url: wd + "/model.json",
	type: "GET",
	dataType: "json",
	success: function (json) {
	    makeLayout(body, json, wd);
	}
    });
}

function getAnalysis(wd) {
    console.log("Getting analysis");
    $.ajax({
	url: wd + "/analysis.json",
	type: "GET",
	dataType: "json",
	success: function (json) {
	    console.log("Found analysis json");
	    updateAnalysis(json, wd);
	},
	error: function() {
	    console.log("Error on analysis");
	}
    });
}

function updateAll(wd) {
    $.ajax({
	url: wd + "/model.json",
	type: "GET",
	dataType: "json",
	success: function (json) {
	    updateFromJSON(json, wd);
	    var now = new Date();
	    var last_updated = document.getElementById("last_updated");
	    last_updated.innerHTML = now;
	}
    });
}

function periodicUpdate(wd) {
    console.log("Loading ajax");
    $.ajax({
	url: wd + "/model.json",
	type: "GET",
	dataType: "json",
	success: function (json) {
	    updateFromJSON(json, wd);
	    getAnalysis(".");

	    var now = new Date();
	    var last_updated = document.getElementById("last_updated");
//TODO	    last_updated.innerHTML = "(Last updated: " + now + ")";
	},
	complete: function() {
	    if ($(".live-trigger").hasClass("live-on")) {
		console.log("live update");
		setTimeout(periodicUpdate, 30000);
	    } else {
		console.log("not live");
	    }
	}
    });
};