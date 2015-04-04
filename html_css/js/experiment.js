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

function makeTab(id, title, active) {
    var li = document.createElement("li");
    if (active) {
	li.className = "active";
    }
    li.id = id + "_tab";
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
    var tab_content = document.createElement("div");
    tab_content.className = "tab-content";
    tab_div.appendChild(tab_content);

    function makeInfoDiv(id, title, active) {
	var tab = makeTab(id, title, active);
	ul.appendChild(tab);

	// Makes a div element for info.
	var info_div = document.createElement("div");
	if (active) {
	info_div.className = "tab-pane fade in active";
	} else {
	    info_div.className = "tab-pane fade";
	}
	info_div.role = "tabpanel";
	info_div.id = id;

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

    var primary_plots = makePlots(ul, rois, "primary",
				  "Primary Stats", 0, false);
    tab_content.appendChild(primary_plots);

    var secondary_plots = makePlots(ul, extras, "secondary", "Secondary Stats",
				    rois.length, false);
    tab_content.appendChild(secondary_plots);

    var analysis = document.createElement("div");
    var anal_tab = makeTab("analysis", "Analysis", false);
    ul.appendChild(anal_tab);
    tab_content.appendChild(analysis);

    analysis.id = "analysis";
    analysis.className = "tab-pane fade";
    var anal_text = document.createElement("div");
    anal_text.id = "anal_text";
    anal_text.innerHTML = "None processed";
    analysis.appendChild(anal_text);
    analysis.role = "tabpanel";
    anal_ul = document.createElement("ul");
    anal_ul.role = "tablist";
    anal_ul.className = "nav nav-tabs";
    anal_ul.id = "analTab";
    analysis.appendChild(anal_ul);
    var anal_tab_content = document.createElement("div");
    anal_tab_content.className = "tab-content";
    anal_tab_content.id = "anal_tab_content";
    analysis.appendChild(anal_tab_content);

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
	while (info_div.firstChild) {
	    info_div.removeChild(info_div.firstChild);
	}
	var str = JSON.stringify(info, undefined, 2);
	str = str.replace(/"/g,"");
	str = str.replace(/,/g,"");
	str = str.replace(/{/g,"");
	str = str.replace(/}/g,"");
	str = str.replace(/\\/g,"");
	str = str.replace(/\[/g,"");
	str = str.replace(/\]/g,"");
	str = str.replace(/^([^0-9]*:)(.*)$/g, "<b>$1</b>$2");
	str = str.replace(/__builder__/g, "constructor");
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
	    updateModels(json, wd);
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
//	    var last_updated = document.getElementById("last_updated");
//	    last_updated.innerHTML = now;
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
	    getAnalysis(wd);

	    var now = new Date();
//	    var last_updated = document.getElementById("last_updated");
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