$(document).ready(function() {
    var port;

    function makeModal() {
	var modal_fade = document.createElement("div");
	modal_fade.id = "confirmModal";
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

	var modal_body = document.createElement("div");
	modal_body.className = "modal-footer";
	modal_content.appendChild(modal_body);
	var yes = document.createElement("button");
	yes.className = "btn btn-default";
	yes.setAttribute("data-dismiss", "modal");
	yes.innerHTML = "Yes";
	yes.id = "yes_btn";
	var no = document.createElement("button");
	no.className = "btn btn-primary";
	no.innerHTML = "Cancel";
	no.setAttribute("data-dismiss", "modal");
	modal_body.appendChild(no);
	modal_body.appendChild(yes);
	modal_body.id = "modal_body";

	$(".live-trigger").on("click", function() {
	    if (this.className == "live-trigger long-trigger") {
		this.innerHTML = "LIVE";
		this.className = "live-trigger long-trigger live-on";
		periodicUpdate();
	    } else {
		this.innerHTML = "GO-LIVE";
		this.className = "live-trigger long-trigger";
	    }
	});

	$(".kill-trigger").on("click", function() {
	    title.innerHTML = "Kill process? (cannot be undone.)";
	    var button = document.getElementById("yes_btn");
	    button.onclick = function() {
		var kill = document.getElementById("kill_button");
		var trigger = kill.children[0];
		trigger.innerHTML = "STOPPING";
		$.ajax({
		    url: "/killme",
		    type: "POST",
		    dataType: "json",
		    data: JSON.stringify({id: port}),
		    contentType: "application/json",
		    success: function(data) {
			console.log(data.response);
			if (data.response == "OK") {
			    console.log("Got OK");
			    $(trigger).addClass("is-hidden");
			}
			trigger.innerHTML = "STOP";
			updateAll();
		    },
		    error: function(xhr, status, error) {
			var err = eval("(" + xhr.responseText + ")");
			console.log(err.Message);
			trigger.innerHTML = "FAILED";
		    }
		});
	    }
	});

	$(".process-trigger").on("click", function() {
	    title.innerHTML = "Process model?";
	    var button = document.getElementById("yes_btn");
	    button.onclick = function() {
		var process = document.getElementById("process_button");
		var trigger = process.children[0];
		trigger.innerHTML = "Running";
		trigger.className = "process-trigger long-trigger";
		$.ajax({
		    url: "/processme",
		    type: "POST",
		    dataType: "json",
		    data: JSON.stringify({id: port}),
		    contentType: "application/json",
		    timeout: 1000000,
		    complete: function() {
			updateAll();
		    },
		    success: function(data) {
			console.log("Got " + data.response);
			if (data.response == "SUCCESS") {
			    trigger.innerHTML = "Process";
			} else {
			    trigger.innerHTML = "Failed";
			}
			updateAll();
			trigger.className = "process-trigger";
		    },
		    error: function(xhr, status, error) {
			trigger.innerHTML = "Failed";
			trigger.className = "process-trigger";
			var err = eval("(" + xhr.responseText + ")");
			console.log(err.Message);
		    }
		});
	    }
	});

	return modal_fade;
    }

    function makeButtons() {
	function makeButton(classname, id, text, modal) {
	    var button = document.createElement("div");
	    button.className = "button_div";
	    button.id = id;

	    var a = document.createElement("a");
	    a.className = classname;
	    a.href = "#";
	    if (modal) {
		a.setAttribute("data-toggle", "modal");
		a.setAttribute("data-target", "#confirmModal");
	    }
	    a.innerHTML = text;

	    button.appendChild(a);

	    return button;
	}

	var kill = makeButton("kill-trigger", "kill_button", "STOP", true);
	var stats = document.getElementById("stats");
	stats.appendChild(kill);

	var process = makeButton("process-trigger", "process_button",
				 "Process", true);
	var tabs = document.getElementById("myTab");
	tabs.appendChild(process);

	var live = makeButton("live-trigger long-trigger", "live_button",
			      "GO-LIVE", false);
	var title = document.getElementById("title_div");
	var live_div = document.createElement("div");
	live_div.id = "last_updated";
	live.appendChild(live_div);
	title.appendChild(live);
    }

    function makePlots(group, id, title, start_idx, active) {
	// Make plot divs.
	var div = document.createElement("div");

	if (active) {
	    div.className = "tab-pane fade in active";
	} else {
	    div.className = "tab-pane fade active";
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

    function makeLayout(json) {
	// Makes the basic layout as defined by the JSON file.
	var stats = json.stats;
	var logs = json.logs;
	var rois = json.results_of_interest;
	var hyperparams = json.hyperparams;
	var name = json.name;
	var results = json.outputs;

	var extras = [];
	for (var log_group in logs) {
	    if (logs.hasOwnProperty(log_group)
		&& ($.inArray(log_group, rois) == -1)) {
		extras.push(log_group);
	    }
	}

	var body = document.getElementsByTagName("body")[0];
	var title_div = document.createElement("div");
	title_div.id = "title_div";
	var title = document.createElement("h1");
	title.innerHTML = name;
	title_div.appendChild(title);
	body.appendChild(title_div);

	function makeInfoDiv(info, id, title) {
	    // Makes a div element for info.
	    var info_div = document.createElement("div");
	    info_div.id = id;
	    info_div.className = "plot_group_div";
	    var info_title = document.createElement("div");
	    var hi = document.createElement("h3");
	    hi.innerHTML = title;
	    info_title.appendChild(hi);
	    info_div.appendChild(info_title);
	    var content_div = document.createElement("div");
	    content_div.id = id + "_content";
	    info_div.appendChild(content_div);

	    return info_div;
	}

	var stat_div = makeInfoDiv(stats, "stats", "General Info");
	body.appendChild(stat_div);

	var param_div = makeInfoDiv(hyperparams, "hyperparams",
				    "Model Hyperparameters");
	body.appendChild(param_div);

	var tab_div = document.createElement("div");
	tab_div.role = "tabpanel";
	tab_div.id = "tabs"
	body.appendChild(tab_div);
	ul = document.createElement("ul");
	ul.role = "tablist";
	ul.className = "nav nav-tabs";
	ul.id = "myTab";
	tab_div.appendChild(ul);
	var primary_tab = makeTab("primary", "Primary Stats", true);
	ul.appendChild(primary_tab);
	var secondary_tab = makeTab("secondary", "Seconary Stats", false);
	ul.appendChild(secondary_tab);

	for (var result in results) {
	    var result_tab = makeTab(result,
				     result.charAt(0).toUpperCase() +
				     result.slice(1), false);
	    ul.appendChild(result_tab);
	}

	var tab_content = document.createElement("div");
	tab_content.className = "tab-content";
	tab_div.appendChild(tab_content);

	var primary_plots = makePlots(rois, "primary",
				      "Primary Stats", 0, true);
	tab_content.appendChild(primary_plots);
	var secondary_plots = makePlots(extras, "secondary", "Secondary Stats",
					rois.length, false);
	tab_content.appendChild(secondary_plots);

	for (var result in results) {
	    var result_div = document.createElement("div");
	    result_div.role = "tabpanel";
	    result_div.id = result;
	    result_div.className = "tab-pane fade active"
	    var pdf_div = document.createElement("div");
	    pdf_div.innerHTML = "No PDF";
	    result_div.appendChild(pdf_div);
	    tab_content.appendChild(result_div);
	}

	makeButtons();
	var modal = makeModal();
	body.appendChild(modal);
    }

    function updatePdf(result, location) {
	var pdf_object = document.createElement("object");
	pdf_object.className = "pdf_object";
	pdf_object.data = location;
	pdf_object.type = "application/pdf";
	var pdf_div = $("#" + result).children()[0];
	$.ajax({
	    url: location,
	    type: "GET",
	    success: function(d) {
		$(pdf_div).replaceWith(pdf_object);
	    },
	    error: function(d) {
		$(pdf_div).replaceWith("<div>" + result +
				       " not found. Try Processing</div>");
	    }
	});
    }

    function updateFromJSON(json) {
	console.log("Updating from json");
	// Update the elements from the JSON file.
	var stats = json.stats;
	var logs = json.logs;
	var rois = json.results_of_interest;
	var hyperparams = json.hyperparams;
	var name = json.name;
	var results = json.outputs;
	var processing = json.processing;
	port = stats.port;

	function updateInfo(info, info_div) {
	    while (info_div.firstChild) {
		info_div.removeChild(info_div.firstChild);
	    }
	    for (var e in info) {
		var stat_div = document.createElement("div");
		if (e == "status") {
		    stat_div.id = e;
		}
		stat_div.className = "param_div";
		stat_div.innerHTML = e + ": " +
		    JSON.stringify(info[e]).replace(/"/g,"");
		info_div.appendChild(stat_div);
	    }
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
	    console.log("Plotting group ");
	    console.log(group);
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
		/*if (stat.indexOf("train") > -1) {
		    label = "train";
		} else if (stat.indexOf("valid") > -1) {
		    label = "valid";
		} else if (stat.indexOf("test") > -1) {
		    label = "test";
		} else {
		    label = stat;
		}*/
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

	for (var result in results) {
	    updatePdf(result, results[result]);
	}

	updateButtons(processing);

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

    function updateButtons(processing) {
	var kill = document.getElementById("kill_button");
	var kill_button = kill.children[0];
	kill_button.innerHTML = "STOP";
	var status = document.getElementById("status");
	var text = status.childNodes[0];
	if ((text.textContent.indexOf("KILLED") > -1) ||
	    (text.textContent.indexOf("COMPLETED") > -1)) {
	    kill.className = "button_div is-hidden";
	} else {
	    kill.className = "button_div";
	}

	var process = document.getElementById("process_button");
	var process_button = process.children[0];
	if (processing) {
	    process_button.innerHTML = "Running";
	} else {
	    process_button.innerHTML = "Process";
	}
    }

    function setLayout() {
	$.ajax({
	    url: "model.json",
	    type: "GET",
	    dataType: "json",
	    success: function (json) {
		makeLayout(json);
	    }
	});
    }

    function updateAll() {
	$.ajax({
	    url: "model.json",
	    type: "GET",
	    dataType: "json",
	    success: function (json) {
		updateFromJSON(json);
		var now = new Date();
		var last_updated = document.getElementById("last_updated");
		last_updated.innerHTML = now;
	    }
	});
    }

    function periodicUpdate() {
	console.log("Loading ajax");
	$.ajax({
	    url: "model.json",
	    type: "GET",
	    dataType: "json",
	    success: function (json) {
		updateFromJSON(json);
		var now = new Date();
		var last_updated = document.getElementById("last_updated");
		last_updated.innerHTML = "(Last updated: " + now + ")";
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
    setLayout();
    periodicUpdate();
});