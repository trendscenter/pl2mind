function getLogs(dirname) {
    var myRegexp = /href="([^"]*[\.log])"/g;
    var foldertxt;
    $.ajax({
	url: dirname,
	type: "get",
	async: false,
	success: function(txt) {
            foldertxt = txt;
	},
	error:function() {
	    return new Array();
	}
    });
    var fList = new Array();
    match = myRegexp.exec(foldertxt);
    while (match != null) {
	fList.push(match[1]);
	match = myRegexp.exec(foldertxt);
    }
    return fList;
}

$(function() {
    var stats;
    var logs;
    var rois;
    var hyperparams;
    var name;

    $.ajax({
            url: "/experiments/test_experiment/model.json",
            type: "GET",
            async: false,
            dataType: "json",
            success: function (json) {
                stats = json.stats;
                logs = json.logs;
                rois = json.results_of_interest;
                hyperparams = json.hyperparams;
                name = json.name;
            }
        });

    var body = document.getElementsByTagName("body")[0];
    var title_div = document.createElement("div");
    title_div.id = "title_div";
    var title = document.createElement("h1");
    title.innerHTML = name;
    title_div.appendChild(title);
    body.appendChild(title_div);

    var info_div = document.createElement("div");
    info_div.className = "plot_group_div";
    var hi = document.createElement("h2");
    hi.innerHTML = "Info";
    info_div.appendChild(hi);
    for (var stat in stats) {
        var stat_div = document.createElement("div");
        stat_div.className = "param_div";
        stat_div.innerHTML = stat + ": " + stats[stat];
        info_div.appendChild(stat_div);
    }
    body.appendChild(info_div);

    var hp_div = document.createElement("div");
    hp_div.className = "plot_group_div";
    var hp = document.createElement("h2");
    hp.innerHTML = "Hyper parameters";
    hp_div.appendChild(hp);
    for (var param in hyperparams) {
        var param_div = document.createElement("div");
        param_div.className = "param_div";
        param_div.innerHTML = param + ": " + hyperparams[param];
        hp_div.appendChild(param_div);
    }
    body.appendChild(hp_div);

    var options = {
        lines: { show: true },
        points: { show: true , radius: 1},
        grid: {
            hoverable: true
        }
    };

    function makePlot(name, id) {
        var div1 = document.createElement("div");
        div1.className = "plot_content_div";
        var h1 = document.createElement("h2");
        h1.innerHTML = name.replace(/_/g, " ");
        div1.appendChild(h1);
        var div = document.createElement("div");
        div.id = "plot" + id.toString();
        div.className = "plot_div";
        div1.appendChild(div);
        return div1;
    }

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
            if (stat.indexOf("train") > -1) {
                label = "train";
            } else if (stat.indexOf("valid") > -1) {
                label = "valid";
            } else if (stat.indexOf("test") > -1) {
                label = "test";
            } else {
                label = stat;
            }

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
        $.plot(placeholder, plot_data, options);

        placeholder.bind("plothover", function (event, pos, item) {
            if (item) {
                var x = item.datapoint[0].toFixed(2),
                y = item.datapoint[1].toFixed(2);

                $("#tooltip").html(item.series.label + "(" + parseInt(x) + "): " + y)
                    .css({top: item.pageY+5, left: item.pageX+5})
                    .fadeIn(200);
            } else {
    	    $("#tooltip").hide();
            }
        });
    }

    var extra_groups = [];
    for (var log_group in logs) {
        if (logs.hasOwnProperty(log_group)
            && ($.inArray(log_group, rois) == -1)) {
            extra_groups.push(log_group);
        }
    }
    var roi_div = document.createElement("div");
    roi_div.className = "plot_group_div";
    var hr = document.createElement("h2");
    hr.innerHTML = "Primary Stats";
    roi_div.appendChild(hr);
    body.appendChild(roi_div);
    for (var g = 0; g < rois.length; ++g) {
        var name = rois[g];
        var plot_div;
        name = name.split("/").pop();
        name = name.split(".")[0];
        plot_div = makePlot(name, g);
        roi_div.appendChild(plot_div);
        group = logs[rois[g]];
        plotGroup(group, g);
    }

    var extras_div = document.createElement("div");
    extras_div.className = "plot_group_div";
    var he = document.createElement("h2");
    he.innerHTML = "Secondary Stats";
    extras_div.appendChild(he);
    body.appendChild(extras_div);
    for (var g = 0; g < extra_groups.length; ++g) {
        var name = extra_groups[g];
        name = name.split("/").pop();
        name = name.split(".")[0];
        plot_div = makePlot(name, g + rois.length);
        extras_div.appendChild(plot_div);
        group = logs[extra_groups[g]];
        plotGroup(group, g + rois.length);
    }
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
});