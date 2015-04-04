var options = {
    lines: { show: true },
    points: { show: true , radius: 1},
    grid: {
	hoverable: true
    },
    labelMargin: 10,
};

function makePlotDiv(name, id) {
    // Make div for plot.
    var div1 = document.createElement("div")
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

function makePlots(ul, group, id, title, start_idx, active) {
    console.log("Making plots for " + id);
    var tab = makeTab(id, title, active);
    ul.appendChild(tab);
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

    for (var g = 0; g < group.length; ++g) {
	var name = group[g];
	name = name.split("/").pop();
	name = name.split(".")[0];

	var plot_div = makePlotDiv(name, g + start_idx);
	div.appendChild(plot_div);
    }
    return div;
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

function bar_plot(id, bins, edges) {
    data = [];
    ticks = [];
    for (var i = 0; i < bins.length; ++i) {
        data.push([i, bins[i]]);
        if (i % (bins.length / 5) == 0) {
            ticks.push([i, (.5 * (edges[i] + edges[i+1])).toPrecision(2)])
        }
    }

    options = {
        bars: {
            show: true
        },
        xaxis: {
            min: 0,
            max: bins.length,
            ticks: ticks
        }
    }

    $.plot("#" + id, [data], options);
}