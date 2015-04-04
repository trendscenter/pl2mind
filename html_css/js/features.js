function makeModelTab(id, title, active) {
    var ul = document.getElementById("analTab");
    var tab_content = document.getElementById("anal_tab_content");
    console.log("Plotting features for " + id);
    if (!document.getElementById(id + "_tab")) {
	var tab = makeTab(id, title, active);
	ul.appendChild(tab);
	var model_div = document.createElement("div");
	tab_content.appendChild(model_div);
	if (active) {
	    model_div.className = "tab-pane fade in active model";
	} else {
	    model_div.className = "tab-pane fade model";
	}
	model_div.role = "tabpanel";
	model_div.id = id;
	tab_content.appendChild(model_div);
    } else {
	var model_div = document.getElementById(id);
    }

    while (model_div.firstChild) {
	model_div.removeChild(model_div.firstChild);
    }

    return model_div;
}

function updateModels(json, wd) {
    console.log("Updating analysis");

    var anal_text = document.getElementById("anal_text");
    anal_text.innerHTML = "";

    var first = true;
    for (var model in json) {
	var model_div = makeModelTab(model, model, first);
	first = false;

        var top_div = d3.select(model_div).append("div")
        .style("border-bottom-style", "solid")
        .style("border-bottom-width", 10);

        makeGraph(model + " graph", json[model].graphs, json[model].features,
                  model + "_graph", top_div, wd);

        var plot_div = top_div.append("div")
        .attr("class", "plot_content_div")
        .append("h4")
        .html(model.toUpperCase() + " Stats");

        var plot = plot_div.append("div")
        .attr("id", "plot" + model + "_" + "stats")
        .attr("class", "plot_div");

	plotGroup(json[model].stats, model + "_" + "stats");

        var feature_array = $.map(json[model].features, function(value, index) {
            return [value];
        });

        var lower_div = d3.select(model_div).append("div");
        var fdiv = lower_div.selectAll("div")
        .data(feature_array, function(d, i) {
            return model + "_feature_" + i.toString();
        })
        .enter().append("div")
        .attr("id", function (d, i) {
            return model + "_feature_" + i.toString();
        })
        .attr("class", function (d, i) {
            if (d.image_type == "mri") {
                return "feature_div mri";
            } else {
                return "feature_div";
            }
	})
	.attr("id", function (d, i) {
            return model + "_feature_" + i.toString();
        })
	.attr("href", "#")
	.attr("data-toggle", "modal")
	.attr("data-target", "#confirmModal")
        .each(function (d, i) {
            plotImage(wd + "/" + d.image, model + "_feature_" + i.toString())
        })
        .on("click", function(d) {
	    switchFeatureModal(d, wd);
	});

        lower_div.insert("div", ":first-child")
        .append("h4")
        .html(model.toUpperCase() + " features")
        .style("border", "1px solid #a1a1a1")
        .style("border-radius", 5)
        .style("background-color", "#eeeeee");

        /*
	var match_models = [];
	for (var feature in json[model].features) {
	    var f = json[model].features[feature];
	    addFeature(feature, f, model, feature, model_div);
	    var matched_features = {};
	    for (var match_model
		 in json[model].features[feature].match_indices) {
		if ($.inArray(match_model, match_models) == -1) {
		    match_models.push(match_model);
		}
		var i = json[model].features[feature].match_indices[match_model];
		matched_features[match_model] = json[match_model].features[i];
	    }
	    json[model].features[feature].matched_features = matched_features;
	}

	for (var match_model in match_models) {
	    console.log(match_models[match_model]);
	    var match_div = makeFeaturesTab(model + "_" + match_models[match_model],
					    (model + "+" + match_models[match_model]).toUpperCase(), false);

	    for (var feature in json[model].features) {
		if (match_models[match_model]
		    in json[model].features[feature].match_indices) {
		    var match_fdiv = document.createElement("div");
		    match_fdiv.className = "match_feature_div";
		    match_div.appendChild(match_fdiv);
		    var f = json[model].features[feature];
		    addFeature("m1" + feature, f, model, feature, match_fdiv);
		    var matched_feature = json[model].features[feature].matched_features[match_models[match_model]];
		    addFeature("m2" + feature, matched_feature, feature, match_model, match_fdiv);
		}
	    }
	}
	*/
    }
    /*
    for (var model in json) {
        var relations = json[model].relations;
        for (var relation in relations) {
            var model_div = makeModelTab(model + "_" + relation, model + "+" + relation, false);
            var top_div = d3.select(model_div).append("div")
            .style("border-bottom-style", "solid")
            .style("border-bottom-width", 10);

            combineGraphs(model + "+" + relation + " graph",
                          json[model].graphs, json[model].features,
                          json[relation].graphs, json[relation].features,
                          model + "+" + relation + "_graph", relations[relation], top_div, wd);
        }
    }
    */
}