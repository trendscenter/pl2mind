function copyGraph(old_graph, features) {
    var nodes = [];
    var links = {};
    var new_graph = {
        nodes: nodes,
        links: links};

    for (var n in old_graph.nodes) {
        var old_node = old_graph.nodes[n];
        var node = {name: old_node.name,
                    feature: features[n]}
        new_graph.nodes.push(node);
    }

    for (var g in old_graph.links) {
        var new_links = [];
        for (var l in old_graph.links[g]) {
            var old_link = old_graph.links[g][l];
            new_links.push({
                source: old_link.source,
                target: old_link.target,
                value: old_link.value
            });
        }
        new_graph.links[g] = new_links;
    }
    return new_graph;
}

function Links(id, svg, links_list, slider_value, stroke_scale, hide_axis) {
    var self = this;
    self.force;

    for (stat in links_list) break;
    self.stat = stat;

    function check(d) {
        return (Math.abs(d.value) > slider_value && d.source.on && d.target.on);
    }

    var min = 1000;
    var max = -1000;

    function findMaxMin() {
        min = 1000;
        max = -1000;
        for (var l in links_list[stat]) {
            if (Math.abs(links_list[stat][l].value) < min) {
                min = links_list[stat][l].value;
            }
            if (Math.abs(links_list[stat][l].value) > max) {
                max = links_list[stat][l].value;
            }
        }
    }

    self.init = function () {
        console.log(links_list);
        for (var s in links_list) {
            var ls = links_list[s];
            for (var l in ls) {
                var strength = Math.pow((Math.abs(ls[l].value) - min) / max, 2);
                ls[l].strength = strength;
            }
            self.force.f
            .links(ls)
            .start();
        }
    }

    self.get = function () {
        return links_list[self.stat].filter(check);
    }
    self.update = function () {
        findMaxMin();

        var link = svg.selectAll("g.line_" + id)
        .data(links_list[self.stat].filter(check),
            function (d) {
                return d.source.name + "-" + d.target.name;
            });

        var linkEnter = link.enter().append("g")
        .attr("class", "line_" + id)
        .append("line")
        .attr("class", "line")
        .attr("id", function (d) {
            return d.source.name + "-" + d.target.name;
        })
        .style({
            stroke: function (d) {
                if (d.value >= 0) {
                    return "#606060";
                } else {
                    return "#E06060";
                }
            },
            "stroke-width": function (d) {
                return stroke_scale * 3 * (Math.abs(d.value) - min) / max + 0.1;
            }
        });

        link.exit().remove();
        return link;
    }

    findMaxMin();

    self.slider = d3.slider()
    .axis(!hide_axis)
    .min(min)
    .max(max)
    .value(0.5)
    .step(0.01)
    .orientation("vertical")
    .on("slide", function(e, v) {
        slider_value = self.slider.value();
        self.force.update();
    });

    self.makeButtons = function(div) {
        var button_div = div.insert("div", ":first-child");
        var first = true;
        for (var l in links_list) {
            var button = button_div.append("button")
            .attr("class", function(d) {
                if (first) {
                    return "btn btn-default btn-lg graph_button active";
                } else {
                return "btn btn-default btn-lg graph_button";
                }
            })
            .attr("aria-label", "Left Align")
            .attr("l", l)
            .on("click", function() {
                $(".graph_button").removeClass("active");
                $(this).addClass("active");
                self.stat = this.getAttribute("l");
                self.force.update();
            })
            .append("span")
            .attr("aria-hidden", "true")
            .html(l);
            first = false;
        }
    }
}

function Nodes(id, svg, nodes, foci, color, popover, slider_value, scale_factor,
               vary_scale, group, hide_axis) {
    var self = this;
    self.check_value = "max_weight_prop";
    self.force;
    var min = 1000;
    var max = -1000;

    function getMinMax() {
        for (var n in nodes) {
            if (nodes[n].feature[self.check_value] < min) {
                min = nodes[n].feature[self.check_value];
            }
            if (nodes[n].feature[self.check_value] > max) {
                max = nodes[n].feature[self.check_value];
            }
        }
    }

    function check(d) {
        return (d.feature[self.check_value] > slider_value);
    }

    function get_value(d) {
        if (vary_scale) {
            return Math.sqrt((d.feature[self.check_value] - min) * (0.7 / max) + 0.3);
        } else {
            return 1;
        }
    }

    self.get_init = function() {
        var i = 0;
        for (var n in nodes) {
            nodes[n].foci = foci;
            nodes[n].id = i;
            nodes[n].on = true;
            if (id != "") {
                nodes[n].name = id + "_" + nodes[n].name;
            }
            if (group) {
                nodes[n].group = group;
            }
            ++i;
        }
        return nodes;
    }

    self.get = function() {
        return nodes.filter(check);
    }

    self.nodes = nodes;

    self.update = function() {
        getMinMax();
        var node = svg.selectAll("g.node_" + id)
        .data(nodes.filter(check)
        , function (d) {
                return d.name;
        });

        var nodeEnter = node.enter().append("g")
        .attr("class", "node node_" + id)
        .each(function (d) {
            nodes[d.id].on = true;
        })
        .call(self.force.f.drag);

        var circle = nodeEnter.append("svg:circle")
        .attr("r", function (d) {
            return scale_factor * get_value(d);
        })
        .attr("name", function (d) {
            return d.name;
        })
        .attr("class", "nodeStrokeClass")
        .style({
            fill: function(d) {
                if (d.hasOwnProperty("group")) {
                    return color(d.group);
                } else {
                    return color(d.name);
                }
            },
            stroke: "#fff",
            "stroke-width": 2
        });

        nodeEnter.append("svg:text")
        .attr("class", "textClass")
        .attr("x", 14)
        .attr("y", ".31em")
        .text(function (d) {
            return d.name.substring(id.length + 1, d.name.length);
        });

        node.exit()
        .each(function (d) {
            nodes[d.id].on = false;
        })
        .remove();

        circle
        .tooltip(function(d, i) {
            return {
                type: "popover",
                title: d.name,
                image: d.feature.image,
                content: popover,
                detection: "shape",
                placement: "mouse",
                gravity: "right",
                position: [d.x, d.y],
                displacement: [scale_factor * get_value(d) + 2, 0],
                mousemove: false
            };
        })
        /*
        .on("click", function(d) {
	    switchFeatureModal(features[d.name], wd);
	});;
	*/
        return node;
    }
    getMinMax();
    self.slider = d3.slider()
    .axis(!hide_axis)
    .min(min)
    .max(max)
    .step(0.01)
    .on("slide", function(e, v) {
        slider_value = self.slider.value();
        // For some reason...
        self.force.update();
        self.force.update();
    });

}

function Force(svg, width, height) {
    var self = this;
    self.f = d3.layout.force();

    nodes_list = [];
    links_list = [];

    this.loadGraph = function(nodes, links) {
        if ($.inArray(nodes, nodes_list) == -1) {
            nodes_list.push(nodes);
            nodes.force = self;

            self.f
            .nodes(nodes.get_init())
            .charge(-80)
            .linkDistance(60)
            .size([width, height]);
        }
        if ($.inArray(links, links_list) == -1) {
            links_list.push(links);
            links.force = self;
            links.init();
        }
        self.f.start();
    }

    this.connectNodes = function(id, nodes_1, nodes_2, links, slider_value, stroke_scale,
                                 hide_axis) {
         if ($.inArray(nodes_1, nodes_list) == -1) {
            console.log("Didn't find ", nodes_1);
            return
         }
         if ($.inArray(nodes_2, nodes_list) == -1) {
            console.log("Didn't find ", nodes_2);
            return
         }
         ls = [];
         for (var s in links) {
            for (var t in links[s]) {
                ls.push({
                    source: nodes_1.nodes[s],
                    target: nodes_2.nodes[t],
                    value: links[s][t]
                });
            }
         }
         var new_links = new Links(id, svg, {"combine": ls}, slider_value,
                                   stroke_scale, hide_axis);
         new_links.force = self;
         new_links.init();
         links_list.push(new_links);
         return new_links;
    }

    this.update = function() {

        var f_nodes = [];
        var f_links = [];

        for (var n in nodes_list) {
            nodes_list[n].update();
            f_nodes = f_nodes.concat(nodes_list[n].get());
        }
        for (var l in links_list) {
            links_list[l].update();
            f_links = f_links.concat(links_list[l].get());
        }

        var node = svg.selectAll(".node");
        var link = svg.selectAll(".line");

        self.f.on("tick", function (e) {
            var alpha = e.alpha;

            f_nodes.forEach(function(o, i) {
                o.y += (o.foci.y - o.y) * o.foci.k * alpha;
                o.x += (o.foci.x - o.x) * o.foci.k * alpha;
                });

            node.attr("transform", function (d) {
                return "translate(" + d.x + "," + d.y + ")";
            });

            link.attr("x1", function (d) {
                return d.source.x;
            })
            .attr("y1", function (d) {
                return d.source.y;
            })
            .attr("x2", function (d) {
                return d.target.x;
            })
            .attr("y2", function (d) {
                return d.target.y;
            });
        });

        console.log(f_nodes.length);
        console.log(f_links.length);
        console.log(link);
        self.f
        .nodes(f_nodes)
        .links(f_links)
        .charge(-80)
        .linkStrength(function (link) {
            return link.strength;})
        .linkDistance(60)
        .size([width, height])
        .start();

        keepNodesOnTop();
    }

    function keepNodesOnTop() {
        $(".nodeStrokeClass").each(function(index) {
            var gnode = this.parentNode;
            gnode.parentNode.appendChild(gnode);
        });
    }
}

function makeGraph(title, graph_list, features, id, pdiv, wd) {
    var width = 600,
        height = 600;

    var div = pdiv.append("div")
    .attr("class", "graph-content")
    .style("display", "inline-block");

    var mid_div = div.append("div")
    .attr("class", "row-fluid");

    var gdiv = mid_div.append("div")
    .attr("id", id)
    .attr("class", "graph_div")
    .style("display", "inline-block");

    var color = d3.scale.category20();

    var svg = gdiv.append("svg:svg")
    .attr("width", width + 20)
    .attr("height", height + 20)
    .attr("fill", "white")
    .attr("id", "svg")
    .append("svg:g");

    var g = copyGraph(graph_list, features);

    var popover = d3.select(
        document.createElement("svg")).attr("height", 0);
    popover.append("g");

    var foci = {k: 0, x: width / 2, y: height / 2}
    var nodes = new Nodes(id, svg, g.nodes, foci, color, popover, 0, 12, 1);
    var links = new Links(id, svg, g.links, 0.5, 1);

    var force = new Force(svg, width, height);
    force.loadGraph(nodes, links);

    links.makeButtons(div);

    mid_div.append("div")
    .attr("id","vslider")
    .style("display", "inline-block")
    .style({"height": 600})
    .style("margin-top", -80)
    .style("bottom", -80)
    .call(links.slider);

    mid_div.append("div")
    .attr("id","hslider")
    .style("width", 600)
    .call(nodes.slider);

    div.insert("div", ":first-child")
    .append("h4")
    .html(title);

    force.update();
}

function combineGraphs(title, graph_list_1, features_1, graph_list_2,
                       features_2, id, relations, pdiv, wd) {

    var width = 600,
    height = 600;

    var div = pdiv.append("div")
    .attr("class", "graph-content")
    .style({display: "inline-block",
           width: 720});

    var mid_div = div.append("div")
    .attr("class", "row-fluid");

    var gdiv = mid_div.append("div")
    .attr("id", id)
    .attr("class", "graph_div")
    .style("display", "inline-block");

    var zoom = d3.behavior.zoom()
            .scaleExtent([1, 10])
            .on("zoom", zoomed);

    var svg = gdiv.append("svg:svg")
    .attr("width", width + 20)
    .attr("height", height + 20)
    .attr("fill", "white")
    .attr("id", "svg")
    .append("svg:g")
    .call(zoom);

    function zoomed() {
        svg.attr("transform", "translate(" + d3.event.translate + ")scale(" + d3.event.scale + ")");
    }

    var g1 = copyGraph(graph_list_1, features_1);
    var g2 = copyGraph(graph_list_2, features_2);

    var popover = d3.select(
        document.createElement("svg")).attr("height", 0);
    popover.append("g");

    var color = d3.scale.category20();

    var foci_1 = {k: 0.1, x: 1 * width / 3, y: 1 * height / 3}
    var nodes_1 = new Nodes("foo", svg, g1.nodes, foci_1, color, popover, 0, 8, true, 1, true);
    var links_1 = new Links("foo", svg, g1.links, 0.7, .5, true);

    var foci_2 = {k: 0.1, x: 2 * width / 3, y: 2 * height / 3}
    var nodes_2 = new Nodes("bar", svg, g2.nodes, foci_2, color, popover, 0, 8, true, 2);
    var links_2 = new Links("bar", svg, g2.links, 0.7, .5, true);

    var force = new Force(svg, width, height);
    force.loadGraph(nodes_1, links_1);
    force.loadGraph(nodes_2, links_2);
    var rel_links = force.connectNodes("relations", nodes_1, nodes_2, relations, 0.5, 0.5)

    //links_1.makeButtons(div);

    mid_div.append("div")
    .attr("id","vslider3")
    .style("display", "inline-block")
    .style({"height": 600})
    .call(links_1.slider);

    mid_div.append("div")
    .attr("id","vslider4")
    .style("display", "inline-block")
    .style({"height": 600})
    .call(links_2.slider);

    mid_div.append("div")
    .attr("id","vslider5")
    .style("display", "inline-block")
    .style({"height": 600})
    .style("margin-top", -99)
    .style("bottom", -99)
    .call(rel_links.slider);

    mid_div.append("div")
    .attr("id","hslider3")
    .style("width", 600)
    .call(nodes_1.slider);

    mid_div.append("div")
    .attr("id","hslider4")
    .style("width", 600)
    .call(nodes_2.slider);

    div.insert("div", ":first-child")
    .append("h4")
    .html(title);

    force.update();

}