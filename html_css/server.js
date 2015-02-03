var http = require("http");
var url = require("url");
var express = require("express");
var bodyParser = require("body-parser");
var directory = require("serve-index");

var port = process.argv[2]
var html_css = process.argv[3];

console.log("Starting server on port " + port);
console.log("Souce at " + html_css);

var server = express();
server.listen(port);
var io = require('socket.io');
var net = require('net');
var zmq = require('zmq');

server.use("html_css", directory(html_css));
server.use(express.static("./"));
server.use(express.static("html_css"));
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({extended: true}));

console.log("Server started. Root directory is " + "./");

server.get("/*/$", function(request, response) {
    response.sendFile("experiment.html", { root: __dirname });
});
/*
server.get(/^\/(.+)/, function(request, response) {
    response.sendFile("/test_plots.html", { root: request.params[0] });
});*/

server.post('/killme', function(req, res) {
    console.log(req.body);
    var port = req.body.id;
    try {
        var requester = zmq.socket('req');
        requester.connect("tcp://mars:" + port);
        requester.send("KILL");

        requester.on("message", function(reply) {
            requester.close()
            console.log("Got " + reply);
            res.type("json");
            res.send(JSON.stringify({response: "OK"}));
        });
    } catch(er) {
        res.send(JSON.stringify({response: er}));
    }
});

var is_processing = {};

server.post('/processme', function(req, res) {
    var port = req.body.id;
    if (!(port in is_processing)) {
        is_processing[port] = false;
    }

    if (!(is_processing[port])) {
        console.log("Got processing request for " + port);
        is_processing[port] = true;
        try {
            var requester = zmq.socket('req');
            requester.connect("tcp://mars:" + port);
            requester.send("PROCESS");

            requester.on("message", function(reply) {
                requester.close()
                console.log("Got " + reply);
                res.type("json");
                res.send(JSON.stringify({response: reply}));
                is_processing[port] = false;
            });
        } catch(er) {
            res.send(JSON.stringify({response: er}));
            is_processing[port] = false;
            console.log("Error processing");
        }
    } else {
        console.log("Got redundant request");
        res.send(JSON.stringify({response: "Already"}));
    }
});