var http = require("http");
var url = require("url");
var express = require("express");
var bodyParser = require("body-parser");
var directory = require("serve-index");

var port = process.argv[2]

console.log("Starting server on port " + port);

var server = express();
server.listen(port);
var io = require('socket.io');
var net = require('net');
var zmq = require('zmq');

//server.use("/experiments", directory(__dirname + "/experiments"));
server.use(express.static(__dirname));
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({extended: true}));

console.log("Server started. Root directory is " + __dirname);

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

server.post('/processme', function(req, res) {
    console.log(req.body);
    var port = req.body.id;
    try {
        var requester = zmq.socket('req');
        requester.connect("tcp://mars:" + port);
        requester.send("PROCESS");

        requester.on("message", function(reply) {
            requester.close()
            console.log("Got " + reply);
            res.type("json");
            res.send(JSON.stringify({response: reply}));
        });
    } catch(er) {
        res.send(JSON.stringify({response: er}));
    }
});