var port = process.argv[2]
var pl2mind_path = process.argv[3];
var lib_path = process.argv[4];

var path = require("path");

console.log("Starting server on port " + port);
console.log("Libraries at " + lib_path);
console.log("pl2mind path at " + pl2mind_path)

var sources = require(lib_path);
var express = sources.express
var bodyParser = sources.bodyParser

var server = express();
sources.redirect(server);
server.listen(port);
var router = express.Router();

server.use(express.static(path.resolve("./")));
server.use(bodyParser.json());
server.use(bodyParser.urlencoded({extended: true}));
server.use("/lib", express.static(lib_path));
server.use("/pl2mind", express.static(pl2mind_path));

console.log("Server started. Root directory is " + path.resolve("./"));

var table_data;

server.get("/table/:table_name$", function (req, res) {
    var table_name = req.params.table_name;
    console.log("Getting table " + table_name);
    var Test = Bookshelf.Collection.extend({
        tableName: table_name,
        toJSON: function () {
            attrs = Bookshelf.Collection.prototype.toJSON.apply(this, arguments);
            for (var a = 0; a < attrs.length; ++a) {
                for (attr in attrs[a]) {

                    if (attrs[a][attr] instanceof Buffer) {
                        var buf = Buffer(attrs[a][attr]);
                        var s = "";
                        for (var i = 0; i < buf.toJSON().length; ++i) {
                            var c = String.fromCharCode(buf.toJSON()[i]);
                            s += c;
                        }
                        attrs[a][attr] = s;
                    }

                    var new_attr = attr.replace(/_/g, ".");
                    new_attr = new_attr.replace(/&/g, "_");

                    if (new_attr != attr) {
                        attrs[a][new_attr] = attrs[a][attr];
                        delete attrs[a][attr];
                    }
                }
            }
            return attrs;
        }
    });
    Test.forge()
    .fetch()
    .then(function (collection) {
      if (!collection) {
        res.status(404).json({error: true, data: {}});
      }
      else {
        console.log("Sending table");
        table_data = collection.toJSON();
        res.sendFile("table.html", { root: __dirname });
      }
    })
    .otherwise(function (err) {
      res.status(500).json({error: true, data: {message: err.message}});
    });
  });

server.get("/get_table", function (req, res) {
    console.log("Getting tables");
    res.json(table_data);
});

server.get("/*/$", function(request, response) {
    console.log("Sending experiment");
    response.sendFile("experiment.html", { root: __dirname });
});

server.get(path.resolve("./") + "(/*/model.json)$", function(request, response) {
    console.log("Model json request: " + path.resolve("./") + request.params[0]);
    fs.exists(path.resolve("./") + request.params[0], function(exists) {
        if (exists) {
            response.sendFile(path.resolve("./") + request.params[0]);
        } else {
            response.status(404).send("Not found");
        }
    });
});

server.get(path.resolve("./") + "(/*)$", function(request, response) {
    console.log(path.resolve("./") + request.params[0] + "/model.json");
    fs.exists(path.resolve("./") + request.params[0] + "/model.json", function(exists) {
        if (exists) {
            response.send(request.params[0]);
        } else {
            response.status(404).send("Not found");
        }
    });
});

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