

$(function() {
    var groups = ["done", "running", "waiting", "failed"]
    for (var g = 0; g < groups.length; ++g) {
	writeTable(groups[g], g)
    }
});

/*
$(function() {
    for (var i = 0; i < 4; ++i) {
	var i_string = i.toString();
	console.log(i_string);
	$('#popover' + i_string)
	    .popover({
		placement: 'right',
		html: true,
		content: '<div id="popover-target' + i_string + '"></div>'
	    })
	    .on('shown.bs.popover', function () {
		$.tablesorter.columnSelector.attachTo( $('.bootstrap-popup' + i_string), '#popover-target' + i_string);
	    });

	$(".bootstrap-popup" + i_string).tablesorter({
	    theme: 'blue',
	    widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
	    widgetOptions : {
		columnSelector_columns : {
		    0: 'disable'
		},
		columnSelector_saveColumns: true,
		columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
		columnSelector_mediaquery: false,
		columnSelector_mediaqueryName: 'Auto: ',
		columnSelector_mediaqueryState: true,
		columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
		columnSelector_priority : 'data-priority',
		columnSelector_cssChecked : 'checked'
	    }
	});
    }
});*/

$(function(){
  $('#popover0')
    .popover({
      placement: 'right',
      html: true,
      content: '<div id="popover-target0"></div>'
    })
    .on('shown.bs.popover', function () {
      // call this function to copy the column selection code into the popover
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup0'), '#popover-target0');
    });

  $(".bootstrap-popup0").tablesorter({
      theme: 'blue',
      widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
      widgetOptions : {
      columnSelector_columns : {
        0: 'disable'
      },
      columnSelector_saveColumns: true,
      columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
      columnSelector_mediaquery: false,
      columnSelector_mediaqueryName: 'Auto: ',
      columnSelector_mediaqueryState: true,
      columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
      columnSelector_priority : 'data-priority',
      columnSelector_cssChecked : 'checked'
    }
  });

  $('#popover1')
    .popover({
      placement: 'right',
      html: true,
      content: '<div id="popover-target1"></div>'
    })
    .on('shown.bs.popover', function () {
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup1'), '#popover-target1');
    });

  $(".bootstrap-popup1").tablesorter({
      theme: 'blue',
      widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
      widgetOptions : {
      columnSelector_columns : {
        0: 'disable'
      },
      columnSelector_saveColumns: true,
      columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
      columnSelector_mediaquery: false,
      columnSelector_mediaqueryName: 'Auto: ',
      columnSelector_mediaqueryState: true,
      columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
      columnSelector_priority : 'data-priority',
      columnSelector_cssChecked : 'checked'
    }
  });

  $('#popover2')
    .popover({
	placement: 'right',
	html: true, // required if content has HTML
	content: '<div id="popover-target2"></div>',
    })
    .on('shown.bs.popover', function () {
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup2'), '#popover-target2');
    });

  $(".bootstrap-popup2").tablesorter({
      theme: 'blue',
      widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
      widgetOptions : {
      columnSelector_columns : {
        0: 'disable'
      },
      columnSelector_saveColumns: true,
      columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
      columnSelector_mediaquery: false,
      columnSelector_mediaqueryName: 'Auto: ',
      columnSelector_mediaqueryState: true,
      columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
      columnSelector_priority : 'data-priority',
      columnSelector_cssChecked : 'checked'
    }
  });

    $('#popover3')
    .popover({
      placement: 'right',
      html: true,
      content: '<div id="popover-target3"></div>'
    })
    .on('shown.bs.popover', function () {
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup3'), '#popover-target3');
    });

    $(".bootstrap-popup3").tablesorter({
	theme: 'blue',
	widgets: ['zebra', 'columnSelector', 'stickyHeaders'],
	widgetOptions : {
	    columnSelector_columns : {
		0: 'disable'
	    },
	    columnSelector_saveColumns: true,
	    columnSelector_layout : '<div><label><input type="checkbox">{name}</label></div>',
	    columnSelector_mediaquery: false,
	    columnSelector_mediaqueryName: 'Auto: ',
	    columnSelector_mediaqueryState: true,
	    columnSelector_breakpoints : [ '20em', '30em', '40em', '50em', '60em', '70em' ],
	    columnSelector_priority : 'data-priority',
	    columnSelector_cssChecked : 'checked'
	}
    });

});

$(function() {
    setInterval(function() {
	var dirnames = document.getElementsByClassName("cd-popup-trigger");
	for (var i=0; i < dirnames.length; ++i) {
	    color_view_button($(dirnames[i]).attr("dir"));
	}
    }, 10 * 1000);
});

function color_view_button(dirname) {
    console.log("coloring " + dirname);
    var button = document.getElementById(dirname);
    $.ajax({
	url: dirname,
	type: "GET",
	async: true,
	success: function() {
	    $(button).removeClass("off");
	},
	error: function() {
	    $(button).addClass("off");
	}
    });
}

function checkDir(dirname) {
    var myRegexp = /href="([^"]*[\.pdf])"/g;
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

function parse_lines(source) {
    var data;
    $.ajax({
	url: source,
	type: "GET",
	async: false,
	success: function(d) {
	    data = d;
	},
	error: function(xhr, status, error) {
	    var err = eval("(" + xhr.responseText + ")");
	    alert(err.Message);
	}
    });
    data = data.split(/\r?\n/);

    var lines = [];
    for (var i = 0; i < data.length; ++i) {
	var line = data[i].split(/[\t]+/)
	line = line.filter(function(n){ return n !== ""; });
	if (line.length > 0) {
	    lines.push(line);
	}
    }

    if (lines.length == 0) {
	return {};
    }
    var column_names = lines[1]
    var priorities = [];
    for (var p = 0; p < lines[0].length; ++p) {
	priorities.push(parseInt(lines[0][p]));
    }
    var table_data = {"column_names": column_names,
		      "priorities": priorities,
		      "rows": []};
    for (var r = 2; r < lines.length; ++r) {
	var row = {}
	for (var c = 0; c < column_names.length; ++c) {
	    row[column_names[c]] = lines[r][c];
	}
	table_data.rows.push(row);
    }

    return table_data
}

function writeColumns(dict, head_tr) {
    if (jQuery.isEmptyObject(dict)) {
	return;
    }
    var column_names = dict.column_names;
    for (var c = 0; c < column_names.length; ++c) {
	var all_repeats = true;
	var rows = dict.rows;
	var first = rows[0][column_names[c]];
	for (var i = 1; i < rows.length; ++i) {
	    if (rows[i][column_names[c]] != first) {
		all_repeats = false;
		break;
	    }
	}
	var th = document.createElement("th");
	if (column_names[c] == "host") {
	} else if (all_repeats) {
	    th.className = "columnSelector-false";
	} else if (dict.priorities[c] == 0) {
	    th.className = "columnSelector-false";
	}
	th.innerHTML = column_names[c];
	head_tr.appendChild(th);
    }
    return
}

function writeTable(group, popup_id) {
    var title = group;
    var process_dict = parse_lines(group + "_stats.txt");
    var params_dict = parse_lines(group + "_hyperparams.txt");
    var results_dict = parse_lines(group + "_results.txt");

    var body = document.getElementsByTagName("body")[0];
    var div1 = document.createElement("div");
    div1.className = "table_div";
    var div2 = document.createElement("div");
    div2.className = "table_header_div"
    var div3a = document.createElement("div");
    div3a.className = "table_title_div";
    div3a.innerHTML = title;
    div2.appendChild(div3a);
    var div3b = document.createElement("div");
    div3b.className = "columnSelectorWrapper";
    button = document.createElement("button");
    button.id = "popover" + popup_id.toString();
    button.type = "button";
    button.className = "btn btn-default";
    button.innerHTML = "Select";
    div3b.appendChild(button);
    div2.appendChild(div3b);
    div1.appendChild(div2);

    var table = document.createElement("table");
    table.className = "tablesorter bootstrap-popup" + popup_id.toString();
    var table_head = document.createElement("thead");
    var head_tr = document.createElement("tr");
    var blank_th = document.createElement("th");
    head_tr.appendChild(blank_th);
    writeColumns(process_dict, head_tr);
    writeColumns(params_dict, head_tr);
    writeColumns(results_dict, head_tr);
    table_head.appendChild(head_tr);
    table.appendChild(table_head);

//    if (process_dict.rows.length != params_dict.rows.length) {
//	throw("Table " + title + " data has inconsistent rows (" + process_lines.length.toString() + ", " + parse_lines.length.toString() + ")");
//    }

    if (!jQuery.isEmptyObject(process_dict)) {
	var table_body = document.createElement("tbody");
	for (var r = 0; r < process_dict.rows.length; ++r) {
	    var tr = document.createElement("tr");
	    var button_td = document.createElement("td");
	    var a = document.createElement("a");
	    if (group == "failed") {
		a.className = "cd-popup-trigger log_btn on";
		a.setAttribute("link", "log_files/" + process_dict.rows[r]["id"] + "/stderr");
	    } else {
		a.className = "cd-popup-trigger dir_btn off";
		a.setAttribute("dir", process_dict.rows[r]["file prefix"]);
		a.id = process_dict.rows[r]["file prefix"];
		a.setAttribute("error_script", "Results not found");
	    }
	    a.href = "#";
	    a.setAttribute("data-toggle", "modal");
	    a.setAttribute("data-target", "#showModal");
	    a.innerHTML = "View";
	    button_td.appendChild(a);
	    tr.appendChild(button_td);
	    for (var c = 0; c < process_dict.column_names.length; ++c) {
		var td = document.createElement("td");
		td.innerHTML = process_dict.rows[r][process_dict.column_names[c]];
		tr.appendChild(td);
	    }
	    for (var c = 0; c < params_dict.column_names.length; ++c) {
		var td = document.createElement("td");
		td.innerHTML = params_dict.rows[r][params_dict.column_names[c]];
		tr.appendChild(td);
	    }
	    if (!jQuery.isEmptyObject(results_dict)) {
		for (var c = 0; c < results_dict.column_names.length; ++c) {
		    var td = document.createElement("td");
		    td.innerHTML = results_dict.rows[r][results_dict.column_names[c]];
		    tr.appendChild(td);
		}
	    }

	    table_body.appendChild(tr);
	}
	table.appendChild(table_body);
	div1.appendChild(table);
    }

    var br = document.createElement("br");
    body.appendChild(div1);
    body.appendChild(br);

}

jQuery(document).ready(function($){
    $(".log_btn").on("click", function() {
	var modal = document.getElementById("showModal");
	var modal_dialog = modal.getElementsByClassName("modal-dialog")[0];
	var modal_content = modal_dialog.getElementsByClassName("modal-content")[0];
	var modal_body = modal_content.getElementsByClassName("modal-body")[0];
	$(modal_body).empty();
	var log_txt;
	$.ajax({
	    url: $(this).attr("link"),
	    type: "get",
	    async: false,
	    success: function(txt) {
		log_txt = txt;
	    },
	    error:function() {
		log_txt = "Logs not found";
	    }
	});
	$(modal_body).html(log_txt);
	console.log($(modal).html());
    });

    $(".dir_btn").on("click", function(){
	var dir = $(this).attr("dir");
	var error_script = $(this).attr("error_script");
	var modal = document.getElementById("showModal");
	var modal_dialog = modal.getElementsByClassName("modal-dialog")[0];
	var modal_content = modal_dialog.getElementsByClassName("modal-content")[0];
	var modal_body = modal_content.getElementsByClassName("modal-body")[0];
	var pdf_files = checkDir(dir);
	$(modal_body).empty();
	if (pdf_files.length == 0) {
	    $(".modal-dialog").addClass("large");
	    var div = document.createElement("div");
	    $(div).html("Results do not exist, process with: <br /><div align=\"center\"><h5>"
			+ error_script + "</h5></div>");
	    modal_body.appendChild(div);
	} else if ($(".modal-dialog").hasClass("large")) {
	    $(".modal-dialog").removeClass("large");
	}
	for (i = 0; i < pdf_files.length; i++) {
	    var div = document.createElement("div");
	    var a = document.createElement("a");
	    a.setAttribute("class", "btn btn-primary pdf_btn");
	    a.setAttribute("data-toggle", "modal");
	    a.setAttribute("pdf", pdf_files[i]);
	    a.setAttribute("data-target", "#pdfModal");
	    a.innerHTML = pdf_files[i].split("/").pop();
	    div.appendChild(a);
	    modal_body.appendChild(div);
	}
	$(".pdf_btn").on("click", function(){
	    var pdf_file = $(this).attr("pdf");

	    var modal = document.getElementById("pdfModal");
	    var modal_dialog = modal.getElementsByClassName("modal-dialog")[0];
	    var modal_content = modal_dialog.getElementsByClassName("modal-content")[0];
	    var modal_header = modal_content.getElementsByClassName("modal-header")[0];
	    var modal_title = modal_header.getElementsByClassName("modal-title")[0];
	    modal_title.innerHTML = pdf_file.split("/").pop();
	    var modal_body = modal_content.getElementsByClassName("modal-body")[0];
	    var object_html = "<object data=\"" + pdf_file + "\" type=\"application/pdf\" width=\"100%\" height=\"100%\"></object>";
	    modal_body.innerHTML = object_html;
	});
    });
});