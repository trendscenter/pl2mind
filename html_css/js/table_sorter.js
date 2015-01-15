$(function(){
  $('#popover0')
    .popover({
      placement: 'right',
      html: true, // required if content has HTML
      content: '<div id="popover-target0"></div>'
    })
    // bootstrap popover event triggered when the popover opens
    .on('shown.bs.popover', function () {
      // call this function to copy the column selection code into the popover
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup0'), '#popover-target0');
    });

  // initialize column selector using default settings
  // note: no container is defined!
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
      html: true, // required if content has HTML
      content: '<div id="popover-target1"></div>'
    })
    // bootstrap popover event triggered when the popover opens
    .on('shown.bs.popover', function () {
      // call this function to copy the column selection code into the popover
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup1'), '#popover-target1');
    });

  // initialize column selector using default settings
  // note: no container is defined!
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
	//placement: 'right',
	html: true, // required if content has HTML
	content: '<div id="popover-target2"></div>',
    })
    // bootstrap popover event triggered when the popover opens
    .on('shown.bs.popover', function () {
      // call this function to copy the column selection code into the popover
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup2'), '#popover-target2');
    });

  // initialize column selector using default settings
  // note: no container is defined!
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
      html: true, // required if content has HTML
      content: '<div id="popover-target3"></div>'
    })
    // bootstrap popover event triggered when the popover opens
    .on('shown.bs.popover', function () {
      // call this function to copy the column selection code into the popover
      $.tablesorter.columnSelector.attachTo( $('.bootstrap-popup3'), '#popover-target3');
    });

  // initialize column selector using default settings
  // note: no container is defined!
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
