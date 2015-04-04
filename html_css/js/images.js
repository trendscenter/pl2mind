function plotImage(image, id) {
    var data = [[[image, 0, 0, 10, 10]]];
    var options = {
	series: {
    	images: {
		margin: 0,
	        show: true
    	}
	},
        xaxis: {
	    show: false
	},
        yaxis: {
	    show: false
	},
	grid: {
	    borderWidth: 0,
	    minBorderMargin: 2
	}
    };

    $.plot.image.loadDataImages(data, options, function () {
	$.plot("#" + id, data, options);
    });
}