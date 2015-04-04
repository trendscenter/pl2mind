function makeModal(wd) {
    var modal_fade = document.createElement("div");
    modal_fade.id = "confirmModal";
    modal_fade.className = "modal fade";
    modal_fade.setAttribute("tabindex", "-1");
    modal_fade.setAttribute("role", "dialog");
    modal_fade.setAttribute("aria-labelledby", "showModalLabel");
    modal_fade.setAttribute("aria-hidden", "true");

    var modal_dialog = document.createElement("div");
    modal_dialog.className = "modal-dialog";
    modal_dialog.id = "modal_dialog";
    modal_fade.appendChild(modal_dialog);

    var modal_content = document.createElement("div");
    modal_content.className = "modal-content";
    modal_dialog.appendChild(modal_content);
    modal_content.id = "modal_content";

    var modal_header = document.createElement("div");
    modal_header.className = "modal-header";
    modal_content.appendChild(modal_header);
    modal_header.id = "modal_header";

    var button = document.createElement("button");
    button.type = "button";
    button.className = "close";
    button.setAttribute("data-dismiss", "modal");
    button.setAttribute("aria-label", "Close");
    var span = document.createElement("span");
    span.setAttribute("aria-hidden", "true");
    span.innerHTML = "&times;";
    button.appendChild(span);
    modal_header.appendChild(button);

    var title = document.createElement("h4");
    modal_header.appendChild(title);
    title.id = "modal_title";

    var modal_body = document.createElement("div");
    modal_body.className = "modal-body";
    modal_body.id = "modal_body";
    modal_content.appendChild(modal_body);

    return modal_fade;
}

function clearModalBody() {
    var modal_body = document.getElementById("modal_body");
    while (modal_body.firstChild) {
	modal_body.removeChild(modal_body.firstChild);
    }
    return modal_body;
}

function add_feature_to_modal(id, obj, wd, div) {
    var container = document.createElement("div");
    div.appendChild(container);
    container.className = "feature_container";
    var fdiv = document.createElement("div");
    if (obj.image_type == "mri") {
	fdiv.className = "feature_div_big mri";
    } else {
	fdiv.className = "feature_div_big";
    }
    fdiv.id = id + "_feature";
    container.appendChild(fdiv);

    plotImage(wd + "/" + obj.image, fdiv.id);

    for (var hist in obj.hists) {
	var hist_container = document.createElement("div");
	container.appendChild(hist_container);
	hist_container.className = "hist_container";
	var h = document.createElement("h3");
	h.innerHTML = hist;
	hist_container.appendChild(h);
	var hdiv = document.createElement("div");
	hist_container.appendChild(hdiv);
	hdiv.className = "hist_plot";
	hdiv.id = id + "_" + hist;
	bar_plot(hdiv.id, obj.hists[hist]["bins"], obj.hists[hist]["edges"]);
    }

}

function switchFeatureModal(obj, wd) {

    var modal_title = document.getElementById("modal_title");
    modal_title.innerHTML = "Feature " + obj.index.toString();

    var modal_body = clearModalBody();
    add_feature_to_modal("base", obj, wd, modal_body);

    console.log(obj);
    for (var match_model in obj.matched_features) {
	add_feature_to_modal(match_model, obj.matched_features[match_model], wd,
			     modal_body);
    }
}

function switchInfoModal(title, info) {
    var modal_title = document.getElementById("modal_title");
    modal_title.innerHTML = title;

    var modal_body = clearModalBody();
    var pre = document.createElement("pre");
    pre.innerHTML = info;
    modal_body.appendChild(pre);

    $("#modal_dialog").addClass("large");

    if ($("#modal_footer")) {
	$("#modal_footer").remove();
    }
}

function switchDecisionModal(query, title) {

    function makeModalFooter() {
	$("#modal_dialog").removeClass("large");
	var modal_footer = document.createElement("div");
	modal_footer.className = "modal-footer";
	modal_footer.id = "modal_footer";

	var yes = document.createElement("button");
	yes.className = "btn btn-default";
	yes.setAttribute("data-dismiss", "modal");
	yes.innerHTML = "Yes";
	yes.id = "yes_button";
	modal_footer.appendChild(yes);

	var no = document.createElement("button");
	no.className = "btn btn-primary";
	no.innerHTML = "Cancel";
	no.setAttribute("data-dismiss", "modal");
	modal_footer.appendChild(no);
	return modal_footer;
    }

    var modal_title = document.getElementById("modal_title");
    modal_title.innerHTML = title;

    var modal_body = clearModalBody();
    modal_body.innerHTML = query;

    if (! $("#modal_footer")[0]) {
	var modal_footer = makeModalFooter();
	$("#modal_content").append($(modal_footer));
    }
    var yes = document.getElementById("yes_button");

    return yes;
}