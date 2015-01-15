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