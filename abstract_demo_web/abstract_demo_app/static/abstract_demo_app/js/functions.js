function getArticle(selector) {
    artID = $(selector).val();
    if (!isNaN(artID) && !artID.includes(",") && !artID.includes(".") && artID.length == 8) {
        $(selector).css("border", "1px solid #888888");
        $(selector).css("color", "#000000");
        $.ajax({
            url : "/ajax/getArticle",
            type: "POST",
            data: $("#article_form").serialize(),
            dataType: "html",
            success: function (data) {
                $("#abstract_container").html(unescapeHtml(data));
            },
            error: function (jXHR, textStatus, errorThrown) {
                alert(errorThrown);
            }
        });

    } else {
        $(selector).css("border", "1px solid red");
        $(selector).css("color", "red");
        alert("Not a valid number.");
    }
}

function analizeArticle() {
    $.ajax({
        url : "/ajax/getArticlePredictions",
        type: "POST",
        data: $("#candidates_form").serialize(),
        dataType: "html",
        success: function (data) {
            $("#abstract_container").html(unescapeHtml(data));
        },
        error: function (jXHR, textStatus, errorThrown) {
            alert(errorThrown);
        }
    });
}

function unescapeHtml(safe) {
    return safe.replace(/&amp;/g, '&')
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g, "'");
}