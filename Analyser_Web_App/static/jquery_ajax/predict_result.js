$("#upload").click(function()
{
    window.location.href='/upload_data_sets';
});

//$("#predict").click(function(e)
//{
//    e.preventDefault();
//    $.ajax(
//    {
//        type:'GET',
//        url:'http://127.0.0.1:5000/Prediction',
//        contentType: "application/json; charset=utf-8",
//        dataType: "json",
//
//        success: function(response)
//        {
//            alert(response);
//        },
//        error: function(XMLHttpRequest, textStatus, errorThrown)
//        {
//            alert('Please try again');
//        }
//    })
//});

$("#predict").click(function(e)
{
    e.preventDefault();
    $.ajax(
    {
        type:'GET',
        url:'http://127.0.0.1:5000/Prediction',
        contentType: "application/json; charset=utf-8",
        dataType: "json",

        success: function(response)
        {
            alert(response);
        },
        error: function(XMLHttpRequest, textStatus, errorThrown)
        {
            alert('Please try again');
        }
    })
});

$("#train").click(function(e)
{
    e.preventDefault();
    $.ajax(
    {
        type:'GET',
        url:'http://127.0.0.1:5000/Training_Accuracy',
        contentType: "application/json; charset=utf-8",
        dataType: "html",

        success: function(response)
        {
            alert(response);
        },
        error: function(XMLHttpRequest, textStatus, errorThrown)
        {
            alert('Please try again');
        }
    })
});

