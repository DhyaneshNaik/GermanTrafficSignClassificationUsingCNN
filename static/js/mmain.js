$(document).ready(function () {
    $('#image-section').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#image').css('background-image', 'url(' + e.target.result + ')');
                $('#image').css('width', '256px');
                $('#image').css('height', '256px');
                $('#image').css('position', 'position');
                $('#image').css('box-shadow', '0px 2px 4px 0px rgba(0, 0, 0, 0.1)');
                $('#image').css('margin-top', '1em');
                $('#image').css('margin-bottom', '1em');
                $('#image').css('border-style', 'solid');
                $('#image').hide();
                $('#image').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('#image-section').show();
        $('.btnUpload').show();
        readURL(this);

    });


});