import os
import model
import flask
from werkzeug.utils import secure_filename

# Creating a new Flask Web application. It accepts the package name.
app = flask.Flask(__name__)


def CNN_predict():
    """
    Upload the image file and classify the severity of DR using pre-trained CNN model
    """

    """
    Predicted rating of image for the severity of diabetic retinopathy on a scale of 0 to 4:

    0 - No DR
    1 - Mild
    2 - Moderate
    3 - Severe
    4 - Proliferative DR
    """
    diagnosis = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

    global secure_filename

    # Getting the image path
    img_path = os.path.join(app.root_path, secure_filename)

    """
    Check if image exists
    """
    if os.path.isfile(img_path):
        # predicting class from pretrained model
        predicted_class = model.classify_image(img_path)

        """
        After predicting the class label of the input image, 
        the prediction label and predicted diagnosis are rendered on an HTML page.
        The HTML page is fetched from the /templates directory. 
        The HTML page accepts two inputs which are the predicted class and predicted diagnosis.
        """
        return flask.render_template(
            template_name_or_list="prediction_result.html",
            predicted_class=predicted_class,
            predicted_diagnosis=diagnosis[predicted_class],
        )
    else:
        # The error page is shown if file is not found
        return flask.render_template(template_name_or_list="error.html")


app.add_url_rule(rule="/predict/", endpoint="predict", view_func=CNN_predict)


def upload_image():
    """
    Viewer function that is called in response to getting to the 'http://localhost:port/upload' URL.
    It uploads the selected image to the server.
    :return: redirects the application to a new page for predicting the class of the image.
    """
    # Global variable to hold the name of the image file for reuse later in prediction by the 'CNN_predict' viewer functions.
    global secure_filename
    if (
        flask.request.method == "POST"
    ):  # Checking of the HTTP method initiating the request is POST.
        img_file = flask.request.files[
            "image_file"
        ]  # Getting the file name to get uploaded.
        secure_filename = secure_filename(
            img_file.filename
        )  # Getting a secure file name. It is a good practice to use it.
        img_path = os.path.join(
            app.root_path, secure_filename
        )  # Preparing the full path under which the image will get saved.
        img_file.save(img_path)  # Saving the image in the specified path.
        print("Image uploaded successfully.")
        """
        After uploading the image file successfully, next is to predict the class label of it.
        The application will fetch the URL that is tied to the HTML page responsible for prediction and redirects the browser to it.
        The URL is fetched using the endpoint 'predict'.
        """
        return flask.redirect(flask.url_for(endpoint="predict"))
    return "Image upload failed."


app.add_url_rule(
    rule="/upload/", endpoint="upload", view_func=upload_image, methods=["POST"]
)


def redirect_upload():
    return flask.render_template(template_name_or_list="upload_image.html")


app.add_url_rule(rule="/", endpoint="homepage", view_func=redirect_upload)

"""
To activate the Web server to receive requests, the application must run from the main method.
"""
if __name__ == "__main__":
    """
    In this example, the app will run based on the following properties:
    host: localhost
    port: 5000 
    debug: flag set to True to return debugging information.
    """
    app.run(host="localhost", port=5000, debug=True)
