from ultralytics import YOLO
from flask import request, Flask, jsonify
from waitress import serve
from PIL import Image

app = Flask(__name__)

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/classify", methods=["POST"])
def classify():
    """
        Handler of /classify POST endpoint
        Receives uploaded file with a name "image_file",
        passes it through YOLOv8 classification network and returns the class.
        :return: a JSON object with class name and probability
    """
    buf = request.files["image_file"]
    class_info = classify_image(Image.open(buf.stream))
    return jsonify(class_info)


def classify_image(buf):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns the predicted class and its probability.
    :param buf: Input image file stream
    :return: Dictionary with class name and probability
    """
    model = YOLO("C:/Users/Admin/Desktop/cLASSIFICATION/runs/classify/train3/weights/best.pt")
    results = model.predict(buf)
    result = results[0]
    output = {
        "class": result.names[result.probs.top1],
        "probability": round(result.probs.top1conf.item(), 2)
    }
    return output


serve(app, host='0.0.0.0', port=8080)
