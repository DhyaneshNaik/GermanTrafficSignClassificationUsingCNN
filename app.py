from flask import *
from werkzeug.utils import secure_filename
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# classes of the Traffic Signs
classes = {
    0: "Speed Limit (20km/h)",
    1: "Speed Limit (30km/h)",
    2: "Speed Limit (50km/h)",
    3: "Speed Limit (60km/h)",
    4: "Speed Limit (70km/h)",
    5: "Speed Limit (80km/h)",
    6: "End of Speed Limit (80km/h)",
    7: "Speed Limit (100km/h)",
    8: "Speed Limit (120km/h)",
    9: "No passing",
    10: "No passing veh over 3.5 tons",
    11: "Right-of-wat at interaction",
    12: "Priority Road",
    13: "Yeild",
    14: "Stop",
    15: "No Vehicles",
    16: "Vehicle > 3.5 ton is prohibited",
    17: "No Entry",
    18: "General Caution",
    19: "Dangerous Curve left",
    20: "Dangerous Curve Right",
    21: "Double Curve",
    22: "Bumpy Road",
    23: "Slippery Road",
    24: "Road Narrows on the right",
    25: "Road Work",
    26: "Traffic Signals",
    27: "Pedestrians",
    28: "Childrens Crossing",
    29: "Bicycles Crossing",
    30: "Beware of Ice/Snow",
    31: "Wild Animals Crossing",
    32: "End Speed + Passing limits",
    33: "Turn Right Ahead",
    34: "Turn Left Ahead",
    35: "Ahead Only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep Right",
    39: "Keep Left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing veh > 3.5 ton",
}


def image_processing(img):
    model = load_model('./model/TSP.h5')

    imgs = image.load_img(img, target_size=(30, 30))
    X = np.array(imgs)
    X = np.expand_dims(X, axis=0)
    pred = model.predict_classes(X)
    return pred


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict", methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['imageUpload']
        file_name = secure_filename(f.filename)
        file_path = "./static/images/" + file_name
        f.save(file_path)

        result = image_processing(file_path)
        output = "Predicted Traffic Sign is : " + classes[int(result)]
        #os.remove(file_path)
        data = {'image': file_path, 'output': output}
        return render_template('predict.html',
                               prediction=output,
                               path=file_name)
    return None


if __name__ == "__main__":
    app.run(debug=True)