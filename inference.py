import io
from torchvision import transforms
import torch
from PIL import Image
import os
from model.multi_model import ConcatModel
from model.vgg16 import Vgg16
from model.ResNet import ResNet50
from flask import Flask, jsonify, request

app = Flask(__name__)
concat = ConcatModel(2)
vgg = Vgg16(2)
resNet = ResNet50(2)


def transform_image(image_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128))
    ])

    image = Image.open(io.BytesIO(image_data))

    return transform(image).unsqueeze(0)


def get_prediction(model, image_data):
    tensor = transform_image(image_data)
    _, res = model.forward(tensor).max(1)
    return res.numpy()[0]


def load_model(path):
    return torch.load(path, map_location="cpu")


@app.route("/predict_concat", methods=['POST'])
def predict_concat_api_wrapper():
    if request.method == 'POST':
        file = request.files['file']
        img_byte = file.read()
        concat.eval()
        prediction = get_prediction(concat, img_byte)
        return jsonify({"res": int(prediction)})


@app.route("/predict_vgg", methods=['POST'])
def predict_vgg_api_wrapper():
    if request.method == 'POST':
        file = request.files['file']
        img_byte = file.read()
        vgg.eval()
        prediction = get_prediction(vgg, img_byte)
        return jsonify({"res": int(prediction)})


@app.route("/predict_res", methods=['POST'])
def predict_res_api_wrapper():
    if request.method == 'POST':
        file = request.files['file']
        img_byte = file.read()
        resNet.eval()
        prediction = get_prediction(resNet, img_byte)
        return jsonify({"res": int(prediction)})


if __name__ == '__main__':
    # print(os.path.exists("saved_model/2020-12-27-10-01-39/Concat"))
    # saved_model_dict = load_model("saved_model/2020-12-31-13-55-41/Concat/ep200-loss0.3701")
    # with open("data/faces/30601258@N03/coarse_tilt_aligned_face.2.8623020082_578acef81b_o.jpg", 'rb') as f:
    #     image_byte = f.read()
    #     saved_model = ConcatModel(2)
    #     saved_model.load_state_dict(saved_model_dict)
    #     saved_model.eval()
    #     print(get_prediction(saved_model, image_byte))
    concat_dict = load_model("saved_model/2020-12-31-13-55-41/Concat/ep200-loss0.3701")
    vgg_dict = load_model("saved_model/2020-12-31-13-55-41/VGG/ep300-loss0.3906")
    res_dict = load_model("saved_model/2020-12-31-13-55-41/ResNet/ep300-loss0.4413")
    concat.load_state_dict(concat_dict)
    vgg.load_state_dict(vgg_dict)
    resNet.load_state_dict(res_dict)
    concat.eval()
    vgg.eval()
    resNet.eval()
    app.run()

