from flask import Flask, request, jsonify
from models import Summarizer, Recognizer
from utils import summarize_image, recognize_image
import numpy as np
import cv2


app = Flask(__name__)

sum_model = Summarizer()
ocr_model = Recognizer()


@app.route('/')
def index() -> str:
    return 'Hello, World!'

@app.route('/summarize', methods=['POST'])
def summarize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    # Read the image and convert it to OpenCV format
    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Summarize the text from the image
    min_length = request.form.get('min_length', 100)
    max_length = request.form.get('max_length', 150)
    txt = summarize_image(ocr_model, sum_model, img=img, min_length=min_length, max_length=max_length)
    return jsonify({'prediction': txt})

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['image']
    # Read the image and convert it to OpenCV format
    filestr = file.read()
    npimg = np.fromstring(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    # Predict the text from the image
    txt = recognize_image(ocr_model, img=img)
    return jsonify({'prediction': txt})


if __name__ == '__main__':
    app.run(debug=True)