import cv2 
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils import summarize_image, recognize_image
from models.Summarizer import Summarizer
from models.Recognizer import Recognizer


app = Flask(__name__)
CORS(app, origins=['http://localhost:3000'])

sum_model = Summarizer()
ocr_model = Recognizer()


@app.route('/')
def index() -> str:
    return 'Hello, World!'

@app.route('/api/summarize', methods=['POST'])
def summarize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    try:
        file = request.files['image']
        # Read the image and convert it to OpenCV format
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # Summarize the text from the image
        min_length = request.form.get('min_length', 100)
        min_length = int(min_length)
        max_length = request.form.get('max_length', 150)
        max_length = int(max_length)
        txt = summarize_image(ocr_model, sum_model, img=img, min_length=min_length, max_length=max_length)
        return jsonify({'prediction': txt})
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/recognize', methods=['POST'])
def recognize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    try:
        file = request.files['image']
        # Read the image and convert it to OpenCV format
        filestr = file.read()
        npimg = np.fromstring(filestr, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        # Predict the text from the image
        txt = recognize_image(ocr_model, img=img)
        return jsonify({'prediction': txt})
    except ValueError as e:
        print('error')
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Server started")
    app.run(host='0.0.0.0', port=5000)