import cv2
from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io

app = Flask(__name__)

# Load the PyTorch YOLO model
model = YOLO("best.pt")


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    results = model.predict(Image, conf=0.6, imgsz=640)
    detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]
    return detected_objects

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
