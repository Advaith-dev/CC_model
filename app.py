from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import base64

# Initialize the Flask app
app = Flask(__name__)

# Load the YOLO model (YOLOv5 in this case)
model = YOLO("best.pt")  

# Endpoint to process the uploaded image
@app.route('/predict', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    # Read the uploaded image
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read()))

    # Run YOLO object detection
    results = model.predict(img, conf=0.6, imgsz=640)
    
    # Get the annotated frame
    detected_objects = [model.names[int(cls)] for cls in results[0].boxes.cls]
    return detected_objects

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
