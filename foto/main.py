import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Load model yang telah disimpan
model = load_model("cifar10_model.h5")

# Label CIFAR-10
class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

# Folder untuk menyimpan gambar yang diunggah
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Load dan preprocess gambar
    img = load_img(file_path, target_size=(32, 32))  # CIFAR-10 berukuran 32x32
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    return jsonify({
        "filename": file.filename,
        "prediction": class_names[predicted_class],
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
