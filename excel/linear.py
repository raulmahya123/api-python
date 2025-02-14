import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
MODEL_PATH = "linear_regression.h5"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Baca data dari Excel
    df = pd.read_excel(file_path)

    # Pastikan kolom yang diperlukan ada
    if "luas_rumah" not in df.columns or "harga_rumah" not in df.columns:
        return jsonify({"error": "File harus memiliki kolom 'luas_rumah' dan 'harga_rumah'"}), 400

    # Pisahkan fitur dan target
    X = df[["luas_rumah"]].values
    y = df["harga_rumah"].values

    # Tentukan threshold harga rumah (gunakan median sebagai batas antara "Mahal" dan "Murah")
    harga_median = np.median(y)

    # Normalisasi data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Bagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Buat model Linear Regression dengan Keras
    model = Sequential([
        Dense(1, input_shape=(1,), activation="linear")
    ])

    # Compile model
    model.compile(optimizer="adam", loss="mse")

    # Latih model
    history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test))

    # Simpan model ke file .h5
    model.save(MODEL_PATH)

    # Tentukan apakah harga rumah termasuk "Mahal" atau "Murah"
    df["kategori_harga"] = df["harga_rumah"].apply(lambda x: "Mahal" if x > harga_median else "Murah")

    return jsonify({
        "message": "Model berhasil dilatih dan disimpan dalam format .h5",
        "file_uploaded": file.filename,
        "final_loss": history.history["loss"][-1],  # Menampilkan loss terakhir
        "harga_median": harga_median,
        "kategori_harga": df[["luas_rumah", "harga_rumah", "kategori_harga"]].to_dict(orient="records")
    })


if __name__ == "__main__":
    app.run(debug=True)
