from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from PIL import Image
import requests



# Load the model and scaler
model = keras.models.load_model("nn.h5")
scaler = joblib.load(open('scaler.pkl', 'rb'))
parameter_ranges = joblib.load(open('parameter_ranges.pkl', 'rb'))
label_encoder = joblib.load(open('label_encoder.pkl', 'rb'))
tanaman_data = joblib.load(open('tanaman_data (1).pkl', 'rb'))

app = Flask(__name__)
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the crop prediction API. Use POST /predict_crop to get predictions."})

# Route to get the temperature, humidity, and rainfall as query parameters
@app.route('/predict_crop', methods=['POST', 'GET'])
def predict_crop():
    try:
        
        # Parse input from query parameters
        temperature = (request.args.get('temperature', type=float))
        humidity = (request.args.get('humidity', type=float))
        rainfall = (request.args.get('rainfall', type=float))

        # Validate and preprocess input
        temperature = max(min(temperature, parameter_ranges.loc['max', 'temperature']), parameter_ranges.loc['min', 'temperature'])
        humidity = max(min(humidity, parameter_ranges.loc['max', 'humidity']), parameter_ranges.loc['min', 'humidity'])
        rainfall = max(min(rainfall, parameter_ranges.loc['max', 'rainfall']), parameter_ranges.loc['min', 'rainfall'])

        new_data = [[temperature, humidity, rainfall]]
        new_data_normalized = scaler.transform(new_data)
        predictions = model.predict(new_data_normalized)[0]

        # Filter predictions
        threshold_min = 0.1
        threshold_max = 1.0
        predictions_tensor = tf.constant(predictions, dtype=tf.float32)
        filtered_indices = tf.where((predictions_tensor >= threshold_min) & (predictions_tensor <= threshold_max))
        filtered_indices = tf.squeeze(filtered_indices).numpy()

        if isinstance(filtered_indices, np.int64):
            filtered_indices = [filtered_indices]

        if len(filtered_indices) > 0:
            filtered_probabilities = [float(predictions[i]) for i in filtered_indices]  # Convert to standard float
            recommended_labels = label_encoder.inverse_transform(filtered_indices)

            # Format results
            results = sorted(zip(recommended_labels, filtered_probabilities), key=lambda x: x[1], reverse=True)
            response = []
            for label, prob in results:
                crop_info = tanaman_data.get(label, {})
                label = label.replace('_', ' ')  # Transform label
                response.append({
                    "Tanaman": label,
                    "Probabilitas": float(prob * 100),  # Convert to standard float
                    "Info": {
                        "Deskripsi": crop_info.get('deskripsi', 'N/A'),
                        "Jenis tanah": crop_info.get('jenis_tanah', 'N/A'),
                        "pH tanah": crop_info.get('ph', 'N/A'),
                        "Waktu penanaman": crop_info.get('waktu_penanaman', 'N/A'),
                        "Cara merawat": crop_info.get('cara_merawat', 'N/A'),
                        "Gambar": crop_info.get('gambar', 'N/A')
                    }
                })
            return jsonify(response)
        else:
            return jsonify({"message": "No recommendations meet the specified probability range."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)