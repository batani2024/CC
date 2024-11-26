from tensorflow import keras
import numpy as np
import joblib
from flask import Flask, jsonify, request

referensi_buah = {
    1: "apel",
    2: "pisang",
    3: "lentil hitam",
    4: "kacang arab",
    5: "kelapa",
    6: "kopi",
    7: "kapas",
    8: "anggur",
    9: "goni",
    10: "kacang merah",
    11: "kacang lentil",
    12: "jagung",
    13: "mangga",
    14: "kacang mat",
    15: "kacang hijau",
    16: "melon kuning",
    17: "jeruk",
    18: "pepaya",
    19: "kacang bali",
    20: "delima",
    21: "padi/beras",
    22: "semangka",
}

# Load the model and scaler
model = keras.models.load_model("nn.h5")
scaler = joblib.load('scaler.pkl')

def transform(x):
    # Transform the data using the scaler
    x = np.array(x).reshape(1, -1)
    data = scaler.transform(x)
    return data

def predict(data):
    # Make a prediction using the model
    prediction = model.predict(data)
    rekomen = np.argsort(prediction.reshape(-1))[-3:][::-1]
    dct_rekomendasi = {}
    for i in range(len(rekomen)):
        dct_rekomendasi[f'Rekomendasi ke-{i+1}']=referensi_buah[rekomen[i]+1]
    return dct_rekomendasi

app = Flask(__name__)

# Route to get the temperature, humidity, and rainfall as query parameters
@app.route('/tanaman')
def get_weather_and_predict():
    # Get the query parameters
    temperature = request.args.get('temperature', type=float)
    humidity = request.args.get('humidity', type=float)
    rainfall = request.args.get('rainfall', type=float)
    
    # Ensure all parameters are provided
    if temperature is None or humidity is None or rainfall is None:
        return jsonify({"error": "Missing required parameters"}), 400
    
    # Combine them into a list for prediction
    input_data = [temperature, humidity, rainfall]
    
    # Transform the input data and make a prediction
    transformed_data = transform(input_data)
    prediction = predict(transformed_data)
    
    return jsonify(prediction)

if __name__ == "__main__":
    app.run(debug=True)
