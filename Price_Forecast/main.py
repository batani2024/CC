from tensorflow import keras
import numpy as np
import joblib
from flask import Flask, jsonify, request
import plotly.graph_objects as go
import pandas as pd
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from google.cloud import storage
import os
import tempfile
from sklearn.preprocessing import StandardScaler

# Muat konfigurasi dari .env
from dotenv import load_dotenv
load_dotenv()

# Ambil konfigurasi dari variabel lingkungan
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "dataml-1")
GCS_MODEL_PATH = os.getenv("GCS_MODEL_PATH", "forecast_final_plis")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "batani-1")

# Data tanaman
tanaman_data = {
    'anggur': {
        'data': 'grape_avg_price.xlsx',
        'scaler': 'scaler_grape.pkl',
        'model': 'best_grape_model.h5'
    },
    'apel': {
        'data': 'grouped_prices_apple.xlsx',
        'scaler': 'scaler_apple.pkl',
        'model': 'best_apple_model1.h5'
    },
    'pisang': {
        'data': 'grouped_prices_banana.xlsx',
        'scaler': 'scaler_banana.pkl',
        'model': 'best_banana_model.h5'
    },
    'kacang_hitam': {
        'data': 'grouped_prices_blackgram.xlsx',
        'scaler': 'scaler_blackgram.pkl',
        'model': 'best_blackgram_model.h5'
    },
    'kelapa': {
        'data': 'grouped_prices_coconut.xlsx',
        'scaler': 'scaler_coconut.pkl',
        'model': 'best_coconut_model.h5'
    },
    'kapas': {
        'data': 'grouped_prices_cotton.xlsx',
        'scaler': 'scaler_cotton.pkl',
        'model': 'best_cotton_model.h5'
    },
    'jute': {
        'data': 'jute_avg_price.xlsx',
        'scaler': 'scaler_jute.pkl',
        'model': 'best_jute_model.h5'
    },
    'kopi': {
        'data': 'grouped_prices_coffee.xlsx',
        'scaler': 'scaler_kopi.pkl',
        'model': 'best_kopi_model.h5'
    },
    'lentil': {
        'data': 'lentil_avg_price.xlsx',
        'scaler': 'scaler_lentil.pkl',
        'model': 'best_lentil_model.h5'
    },
    'jagung': {
        'data': 'maize_avg_price.xlsx',
        'scaler': 'scaler_maize.pkl',
        'model': 'best_maize_model.h5'
    },
    'mangga': {
        'data': 'mango_avg_price.xlsx',
        'scaler': 'scaler_mango.pkl',
        'model': 'best_mango_model.h5'
    },
    'melonmusk': {
        'data': 'melonmusk_avg_price.xlsx',
        'scaler': 'scaler_melonmusk.pkl',
        'model': 'best_melonmusk_model.h5'
    },
    'jeruk': {
        'data': 'orange_avg_price.xlsx',
        'scaler': 'scaler_orange.pkl',
        'model': 'best_orange_model.h5'
    },
    'pepaya': {
        'data': 'papaya_avg_price.xlsx',
        'scaler': 'scaler_ papaya.pkl',
        'model': 'best_papaya_model.h5'
    },
    'delima': {
        'data': 'pomegranate_avg_price.xlsx',
        'scaler': 'scaler_pomegranate.pkl',
        'model': 'best_pomegranate_model.h5'
    },
    'padi': {
        'data': 'rice_avg_price.xlsx',
        'scaler': None,  # Tidak ada scaler untuk padi
        'model': 'best_rice_model.h5'
    },
    'semangka': {
        'data': None,  # Tidak ada data semangka 
        'scaler': 'scaler_watermelon.pkl',
        'model': 'best_watermelon_model.h5'
    }
}

# Fungsi untuk mengunduh file dari GCS
def download_from_gcs(bucket_name, source_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    temp_file = os.path.join(tempfile.gettempdir(), source_blob_name.split("/")[-1])
    try:
        blob.download_to_filename(temp_file)
        print(f"File diunduh dari GCS: {source_blob_name} -> {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error downloading {source_blob_name}: {str(e)}")
        raise

# Fungsi transformasi data
def transform(x, scaler):
    return scaler.transform(x.values)

# Fungsi untuk prediksi data
def predict(data, days, model):
    forecast = []
    result_forecast = data[-days:].tolist()
    result_forecast = [item[0] if isinstance(item, list) else float(item) for item in result_forecast]

    for _ in range(days):
        input_data = np.array(result_forecast[-days:]).reshape(1, days, 1)
        prediksi = model.predict(input_data)[0][0]
        forecast.append(prediksi)
        result_forecast.append(prediksi)

    return forecast

# Fungsi untuk mengonversi hasil prediksi
def convert_result(initial_x, converted_x, forecast, days, scaler):
    if not isinstance(initial_x.index, pd.DatetimeIndex):
        initial_x.index = pd.to_datetime(initial_x.index)

    last_30_df = pd.DataFrame(
        scaler.inverse_transform(converted_x[-days:]),
        columns=[initial_x.columns[0]],
        index=initial_x.index[-days:]
    )

    new_dates = pd.date_range(start=last_30_df.index[-1] + pd.Timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame(
        {'Modal Price (Rs./Quintal)': scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()},
        index=new_dates
    )

    extended_df = pd.concat([last_30_df, forecast_df])

    return extended_df, last_30_df

# Flask App
app = Flask(__name__)

# Logging tambahan
import logging
logging.basicConfig(level=logging.INFO)

@app.before_request
def log_request_info():
    logging.info(f"Headers: {request.headers}")
    logging.info(f"Body: {request.get_data()}")

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Forecast API"}), 200

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    tanaman_list = request.args.getlist('tanaman')
    days2forecast = request.args.get('days2forecast', type=int)

    if not tanaman_list:
        return jsonify({'error': 'At least one tanaman name must be provided'}), 400

    results = []
    for tanaman in tanaman_list:
        if tanaman not in tanaman_data:
            results.append({'error': f'Data {tanaman} tidak ada'})
            continue

        try:
            # Unduh file Excel dari GCS
            file_path = download_from_gcs(GCS_BUCKET_NAME, f"{GCS_MODEL_PATH}/{tanaman_data[tanaman]['data']}")
            
            # Unduh model dan scaler
            model_file_path = download_from_gcs(GCS_BUCKET_NAME, f"{GCS_MODEL_PATH}/{tanaman_data[tanaman]['model']}")
            scaler_file_path = download_from_gcs(GCS_BUCKET_NAME, f"{GCS_MODEL_PATH}/{tanaman_data[tanaman]['scaler']}")
            
            model = keras.models.load_model(model_file_path, custom_objects={'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()})
            scaler = joblib.load(scaler_file_path)
            
            # Baca file Excel
            lastprice_df = pd.read_excel(file_path)
            lastprice_df.set_index(lastprice_df.columns[0], inplace=True)
            
            if 'Modal Price (Rs./Quintal)' not in lastprice_df.columns:
                results.append({'error': f'Excel file for {tanaman} missing required column'})
                continue

            initial_data = transform(lastprice_df, scaler)
            prediksi = predict(initial_data, days2forecast, model)
            true, predicted = convert_result(lastprice_df, initial_data, prediksi, days2forecast, scaler)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=true.index, y=true['Modal Price (Rs./Quintal)'], mode='lines', name='True Price', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=predicted.index, y=predicted['Modal Price (Rs./Quintal)'], mode='lines', name='Predicted Price', line=dict(color='red')))
            fig.update_layout(
                title=f'{tanaman} Modal Price Over Time',
                xaxis_title='Price Date',
                yaxis_title='Modal Price (Rs./Quintal)',
                legend_title='Legend',
                template='plotly_white'
            )
            results.append({'tanaman': tanaman, 'plot': fig.to_json()})
        except Exception as e:
            results.append({'error': f'Failed to process {tanaman}: {str(e)}'})

    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)