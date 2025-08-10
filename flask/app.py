from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import traceback
import librosa
import soundfile as sf
import pandas as pd
from pymongo import MongoClient
import datetime
import os
import ffmpeg
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from extract_script import extract_jitter_shimmer
from pydub import AudioSegment  
import uuid

app = Flask(__name__)
CORS(app)

model = load_model('./model/parkinsons_model.h5')
scaler = joblib.load('./model/scaler.pkl')

client = MongoClient("---MONGO DB URI---") #PLACE YOUR MONGO DB ATLAS URI HERE
db = client.parkinson_db
collection = db.predictions

expected_order = [
    'Jitter(%)', 'Jitter(Abs)', 'Jitter(RAP)', 'Jitter(PPQ5)', 'Jitter:DDP',
    'Shimmer(local)', 'Shimmer(dB)', 'Shimmer(APQ3)',
    'Shimmer(APQ5)', 'Shimmer(APQ11)', 'Shimmer(DDA)'
]


UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def convert_to_wav_pcm(input_path):
    output_path = input_path.rsplit(".", 1)[0] + f"_converted_{uuid.uuid4().hex}.wav"
    (
    ffmpeg
    .input(input_path)
    .output(output_path, ac=1, ar=44100, sample_fmt='s16')
    .overwrite_output()
    .run(cmd=r"C:\ffmpeg-2025-08-07-git-fa458c7243-full_build\ffmpeg-2025-08-07-git-fa458c7243-full_build\bin\ffmpeg.exe", quiet=True)
   )
    return output_path

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        #  Convert to PCM WAV
        converted_path = convert_to_wav_pcm(filepath)

        #  Extract features
        features_dict = extract_jitter_shimmer(converted_path)
        features = [features_dict[key] for key in expected_order]
        # Create DataFrame with correct column names
        df_features = pd.DataFrame([features], columns=expected_order)
        # Scale using the same structure as training
        features_scaled = scaler.transform(df_features)
        print(features)

        # Predict
        probability = model.predict(features_scaled)[0][0]
        prediction = int(probability >= 0.5)
        result_text = "Parkinson's detected" if prediction == 1 else "No Parkinson's detected"

        patient_info = {
            'name': request.form.get('name', ''),
            'age': request.form.get('age', ''),
            'country': request.form.get('country', ''),
            'phone': request.form.get('phone', ''),
            'email': request.form.get('email', '')
        }

        record = {
            "patient_info": patient_info,
            "features": features_dict,
            "prediction": result_text,

            "timestamp": datetime.datetime.now()
        }
        collection.insert_one(record)

        return jsonify({
            'prediction': prediction,
            'probability': round(probability * 100, 2),
            'message': result_text,
            'features': features_dict
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
