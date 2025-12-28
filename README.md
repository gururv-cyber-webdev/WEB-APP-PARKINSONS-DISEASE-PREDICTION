# 🧠 Secure Web-Based Application for Predicting Parkinson’s Disease

### Using Speech Signal Features and Machine Learning Techniques

## 📌 Overview

Parkinson’s disease is a progressive neurodegenerative disorder where **speech instability appears as one of the earliest symptoms**. This project presents a **secure, non-invasive, and web-based screening system** that predicts Parkinson’s disease using **speech signal features** extracted from real-time or uploaded voice recordings.

The system is designed for **early detection, remote accessibility, and patient privacy**, making it especially useful for **rural and resource-limited healthcare environments**.

---

## 🎯 Objectives

* Enable **early Parkinson’s disease screening** using voice recordings
* Extract clinically relevant speech biomarkers such as **jitter, shimmer, pitch, and MFCCs**
* Improve prediction accuracy using **ensemble machine learning techniques**
* Provide a **secure web application** with real-time feedback
* Offer **basic personalized treatment suggestions** (diet, yoga, exercises) for affected users

---

## 🧩 System Architecture

```
User Voice Input
        ↓
Noise Reduction & Voice Preprocessing
        ↓
Feature Extraction (Speech Biomarkers)
        ↓
Feature Selection & Normalization
        ↓
Multiple ML Classifiers
        ↓
Ensemble Model (Stacking / Voting)
        ↓
Secure Web Application
        ↓
Prediction Result + Treatment Guidance
```

---

## 🔬 Speech Features Used

* **Jitter** – Frequency variation between vocal fold vibrations
* **Shimmer** – Amplitude variation in voice
* **Pitch** – Fundamental frequency stability
* **MFCCs** – Mel-frequency cepstral coefficients
* **Formants & Energy-based features**

These features act as **biomarkers** for detecting vocal tremors associated with Parkinson’s disease.

---

## ⚙️ Modules

### 1️⃣ Voice Recording & Feature Extraction

* Live voice recording or `.wav` file upload
* Libraries used:

  * `sounddevice`
  * `scipy.io.wavfile`
  * `librosa`
  * `praat-parselmouth`

---

### 2️⃣ Voice Preprocessing (Dedicated ML Model)

* Noise reduction using:

  * Wiener filtering
  * Deep learning–based denoising
* Normalization and silence removal
* Ensures **robust performance under real-world conditions**

---

### 3️⃣ Feature Selection

* Minimum Redundancy Maximum Relevance (mRMR)
* Statistical correlation analysis
* Helps remove redundancy while retaining medical relevance

---

### 4️⃣ Classification Models

* Logistic Regression
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* CatBoost
* XGBoost
* AdaBoost
* Multilayer Perceptron (MLP)

---

### 5️⃣ Ensemble Learning

* **Stacking classifier** with **XGBoost as meta-learner**
* Improves accuracy by prioritizing strong models
* Final accuracy achieved: **~95%**

---

### 6️⃣ Secure Web-Based Application

* Displays:

  * Disease detection result
  * Confidence score
  * Personalized lifestyle guidance
* Designed for **telemedicine and remote screening**

---

## 🌐 Web Application Tech Stack

### Frontend

* **React.js**
* Responsive UI
* Real-time audio input
* Smooth animations & transitions

### Backend

* **Flask API**
* Handles:

  * Audio processing
  * Feature extraction
  * Model inference
  * Secure response handling

### Database

* **MongoDB Atlas (Cloud-based)**
* Secure storage of predictions and logs

---

## 🔐 Security Implementation

* Password hashing using **bcrypt**
* **JWT-based authentication**
* All communication over **HTTPS**
* **Raw voice recordings deleted immediately after feature extraction**
* Role-based access control in database
* Designed with **privacy-by-design principles**

---

## 📊 Datasets Used

* UCI Parkinson Speech Dataset
* Multiple speech datasets including:

  * Sustained vowels
  * Words
  * Sentences
  * Home-recorded samples

---

## 🚀 How the Flask API Works

1. Receives voice input from frontend
2. Applies noise reduction and preprocessing
3. Extracts speech features
4. Feeds features into trained ensemble model
5. Returns prediction & confidence score as JSON
6. Stores results securely and deletes raw audio

---

## 🧪 Results

* **Stacking ensemble accuracy:** ~95.20%
* High recall to reduce false negatives
* Consistent performance across datasets
* Response time under **3 seconds**

---

## 🧩 Challenges Faced

* Handling confidential voice data → solved using public datasets & strict deletion policies
* Noise in real-world audio → addressed with denoising pipelines
* Dataset imbalance → handled using SMOTE
* Secure deployment of healthcare data → addressed via encryption & authentication

---

## 🔮 Future Work

* Commercial deployment for hospitals and individual users
* Support for multiple languages and accents
* Integration with wearable devices for continuous monitoring
* Extension to other neurodegenerative diseases (Alzheimer’s, ALS)
* Multimodal inputs (facial expressions, motion patterns, MRI)

---


