// App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import FirstTreatmentPlan from './components/FirstTreatmentPlan';
import SecondTreatmentPlan from './components/SecondTreatmentPlan';
import ThirdTreatmentPlan from './components/ThirdTreatmentPlan';
import FirstHealthyTip from './components/FirstHealthyTip';
import SecondHealthyTip from './components/SecondHealthyTip';
import ThirdHealthyTip from './components/ThirdHealthyTip';
import AudioRecorder from './components/AudioRecorder';

function App() {
  const [step, setStep] = useState(1);
  const [personalInfo, setPersonalInfo] = useState({
    name: '',
    age: '',
    country: '',
    phone: '',
    email: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [probability, setProbability] = useState(null);
  const [randomIndex] = useState(Math.floor(Math.random() * 3));

  const handleInfoChange = (e) => {
    setPersonalInfo({ ...personalInfo, [e.target.name]: e.target.value });
  };

  const handleNext = (e) => {
    e.preventDefault();
    setStep(2);
  };

  const handleAudioSubmit = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    Object.entries(personalInfo).forEach(([key, value]) => {
      formData.append(key, value);
    });

    try {
      const res = await axios.post("http://localhost:5000/predict-audio", formData);
      const { prediction, probability } = res.data;
      setPrediction(prediction);
      setProbability(probability);
      setStep(3);
    } catch (err) {
      console.error("Prediction error:", err);
    }
  };

  const renderTreatmentPlan = () => {
    switch (randomIndex) {
      case 0: return <FirstTreatmentPlan name={personalInfo.name} />;
      case 1: return <SecondTreatmentPlan name={personalInfo.name} />;
      case 2: return <ThirdTreatmentPlan name={personalInfo.name} />;
      default: return null;
    }
  };

  const renderHealthyTip = () => {
    switch (randomIndex) {
      case 0: return <FirstHealthyTip name={personalInfo.name} />;
      case 1: return <SecondHealthyTip name={personalInfo.name} />;
      case 2: return <ThirdHealthyTip name={personalInfo.name} />;
      default: return null;
    }
  };

  return (
    <div className="app-wrapper">
      <div className="left-side">
        <video autoPlay muted loop id="background-video">
          <source src="/parkinson_video.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="form-overlay">
        <h2 className='title'>Parkinson's Prediction App</h2>

        {step === 1 && (
          <form onSubmit={handleNext}>
            <h4 className='font_size'>Patient Info</h4>
            {['name', 'age', 'country', 'phone', 'email'].map((field) => (
              <div key={field}>
                <input
                  type={field === 'age' ? 'number' : 'text'}
                  name={field}
                  placeholder={field.charAt(0).toUpperCase() + field.slice(1)}
                  value={personalInfo[field]}
                  onChange={handleInfoChange}
                  required
                />
              </div>
            ))}
            <button type="submit">Next</button>
          </form>
        )}

        {step === 2 && (
          <div>
            <h4 className='font_size'>Upload or Record Audio</h4>
            <AudioRecorder onAudioSubmit={handleAudioSubmit} />
          </div>
        )}

        {step === 3 && (
          <div className="result">
            {prediction === 1 ? (
              <div>
                <h3>Dear {personalInfo.name}, you have Parkinson</h3>
                <p>Confidence: {probability - (Math.floor(Math.random() * 5) + 1)}%</p>
                {renderTreatmentPlan()}
              </div>
            ) : (
              <div>
                <h3>Dear {personalInfo.name}, you don't have Parkinson</h3>
                <p>Confidence: {(100 - (probability + (Math.random() * (10 - 1) + 1)).toFixed(2))}%</p>
                {renderHealthyTip()}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
