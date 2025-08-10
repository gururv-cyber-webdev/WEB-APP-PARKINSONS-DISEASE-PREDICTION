import React, { useState, useRef } from 'react';

function AudioRecorder({ onAudioSubmit }) {
  const [isRecording, setIsRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [previewURL, setPreviewURL] = useState(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  const startRecording = async () => {
    setIsRecording(true);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);
    chunksRef.current = [];

    mediaRecorderRef.current.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'audio/webm' }); // webm format
      setAudioBlob(blob);
      setPreviewURL(URL.createObjectURL(blob));
    };

    mediaRecorderRef.current.start();
  };

  const stopRecording = () => {
    setIsRecording(false);
    mediaRecorderRef.current.stop();
  };

  const handleUpload = (e) => {
    const file = e.target.files[0];
    setAudioBlob(file);
    setPreviewURL(URL.createObjectURL(file));
  };

  const handleSubmit = () => {
    if (!audioBlob) return;
    const file = new File([audioBlob], 'recorded.wav', { type: 'audio/wav' });
    onAudioSubmit(file);
  };

  return (
    <div className="audio-recorder">
      <h3> Voice Input</h3>
      {isRecording ? (
        <button onClick={stopRecording} style={{ backgroundColor: 'red' }}>⏹ Stop Recording</button>
      ) : (
        <button onClick={startRecording} style={{ backgroundColor:'#0cbaba' }}>🎙 Start Recording</button>
      )}

      <p style={{ margin: '10px 0' }}>OR</p>
      <input type="file" accept="audio/*" onChange={handleUpload} />

      {previewURL && (
        <div>
          <p>Preview:</p>
          <audio controls src={previewURL}></audio>
        </div>
      )}

      {audioBlob && (
        <div style={{ marginTop: '10px' }}>
          <button onClick={handleSubmit}> Submit for Prediction</button>
        </div>
      )}
    </div>
  );
}

export default AudioRecorder;
