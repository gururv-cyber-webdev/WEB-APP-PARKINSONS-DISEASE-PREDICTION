// backend/models/Prediction.js
const mongoose = require('mongoose');

const PredictionSchema = new mongoose.Schema({
  subject_id: Number,
  Jitter_percent: Number,
  Jitter_Abs: Number,
  Jitter_RAP: Number,
  Jitter_PPQ5: Number,
  Jitter_DDP: Number,
  Shimmer_local: Number,
  Shimmer_dB: Number,
  Shimmer_APQ3: Number,
  Shimmer_APQ5: Number,
  Shimmer_APQ11: Number,
  Shimmer_DDA: Number,
  prediction: String,
  createdAt: {
    type: Date,
    default: Date.now,
  },
});

module.exports = mongoose.model('Prediction', PredictionSchema);
