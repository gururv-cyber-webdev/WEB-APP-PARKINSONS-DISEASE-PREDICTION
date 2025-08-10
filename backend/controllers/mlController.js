const axios = require('axios');
const Prediction = require('../models/schema.js');

const makePrediction = async (req, res) => {
  try {
    const features = req.body;

    const flaskRes = await axios.post('http://localhost:5000/predict', { features });

    const newPrediction = new Prediction({
      ...features,
      prediction: flaskRes.data.prediction,
    });

    await newPrediction.save();

    res.json({ prediction: flaskRes.data.prediction });
  } catch (err) {
    console.error('Prediction error:', err.message);
    res.status(500).send('Internal server error');
  }
};

module.exports = {
  makePrediction,
};
