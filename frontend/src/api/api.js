import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:5000',  // Flask is running here
});

export const predictParkinson = (features) =>
  API.post('/predict', { features }); // send it with key "features"
