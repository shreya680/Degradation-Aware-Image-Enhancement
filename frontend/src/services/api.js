import axios from 'axios';

const api = axios.create({
  baseURL: '',
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// ---- Upload ----
export const uploadImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// ---- Analyze ----
export const analyzeImage = (filename) =>
  api.post('/api/analyze', { filename });

// ---- Enhance ----
export const enhanceImage = (filename) =>
  api.post('/api/enhance', { filename });

// ---- Detect ----
export const detectObjects = (filename, enhancedFilename, confThreshold = 0.25) =>
  api.post('/api/detect', { filename, enhanced_filename: enhancedFilename, conf_threshold: confThreshold });

// ---- Evaluate metrics ----
export const evaluateMetrics = (filename, enhancedFilename, referenceFilename = null) =>
  api.post('/api/evaluate', { filename, enhanced_filename: enhancedFilename, reference_filename: referenceFilename });

// ---- All-in-one process ----
export const processImage = (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/api/process', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

// ---- Health ----
export const getHealth = () => api.get('/api/health');

// ---- Models status ----
export const getModelsStatus = () => api.get('/api/models/status');

// ---- Datasets status ----
export const getDatasetsStatus = () => api.get('/api/datasets/status');

// ---- History ----
export const getHistory = () => api.get('/api/history');

export default api;
