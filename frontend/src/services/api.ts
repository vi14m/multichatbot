import axios from 'axios';
import { ChatRequest, ChatResponse, TTSRequest, TranscriptionResponse, ModelMapping, FileAnalysis } from '@/types/chat';

const API_URL = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Chat API
export const sendChatMessage = async (request: ChatRequest): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>('/chat', request);
  return response.data;
};

// Text-to-Speech API
export const convertTextToSpeech = async (request: TTSRequest): Promise<Blob> => {
  const response = await api.post('/tts', request, {
    responseType: 'blob',
  });
  return response.data;
};

// Transcription API
export const transcribeAudio = async (file: File): Promise<TranscriptionResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<TranscriptionResponse>('/transcribe', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// File Analysis API
export const analyzeFile = async (file: File): Promise<FileAnalysis> => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<FileAnalysis>('/analyze-file', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

// Get available models
export const getModels = async (): Promise<ModelMapping> => {
  const response = await api.get<ModelMapping>('/models');
  return response.data;
};

// Error handling interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMessage = error.response?.data?.detail || 'An error occurred';
    console.error('API Error:', errorMessage);
    return Promise.reject(errorMessage);
  }
);

export default api;