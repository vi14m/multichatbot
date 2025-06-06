// Available chat modes
export type ChatMode = 
  | 'chat'
  | 'code'
  | 'write'
  | 'brainstorm'
  | 'math'
  | 'research'
  | 'email'
  | 'text-to-speech'
  | 'transcribe'
  | 'moderate';

// Chat message role
export type MessageRole = 'user' | 'assistant' | 'system';

// Chat message structure
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
  mode?: ChatMode;
  audioUrl?: string; // For text-to-speech messages
}

// Chat API request
export interface ChatRequest {
  mode: ChatMode;
  message: string;
  conversation_history?: { role: MessageRole; content: string }[];
}

// Chat API response
export interface ChatResponse {
  response: string;
  processing_time: number;
  token_count?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// Text-to-Speech API request
export interface TTSRequest {
  text: string;
  voice?: string;
}

// Transcription API response
export interface TranscriptionResponse {
  text: string;
  processing_time: number;
}

// Model information
export interface ModelInfo {
  provider: 'groq' | 'openrouter';
  model: string;
}

// Model mapping
export type ModelMapping = {
  [K in ChatMode]: ModelInfo;
};