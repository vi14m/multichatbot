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
  | 'analyze'
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
  fileAnalysis?: FileAnalysis; // For file analysis results
  toolCalls?: ToolCall[]; // For function calling
  toolResults?: ToolResult[]; // Results of function calls
}

// Chat API request
export interface ChatRequest {
  mode: ChatMode;
  message: string;
  conversation_history?: { role: MessageRole; content: string }[];
  file_context?: {
    file_name: string;
    file_type: string;
    extracted_text: string;
  } | null;
  use_tools?: boolean; // Whether to enable function calling
  selected_tools?: string[]; // List of selected tools to use
}

// Tool call structure
export interface ToolCall {
  id: string;
  function: {
    name: string;
    arguments: string;
  };
}

// Tool result structure
export interface ToolResult {
  tool_call_id: string;
  name: string;
  result?: string;
  error?: string;
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
  tool_calls?: ToolCall[];
  tool_results?: ToolResult[];
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

// File analysis metadata
export interface FileMetadata {
  [key: string]: any;
}

// File analysis response
export interface FileAnalysis {
  file_type: string;
  file_name?: string;
  metadata: FileMetadata;
  extracted_text?: string;
  ai_analysis?: string;
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