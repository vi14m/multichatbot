from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
import os
import json
import requests
import io
import tempfile
import time
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="MultiChatBot API",
    description="API for a multifunctional chatbot using Groq and OpenRouter",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not GROQ_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Missing API keys. Please set GROQ_API_KEY and OPENROUTER_API_KEY in .env file")

# Model mapping
MODEL_MAPPING = {
    "chat": {"provider": "groq", "model": "llama3-8b-8192"},
    "code": {"provider": "groq", "model": "llama3-70b-8192"},
    "write": {"provider": "groq", "model": "llama3-8b-8192"},
    "brainstorm": {"provider": "openrouter", "model": "qwen/qwen3-32b:free"},
    "math": {"provider": "openrouter", "model": "thudm/glm-z1-32b:free"},
    "research": {"provider": "openrouter", "model": "deepseek/deepseek-r1-0528:free"},
    "email": {"provider": "openrouter", "model": "sarvamai/sarvam-m:free"},
    "text-to-speech": {"provider": "groq", "model": "playai-tts"},
    "transcribe": {"provider": "groq", "model": "distil-whisper-large-v3-en"},
    "moderate": {"provider": "groq", "model": "llama-guard-4-12b"},
}

# Pydantic models for request validation
class ChatRequest(BaseModel):
    mode: str = Field(..., description="Chat mode: chat, code, write, brainstorm, math, research, email, moderate")
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous conversation history")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field(default="default", description="Voice to use for TTS")

class TranscriptionResponse(BaseModel):
    text: str
    processing_time: float

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    token_count: Optional[Dict[str, int]] = None

# Helper functions
def get_groq_response(prompt, model, conversation_history=None):
    """Get response from Groq API"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    return {
        "response": result["choices"][0]["message"]["content"],
        "token_count": {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"]
        }
    }

def get_openrouter_response(prompt, model, conversation_history=None):
    """Get response from OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://multichatbot.app",  # Replace with your actual domain
        "X-Title": "MultiChatBot"
    }
    
    messages = []
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    return {
        "response": result["choices"][0]["message"]["content"],
        "token_count": {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"]
        }
    }

def get_openrouter_tts(text, voice="default"):
    """Convert text to speech using OpenRouter TTS API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://multichatbot.app",  # Replace with your actual domain
        "X-Title": "MultiChatBot"
    }
    
    data = {
        "model": "tts-multilingual",
        "input": text,
        "voice": voice
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/audio/speech",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    return response.content

def get_openrouter_transcription(audio_file):
    """Transcribe audio using OpenRouter API"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://multichatbot.app",
        "X-Title": "MultiChatBot"
    }
    
    files = {
        "file": ("audio.wav", audio_file, "audio/wav")
    }
    
    data = {
        "model": "distil-whisper-large-v3-en"
    }
    
    response = requests.post(
        "https://openrouter.ai/api/v1/audio/transcriptions",
        headers=headers,
        files=files,
        data=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    return result["text"]

# API Routes
@app.get("/")
async def root():
    return {"message": "Welcome to MultiChatBot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    mode = request.mode.lower()
    if mode not in MODEL_MAPPING:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")
    
    model_info = MODEL_MAPPING[mode]
    provider = model_info["provider"]
    model = model_info["model"]
    
    # Prepare mode-specific prompts
    prompt = request.message
    if mode == "code":
        prompt = f"You are an expert programming assistant. Please help with the following code request: {prompt}"
    elif mode == "write":
        prompt = f"You are a creative writing assistant. Please help with the following writing task: {prompt}"
    elif mode == "brainstorm":
        prompt = f"You are a creative brainstorming assistant. Please generate ideas for: {prompt}"
    elif mode == "math":
        prompt = f"You are a math problem-solving assistant. Please solve and explain step-by-step: {prompt}"
    elif mode == "research":
        prompt = f"You are a research assistant. Please provide detailed information on: {prompt}"
    elif mode == "email":
        prompt = f"You are an email drafting assistant. Please help draft an email for: {prompt}"
    elif mode == "moderate":
        prompt = f"You are a content moderation assistant. Please analyze the following content for policy violations: {prompt}"
    
    try:
        if provider == "groq":
            result = get_groq_response(prompt, model, request.conversation_history)
        else:  # openrouter
            result = get_openrouter_response(prompt, model, request.conversation_history)
        
        processing_time = time.time() - start_time
        
        return {
            "response": result["response"],
            "processing_time": processing_time,
            "token_count": result["token_count"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        audio_content = get_openrouter_tts(request.text, request.voice)
        return StreamingResponse(
            io.BytesIO(audio_content),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=speech.mp3"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    start_time = time.time()
    
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    try:
        # Read the file content
        audio_content = await file.read()
        
        # Transcribe the audio
        transcription = get_openrouter_transcription(audio_content)
        
        processing_time = time.time() - start_time
        
        return {
            "text": transcription,
            "processing_time": processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    return MODEL_MAPPING

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)