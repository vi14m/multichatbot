from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import asyncio
from typing import Optional, List, Dict, Any, Union, DefaultDict, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import io
import json
import os
import requests
import tempfile
import time

from dotenv import load_dotenv
from pathlib import Path

# Import the FileAnalyzer
from file_analyzer import FileAnalyzer

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

# Initialize FileAnalyzer
file_analyzer = FileAnalyzer(GROQ_API_KEY)

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
    "analyze": {"provider": "groq", "model": "llama3-70b-8192"},
    "moderate": {"provider": "groq", "model": "llama-guard-4-12b"},
}

# Pydantic models for request validation
class ChatRequest(BaseModel):
    mode: str = Field(..., description="Chat mode: chat, code, write, brainstorm, math, research, email, moderate")
    message: str = Field(..., description="User message")
    conversation_history: Optional[List[Dict[str, str]]] = Field(default=[], description="Previous conversation history")
    file_context: Optional[Dict[str, Any]] = Field(default=None, description="Context from previously analyzed file")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    voice: Optional[str] = Field(default="default", description="Voice to use for TTS")

class TranscriptionResponse(BaseModel):
    text: str
    processing_time: float

class FileAnalysisResponse(BaseModel):
    file_type: str
    metadata: Dict[str, Any]
    extracted_text: Optional[str] = None
    ai_analysis: Optional[str] = None
    processing_time: float

class ChatResponse(BaseModel):
    response: str
    processing_time: float
    token_count: Optional[Dict[str, int]] = None

# Add to the existing Pydantic models section
class FileChatRequest(BaseModel):
    session_id: str
    message: str
    mode: Optional[str] = "chat"  # chat, code-generation, analyze
    conversation_history: Optional[List[Dict[str, str]]] = []

class FileChatResponse(BaseModel):
    response: str
    file_context: Optional[Dict[str, Any]] = None
    processing_time: float
    token_count: Optional[Dict[str, int]] = None

# Update MODEL_MAPPING with file-specific models
MODEL_MAPPING.update({
    "file-chat": {"provider": "groq", "model": "llama3-70b-8192"},
    "file-code": {"provider": "groq", "model": "llama3-70b-8192"},
    "file-analyze": {"provider": "openrouter", "model": "deepseek/deepseek-r1-0528:free"},
})

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

# File Session Management
class FileSession:
    def __init__(self):
        self.file_content: Dict[str, Any] = {}
        self.last_accessed: datetime = datetime.now()

    def update_content(self, content: Dict[str, Any]):
        self.file_content = content
        self.last_accessed = datetime.now()

    def get_content(self) -> Dict[str, Any]:
        self.last_accessed = datetime.now()
        return self.file_content

    def is_expired(self, ttl_minutes: int = 30) -> bool:
        return datetime.now() - self.last_accessed > timedelta(minutes=ttl_minutes)

# Session storage
file_sessions: DefaultDict[str, FileSession] = defaultdict(FileSession)

# Clean up expired sessions periodically
def cleanup_sessions():
    for session_id in list(file_sessions.keys()):
        if file_sessions[session_id].is_expired():
            del file_sessions[session_id]

@app.on_event("startup")
def startup_event():
    from threading import Timer
    def periodic_cleanup():
        cleanup_sessions()
        Timer(300, periodic_cleanup).start()  # 300 seconds = 5 minutes
    Timer(300, periodic_cleanup).start()

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
    elif mode == "analyze":
        if request.file_context:
            # Include file context in the prompt
            file_name = request.file_context.get("file_name", "")
            file_type = request.file_context.get("file_type", "")
            extracted_text = request.file_context.get("extracted_text", "")
            
            # Limit extracted text to avoid token limits
            if extracted_text and len(extracted_text) > 3000:
                extracted_text = extracted_text[:3000] + "... (text truncated)"
                
            prompt = f"""You are a document analysis assistant. The user has previously uploaded a document with the following details:
            
File: {file_name}
Type: {file_type}

Extracted content from the document:
{extracted_text}

User's question: {prompt}

Please provide a helpful response based on the document content."""
        else:
            prompt = f"You are a document analysis assistant. Please analyze the following content and provide insights: {prompt}"
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

@app.post("/analyze-file", response_model=FileAnalysisResponse)
async def analyze_file(file: UploadFile = File(...)):
    """Analyze an uploaded file (image, PDF, text, etc.)"""
    try:
        # Use the FileAnalyzer to analyze the file
        result = await file_analyzer.analyze_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-file-with-chat", response_model=FileAnalysisResponse)
async def analyze_file_with_chat(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Analyze a file and store its content for future chat interactions"""
    try:
        # Analyze the file
        result = await file_analyzer.analyze_file(file)
        
        # Store the analysis in the session
        file_sessions[session_id].update_content({
            "file_type": result["file_type"],
            "metadata": result["metadata"],
            "extracted_text": result["extracted_text"],
            "ai_analysis": result["ai_analysis"],
            "filename": file.filename
        })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/file-chat", response_model=FileChatResponse)
async def chat_about_file(request: FileChatRequest):
    """Chat about a previously analyzed file"""
    start_time = time.time()
    
    if request.session_id not in file_sessions:
        raise HTTPException(status_code=404, detail="File session not found")
    
    try:
        # Get file content from session
        file_content = file_sessions[request.session_id].get_content()
        
        # Prepare the prompt based on the mode
        if request.mode == "code-generation":
            prompt = f"""Based on the following file content, generate the requested code:
            
File: {file_content.get('filename')}
Content: {file_content.get('extracted_text')}

User's request: {request.message}

Please provide the code implementation along with explanations."""
        else:
            prompt = f"""Help me understand this file:

File: {file_content.get('filename')}
Type: {file_content.get('file_type')}
Content: {file_content.get('extracted_text')}

User's question: {request.message}"""

        # Choose the appropriate model
        model_key = f"file-{request.mode}" if request.mode in ["code", "analyze"] else "file-chat"
        if model_key not in MODEL_MAPPING:
            model_key = "file-chat"  # fallback
            
        model_info = MODEL_MAPPING[model_key]
        
        # Get response from the appropriate provider
        if model_info["provider"] == "groq":
            result = get_groq_response(prompt, model_info["model"], request.conversation_history)
        else:
            result = get_openrouter_response(prompt, model_info["model"], request.conversation_history)
        
        processing_time = time.time() - start_time
        
        return FileChatResponse(
            response=result["response"],
            file_context={
                "filename": file_content.get('filename'),
                "file_type": file_content.get('file_type'),
                "metadata": file_content.get('metadata')
            },
            processing_time=processing_time,
            token_count=result["token_count"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    return MODEL_MAPPING

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)