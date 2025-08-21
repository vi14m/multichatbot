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
import random
import requests
import tempfile
import time

from dotenv import load_dotenv
from pathlib import Path

# Import the FileAnalyzer and tools
from file_analyzer import FileAnalyzer
from tools import tool_registry

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
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Missing API keys. Please set GROQ_API_KEY and OPENROUTER_API_KEY in .env file")

if not TAVILY_API_KEY:
    raise ValueError("Missing TAVILY_API_KEY. Please set TAVILY_API_KEY in .env file")

# Initialize FileAnalyzer
file_analyzer = FileAnalyzer(GROQ_API_KEY, tavily_api_key=TAVILY_API_KEY)

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
    use_tools: Optional[bool] = Field(default=True, description="Whether to use function calling tools")
    selected_tools: Optional[List[str]] = Field(default=None, description="List of selected tools to use")
    consistency_check: Optional[bool] = Field(default=False, description="Whether to perform consistency check on the response")

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
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    consistency_info: Optional[Dict[str, Any]] = None

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
def perform_consistency_check(response1, response2=None, reflection_prompt=None):
    """Perform consistency check by comparing two responses or reflecting on a single response"""
    if response2:
        # Compare two responses
        similarity_score = 0.7  # Placeholder for actual similarity calculation
        differences = []
        
        # Simple difference detection (in a real implementation, use NLP techniques)
        sentences1 = response1.split('.')
        sentences2 = response2.split('.')
        
        # Find unique sentences in each response
        unique_to_1 = [s for s in sentences1 if s and s not in sentences2]
        unique_to_2 = [s for s in sentences2 if s and s not in sentences1]
        
        if unique_to_1:
            differences.append(f"Response 1 uniquely contains: {' '.join(unique_to_1)}")
        if unique_to_2:
            differences.append(f"Response 2 uniquely contains: {' '.join(unique_to_2)}")
        
        consistency_info = {
            "method": "comparison",
            "similarity_score": similarity_score,
            "differences": differences,
            "is_consistent": similarity_score > 0.6 and len(differences) < 3
        }
        
        # Choose the better response or merge them
        if consistency_info["is_consistent"]:
            final_response = response1  # Default to first response if consistent
        else:
            # In a real implementation, use a more sophisticated merging strategy
            final_response = f"Response 1: {response1}\n\nResponse 2: {response2}\n\nAnalysis: The responses show some inconsistencies. {' '.join(differences)}"
    
    else:
        # Self-reflection approach
        # In a real implementation, this would call the model again with a reflection prompt
        reflection = "Upon reflection, the response appears to be accurate and complete based on available information."
        
        if reflection_prompt:
            # This would be a call to the model with the reflection prompt
            pass
        
        consistency_info = {
            "method": "self_reflection",
            "reflection": reflection,
            "is_consistent": True  # Placeholder
        }
        
        final_response = response1  # Keep original response for self-reflection
    
    return final_response, consistency_info

def get_groq_response(prompt, model, conversation_history=None, use_tools=True, max_history_messages=10, selected_tools=None, mode: str = "chat"):
    """Get response from Groq API with optional function calling"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # System prompt to encourage tool use
    system_prompt = "You are a helpful assistant. When the user asks for information that is likely to be recent or that you don't know, use the 'web_search' tool. When asked to execute code, use the 'execute_code_online' tool."
    
    messages = [{"role": "system", "content": system_prompt}]
    
    if conversation_history:
        # Limit conversation history to the last N messages
        limited_history = conversation_history[-max_history_messages:]
        for msg in limited_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    
    # Add function calling if enabled
    if use_tools:
        data["tools"] = tool_registry.get_tool_descriptions(mode=mode, selected_tools=selected_tools)


        data["tool_choice"] = "auto"
    
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    response_message = result["choices"][0]["message"]
    
    # Process tool calls if present
    tool_calls = response_message.get("tool_calls", [])
    tool_results = []
    
    if tool_calls:
        for tool_call in tool_calls:
            function_call = tool_call.get("function", {})
            function_name = function_call.get("name")
            function_args = json.loads(function_call.get("arguments", "{}"))
            
            # Execute the tool
            try:
                tool_result = tool_registry.execute_tool(function_name, function_args)
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "result": tool_result
                })
            except Exception as e:
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "error": str(e)
                })
        
        # If we have tool results, make a second call to process them
        if tool_results:
            # Add tool results to messages
            for tool_result in tool_results:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_result["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": tool_result["name"],
                            "arguments": function_call.get("arguments", "{}")
                        }
                    }]
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": str(tool_result.get("result", tool_result.get("error", "")))
                })
            
            # Make second call to process tool results
            data["messages"] = messages
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            result = response.json()
            response_message = result["choices"][0]["message"]

            # If the LLM doesn't provide a content response after tool execution,
            # generate a default response based on tool results.
            if not response_message.get("content") and tool_results:
                response_content = "I have executed the requested tools. Here are the results:\n\n"
                for tr in tool_results:
                    if tr['name'] == 'web_search' and tr.get('result'):
                        # Special formatting for web_search results
                        response_content += "Here are the web search results:\n\n"
                        results = tr['result'].strip().split('\n\n')
                        for res in results:
                            lines = res.split('\n')
                            title = "N/A"
                            content = "N/A"
                            url = "N/A"
                            for line in lines:
                                if line.startswith("TITLE:"):
                                    title = line.replace("TITLE:", "").strip()
                                elif line.startswith("CONTENT:"):
                                    content = line.replace("CONTENT:", "").strip()
                                elif line.startswith("URL:"):
                                    url = line.replace("URL:", "").strip()
                            response_content += f"- **Title:** {title}\n  **Content:** {content}\n  **URL:** {url}\n\n"
                    else:
                        response_content += f"- **Tool:** `{tr['name']}`\n"
                        if tr.get('result'):
                            # Attempt to parse result as JSON for pretty printing
                            try:
                                parsed_result = json.loads(tr['result'])
                                response_content += f"  **Result:**\n```json\n{json.dumps(parsed_result, indent=2)}\n```\n\n"
                            except json.JSONDecodeError:
                                response_content += f"  **Result:**\n```\n{tr['result']}\n```\n\n"
                        elif tr.get('error'):
                            response_content += f"  **Error:** `{tr['error']}`\n\n"
                response_message["content"] = response_content.strip()

    return {
        "response": response_message.get("content", ""),
        "token_count": {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"]
        },
        "tool_calls": tool_calls,
        "tool_results": tool_results
    }

def get_openrouter_response(prompt, model, conversation_history=None, use_tools=True, max_history_messages=10, selected_tools=None, mode: str = "chat"):
    """Get response from OpenRouter API with optional function calling"""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://multichatbot.app",  # Replace with your actual domain
        "X-Title": "MultiChatBot"
    }
    
    messages = []
    if conversation_history:
        # Limit conversation history to the last N messages
        limited_history = conversation_history[-max_history_messages:]
        for msg in limited_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
    }
    
    # Add function calling if enabled
    if use_tools:
        data["tools"] = tool_registry.get_tool_descriptions(mode=mode, selected_tools=selected_tools)
        data["tool_choice"] = "auto"
    
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    
    result = response.json()
    response_message = result["choices"][0]["message"]
    
    # Process tool calls if present
    tool_calls = response_message.get("tool_calls", [])
    tool_results = []
    
    if tool_calls:
        for tool_call in tool_calls:
            function_call = tool_call.get("function", {})
            function_name = function_call.get("name")
            function_args = json.loads(function_call.get("arguments", "{}"))
            
            # Execute the tool
            try:
                tool_result = tool_registry.execute_tool(function_name, function_args)
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "result": tool_result
                })
            except Exception as e:
                tool_results.append({
                    "tool_call_id": tool_call.get("id"),
                    "name": function_name,
                    "error": str(e)
                })
        
        # If we have tool results, make a second call to process them
        if tool_results:
            # Add tool results to messages
            for tool_result in tool_results:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_result["tool_call_id"],
                        "type": "function",
                        "function": {
                            "name": tool_result["name"],
                            "arguments": function_call.get("arguments", "{}")
                        }
                    }]
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_result["tool_call_id"],
                    "content": str(tool_result.get("result", tool_result.get("error", "")))
                })
            
            # Make second call to process tool results
            data["messages"] = messages
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)
            
            result = response.json()
            response_message = result["choices"][0]["message"]

            # If the LLM doesn't provide a content response after tool execution,
            # generate a default response based on tool results.
            if not response_message.get("content") and tool_results:
                response_content = "I have executed the requested tools. Here are the results:\n\n"
                for tr in tool_results:
                    response_content += f"- **Tool:** `{tr['name']}`\n"
                    if tr.get('result'):
                        # Attempt to parse result as JSON for pretty printing
                        try:
                            import json
                            parsed_result = json.loads(tr['result'])
                            response_content += f"  **Result:**\n```json\n{json.dumps(parsed_result, indent=2)}\n```\n\n"
                        except json.JSONDecodeError:
                            response_content += f"  **Result:**\n```\n{tr['result']}\n```\n\n"
                    elif tr.get('error'):
                        response_content += f"  **Error:** `{tr['error']}`\n\n"
                response_message["content"] = response_content.strip()

    return {
        "response": response_message.get("content", ""),
        "token_count": {
            "prompt_tokens": result["usage"]["prompt_tokens"],
            "completion_tokens": result["usage"]["completion_tokens"],
            "total_tokens": result["usage"]["total_tokens"]
        },
        "tool_calls": tool_calls,
        "tool_results": tool_results
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
        # Only /chat endpoint uses function calling/tools
        if provider == "groq":
            result = get_groq_response(prompt, model, request.conversation_history, use_tools=True, max_history_messages=10, selected_tools=request.selected_tools, mode=mode)
        else:
            result = get_openrouter_response(prompt, model, request.conversation_history, use_tools=True, max_history_messages=10, selected_tools=request.selected_tools, mode=mode)
        
        # Perform consistency check if requested
        consistency_info = None
        final_response = result["response"]
        
        if request.consistency_check:
            # Approach 1: Generate a second response with a slightly different prompt
            if random.choice([True, False]):  # Randomly choose between two approaches
                # Create a slightly different prompt
                rephrased_prompt = f"Please answer this question in a different way: {request.message}"
                
                # Get second response
                if provider == "groq":
                    result2 = get_groq_response(rephrased_prompt, model, request.conversation_history, use_tools=True, max_history_messages=10, selected_tools=request.selected_tools, mode=mode)
                else:
                    result2 = get_openrouter_response(rephrased_prompt, model, request.conversation_history, use_tools=True, max_history_messages=10, selected_tools=request.selected_tools, mode=mode)
                
                # Compare responses
                final_response, consistency_info = perform_consistency_check(result["response"], result2["response"])
            
            # Approach 2: Self-reflection
            else:
                reflection_prompt = f"Please reflect on the following response to the question '{request.message}': {result['response']}. Is it accurate, complete, and helpful?"
                
                # Get reflection
                if provider == "groq":
                    reflection_result = get_groq_response(reflection_prompt, model, [], use_tools=False, max_history_messages=10)
                else:
                    reflection_result = get_openrouter_response(reflection_prompt, model, [], use_tools=False, max_history_messages=10)
                
                # Process reflection
                final_response, consistency_info = perform_consistency_check(result["response"], reflection_prompt=reflection_result["response"])
        
        processing_time = time.time() - start_time
        
        return {
            "response": final_response,
            "processing_time": processing_time,
            "token_count": result["token_count"],
            "tool_calls": result.get("tool_calls"),
            "tool_results": result.get("tool_results"),
            "consistency_info": consistency_info
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
            result = get_groq_response(prompt, model_info["model"], request.conversation_history, mode=request.mode)
        else:
            result = get_openrouter_response(prompt, model_info["model"], request.conversation_history, mode=request.mode)
        
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