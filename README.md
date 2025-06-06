# 🤖 Multifunctional Chatbot using OpenRouter & Groq

A modular, multi-purpose chatbot with an advanced React UI that lets users select from a variety of specialized tasks such as **Chatting, Coding, Story Writing, Math solving, Text-to-Speech, Audio Transcription, Research Assistance, Email Drafting**, and **Content Moderation**. Powered by ultra-fast models from **Groq** and **OpenRouter APIs**.

## 🎯 Key Features

- 🧠 Task-specific model routing for precision, speed, and reliability
- 🧑‍💻 Developer-friendly coding assistant with debugging & generation
- ✍️ Creative writing and brainstorming support for stories, blogs, essays, and ideas
- 🔢 Math problem solving with step-by-step explanations
- 🧑‍🔬 Research assistant for detailed info gathering, summarization, and Q&A
- 📧 Email drafting for professional and personal communications
- 🔊 Text-to-Speech (TTS) for audio output of responses
- 🔉 Audio transcription of `.wav` files using Whisper
- 🔒 Real-time content moderation with LLaMA Guard
- 🌐 Clean, dynamic React UI with dropdown mode selector and file uploader

## 🧠 Use Modes & Model Mapping

| Mode             | Purpose                                  | Model                                  | Provider     |
|------------------|------------------------------------------|--------------------------------------|---------------|
| `Chat`           | General conversation and Q&A              | `llama3-8b-8192`                     | Groq         |
| `Code`           | Programming help and code generation      | `llama3-70b-8192`                    | Groq         |
| `Write`          | Long-form writing, storytelling           | `llama-3.3-8b-8192`            | Groq   |
| `Brainstorm`     | Idea generation and creative prompts      | `qwen/qwen3-32b:free`            | OpenRouter   |
| `Math`           | Solve math problems with explanations     | `thudm/glm-z1-32b:free`| OpenRouter   |
| `Research`       | Deep info gathering, summarization, Q&A    | `deepseek/deepseek-r1-0528:free`     | OpenRouter   |
| `Email`          | Draft professional or casual emails       | `sarvamai/sarvam-m:free`            | OpenRouter   |
| `Text-to-Speech` | Convert chatbot replies to audio          | `playai-tts`                   | Groq   |
| `Transcribe`     | Audio file (.wav) transcription           | `distil-whisper-large-v3-en`        | Groq  |
| `Moderate`       | Content safety and policy moderation      | `llama-guard-4-12b`                  | Groq   |

## 🧰 Tech Stack Overview

| Layer         | Technology                              |
|---------------|---------------------------------------|
| Frontend UI   | React Js                                 |
| Backend       | Python (FastAPI)                       |
| Model Access  | Groq API, OpenRouter API              |
| Task Routing  | Dropdown & command-based mode selection|
| File Support  | `.wav` upload for transcription        |
| TTS Output    | Audio playback component integrated    |
| Environment   | Configured via environment variables   |
| Hosting       | Docker ready                           |

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+)
- Python (v3.11+)
- Groq API key
- OpenRouter API key

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/vi14m/multichatbot.git
   cd multichatbot
   ```

2. Set up the backend
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the frontend
   ```bash
   cd ../frontend
   npm install
   ```

4. Create a `.env` file in the backend directory with your API keys
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

5. Start the backend server
   ```bash
   cd backend
   uvicorn app:app --reload
   ```

6. Start the frontend development server
   ```bash
   cd frontend
   npm run dev
   ```

7. Open your browser and navigate to `http://localhost:5173`

## 📁 Project Structure

```
/
├── backend/                # Python FastAPI backend
│   ├── app.py             # Main FastAPI application
│   ├── models/            # Data models
│   ├── routes/            # API routes
│   ├── services/          # Service layer for API integrations
│   └── requirements.txt   # Python dependencies
│
├── frontend/              # React frontend
│   ├── public/            # Static assets
│   ├── src/               # Source code
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── services/      # API service integrations
│   │   ├── types/         # TypeScript type definitions
│   │   ├── utils/         # Utility functions
│   │   ├── App.tsx        # Main App component
│   │   └── main.tsx       # Entry point
│   ├── package.json       # Node dependencies
│   └── vite.config.ts     # Vite configuration
│
└── docker-compose.yml     # Docker configuration
```