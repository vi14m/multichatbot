# 🤖 Multifunctional Chatbot using Groq

A modular, multi-purpose chatbot with an advanced React UI that lets users select from a variety of specialized tasks such as **Chatting, Coding, Story Writing, Math solving, Text-to-Speech, Audio Transcription, Research Assistance, Email Drafting, File Analysis**, and **Content Moderation**. Powered by ultra-fast models from **Groq API**.

---

## 🎯 Key Features

- 🧠 Task-specific model routing for precision, speed, and reliability
- 🧑‍💻 Developer-friendly coding assistant with debugging & generation
- ✍️ Creative writing and brainstorming support for stories, blogs, essays, and ideas
- 🔢 Math problem solving with step-by-step explanations
- 🧑‍🔬 Research assistant for detailed info gathering, summarization, and Q&A
- 📧 Email drafting for professional and personal communications
- 🔊 Text-to-Speech (TTS) for audio output of responses
- 🔉 Audio transcription of `.wav` files using Whisper
- 📄 File analysis for images, PDFs, and text files with AI-powered insights
- 🔒 Real-time content moderation with LLaMA Guard
- 🌐 Clean, dynamic React UI with dropdown mode selector and file uploader

---

## 🧠 Use Modes & Model Mapping

| Mode             | Purpose                                  | Model                                  | Provider     |
|------------------|------------------------------------------|--------------------------------------|---------------|
| `Chat`           | General conversation and Q&A              | `llama3-8b-8192`                     | Groq         |
| `Code`           | Programming help and code generation      | `llama3-70b-8192`                    | Groq         |
| `Write`          | Long-form writing, storytelling           | `llama3-8b-8192`                     | Groq         |
| `Brainstorm`     | Idea generation and creative prompts      | `llama3-8b-8192`                     | Groq         |
| `Math`           | Solve math problems with explanations     | `llama3-8b-8192`                     | Groq         |
| `Research`       | Deep info gathering, summarization, Q&A   | `llama3-70b-8192`                    | Groq         |
| `Email`          | Draft professional or casual emails       | `llama3-8b-8192`                     | Groq         |
| `Text-to-Speech` | Convert chatbot replies to audio          | `playai-tts`                         | Groq         |
| `Transcribe`     | Audio file (.wav) transcription           | `distil-whisper-large-v3-en`         | Groq         |
| `Analyze`        | Analyze images, PDFs, and text files      | `llava-next-34b`                     | Groq         |
| `Moderate`       | Content safety and policy moderation      | `llama-guard-4-12b`                  | Groq         |

---

## 🧰 Tech Stack Overview

| Layer         | Technology                              |
|---------------|---------------------------------------|
| Frontend UI   | React Js                                 |
| Backend       | Python (FastAPI)                       |
| Model Access  | Groq API,OpenRouter API                                        |
| Task Routing  | Dropdown & command-based mode selection|
| File Support  | `.wav` upload for transcription, images (JPG, PNG), PDFs, text files (TXT, CSV, JSON, MD) for analysis |
| OCR           | EasyOCR (pure Python, no Tesseract needed) |
| TTS Output    | Audio playback component integrated    |
| Environment   | Configured via environment variables   |
| Hosting       | Docker & Render ready                  |

---

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
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Set up the frontend
   ```bash
   cd ../frontend
   npm install
   ```

4. Create a `.env` file in the backend directory with your API key
   ```
   GROQ_API_KEY=your_groq_api_key
   OPENROUTER_API_KEY=your_openrouter_api_key
   ```

5. Start the backend server
   ```bash
   cd backend
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. Start the frontend development server
   ```bash
   cd frontend
   npm run dev
   ```

7. Open your browser and navigate to `http://localhost:5173`

---

## 📁 Project Structure

```
multichatbot/
├── backend/                # Python FastAPI backend
│   ├── app.py             # Main FastAPI application
│   ├── file_analyzer.py   # File analysis logic
│   ├── requirements.txt   # Python dependencies
│   └── ...
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
└── docker-compose.yml     # Docker configuration
```

---

## 📄 File Analysis Feature

### Supported File Types

- **Images**: JPG, JPEG, PNG, GIF, BMP
- **Documents**: PDF
- **Text Files**: TXT, CSV, JSON, MD

### Analysis Capabilities

- **Image Analysis**: Extract metadata (dimensions, format, size) and perform OCR to extract text (EasyOCR)
- **PDF Analysis**: Extract metadata, document info, and text content
- **Text File Analysis**: Analyze structure and content of text files
- **CSV Analysis**: Extract headers, sample data, and structure information
- **AI-Powered Insights**: Get AI-generated summaries and insights about the file contents using Groq LLMs

### Technical Implementation

- **Backend**: A `FileAnalyzer` class that handles different file types and extracts relevant information
- **OCR**: EasyOCR for extracting text from images (no Tesseract needed)
- **PDF Processing**: PyPDF2 and PyMuPDF for extracting text and metadata from PDFs
- **AI Analysis**: Groq API for generating insights about file contents
- **Frontend**: Enhanced UI components for file upload and analysis result display

---

## 🚀 Deploying on Render

1. **Push your latest code to your Git repository.**
2. **Update your `requirements.txt` and `package.json` if you added new dependencies.**
3. **Set the `GROQ_API_KEY` environment variable in your Render dashboard.**
4. **Set the build and start commands:**
   - Backend: `uvicorn app:app --host 0.0.0.0 --port 8000`
   - Frontend: `npm run build` and `npm start` (or as per your setup)
5. **Trigger a new deploy in Render.**
6. **Test the analyze-file and chat features on your deployed site.**

---

## 🙋 FAQ

- **Q: Do I need Tesseract installed?**
  - No! This project uses EasyOCR for image text extraction, so no system-level OCR dependencies are required.
- **Q: How do I add more file types?**
  - Extend the `FileAnalyzer` backend class and update the frontend file type checks.

---