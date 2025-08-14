import os
import tempfile
import time
import re
import json
import csv
import markdown
import requests
import chromadb
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import UploadFile, HTTPException
from collections import Counter
from statistics import mean, median, stdev
from io import StringIO

# Import Tavily for search, extract and crawl functionality
from tavily import TavilyClient

try:
    import PyPDF2
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    import camelot
    HAS_TABLE_SUPPORT = True
except ImportError:
    HAS_TABLE_SUPPORT = False


class FileAnalyzer:
    """Advanced file analyzer with optimized RAG pipeline and knowledge fallback."""
    
    def __init__(self, groq_api_key: str, tavily_api_key: str = None):
        """
        Initialize the FileAnalyzer with API keys.
        
        Args:
            groq_api_key: API key for Groq service
            tavily_api_key: API key for Tavily search service
        """
        if not groq_api_key:
            raise ValueError("Groq API key is required")
            
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        
        # Initialize Tavily client if API key is provided
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        
        # Initialize ChromaDB with optimized settings
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "rag_collection",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None  # We'll handle embeddings ourselves
        )
        
        # Configuration
        self.embedding_model = "BAAI/bge-small-en-v1.5"
        self.embedding_batch_size = 32
        self.chunk_min_size = 256
        self.chunk_max_size = 1024
        self.rag_top_k = 5
        self.rag_min_score = 0.85

    async def analyze_file(self, file: UploadFile) -> Dict[str, Any]:
        """
        Analyze a file and extract structured information.
        
        Args:
            file: UploadFile object to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        
        # Validate and process file
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
            
        file_ext = os.path.splitext(file.filename)[1].lower()
        content = await file.read()
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Route to appropriate analyzer
            if file_ext == ".pdf":
                result = self._analyze_pdf(temp_path)
            elif file_ext in [".txt", ".csv", ".md", ".json"]:
                result = self._analyze_text(content, file_ext)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
            # Process text content if available
            if "extracted_text" in result and result["extracted_text"]:
                # Advanced semantic chunking
                chunks = self._semantic_chunk_text(result["extracted_text"])
                
                # Get embeddings in batches
                embeddings = []
                for i in range(0, len(chunks), self.embedding_batch_size):
                    batch = chunks[i:i + self.embedding_batch_size]
                    embeddings.extend(self._get_hf_embeddings(batch))
                
                # Store in vector database
                ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
                metadatas = [{
                    "filename": file.filename,
                    "chunk_id": i,
                    "file_type": file_ext[1:],  # Remove dot
                    "length": len(chunk)
                } for i, chunk in enumerate(chunks)]
                
                self.chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=chunks,
                    metadatas=metadatas
                )
                
                # Get AI analysis with context management
                result["ai_analysis"] = self._get_ai_analysis(
                    result["extracted_text"],
                    file.filename
                )
            
            result["processing_time"] = time.time() - start_time
            return result
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def _get_hf_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings using HuggingFace Inference API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        HF_TOKEN = os.environ.get('HF_TOKEN')
        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"https://router.huggingface.co/hf-inference/models/{self.embedding_model}/pipeline/feature-extraction",
                headers=headers,
                json={"inputs": texts},
                timeout=30
            )
            response.raise_for_status()
            
            embeddings = response.json()
            if isinstance(embeddings, dict):
                return [embeddings]
            return embeddings
            
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f"Embedding generation failed: {str(e)}"
            )

    def _semantic_chunk_text(self, text: str) -> List[str]:
        """
        Split text into semantically meaningful chunks.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        # First split by major sections
        sections = re.split(r'\n\s*\n|\f', text)
        chunks = []
        current_chunk = ""
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
                
            # If section is too big, split further
            if len(section) > self.chunk_max_size:
                sentences = re.split(r'(?<=[.!?])\s+', section)
                for sent in sentences:
                    if len(current_chunk) + len(sent) > self.chunk_max_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        # If single sentence is too long, split by words
                        if len(sent) > self.chunk_max_size:
                            words = sent.split()
                            for word in words:
                                if len(current_chunk) + len(word) > self.chunk_max_size:
                                    if current_chunk:
                                        chunks.append(current_chunk)
                                        current_chunk = ""
                                current_chunk += " " + word if current_chunk else word
                        else:
                            current_chunk = sent
                    else:
                        current_chunk += " " + sent if current_chunk else sent
            else:
                if len(current_chunk) + len(section) > self.chunk_max_size:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = section
                else:
                    current_chunk += "\n\n" + section if current_chunk else section
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Merge small chunks (except possibly the last one)
        merged_chunks = []
        buffer = ""
        
        for chunk in chunks:
            if len(buffer) + len(chunk) < self.chunk_min_size and chunk != chunks[-1]:
                buffer += "\n\n" + chunk if buffer else chunk
            else:
                if buffer:
                    merged_chunks.append(buffer)
                    buffer = ""
                merged_chunks.append(chunk)
        
        return merged_chunks

    def rag_query(self, question: str, filename: str) -> str:
        """
        Retrieve and generate answer using RAG pipeline with knowledge fallback.
        
        Args:
            question: User's question
            filename: Name of file to query against
            
        Returns:
            Generated answer
        """
        # Get question embedding
        question_embed = self._get_hf_embeddings([question])[0]
        
        # Retrieve relevant chunks
        results = self.chroma_collection.query(
            query_embeddings=[question_embed],
            n_results=self.rag_top_k * 2,  # Retrieve extra for reranking
            where={"filename": filename},
            include=["documents", "distances", "metadatas"]
        )
        
        # Rerank and filter results
        scored_chunks = []
        for doc, dist, meta in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            score = 1 - dist  # Convert distance to similarity score
            if score >= self.rag_min_score:
                scored_chunks.append({
                    "text": doc,
                    "score": score,
                    "length": meta.get("length", 0)
                })
        
        # Sort by score and then by length (prefer more content at same score)
        scored_chunks.sort(key=lambda x: (-x["score"], -x["length"]))
        
        # Select top chunks without exceeding context window
        selected_chunks = []
        total_length = 0
        max_context_length = 3000  # Leave room for prompt and response
        
        for chunk in scored_chunks[:self.rag_top_k]:
            if total_length + len(chunk["text"]) <= max_context_length:
                selected_chunks.append(chunk["text"])
                total_length += len(chunk["text"])
        
        # Generate different prompts based on whether we have context
        if selected_chunks:
            context = "\n\n---\n\n".join(selected_chunks)
            prompt = f"""You are an AI assistant helping with a document. Use the following context where relevant, 
but supplement with your general knowledge when needed. If the context contradicts your knowledge, 
give preference to the context but note any discrepancies.

Document Context:
{context}

Question: {question}

Please provide a helpful answer using both the context and your knowledge:"""
        else:
            prompt = f"""You are an AI assistant. No specific context was found for this question in the documents. 
Please answer using your general knowledge:

Question: {question}

Please provide a helpful answer:"""
        
        return self._generate_ai_response(prompt)

    def _generate_ai_response(self, prompt: str) -> str:
        """Generate response using Groq API with proper error handling."""
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def tavily_search(self, query: str, search_depth: str = "basic") -> Dict[str, Any]:
        """Perform a web search using Tavily API.
        
        Args:
            query: The search query
            search_depth: The depth of search ("basic" or "deep")
            
        Returns:
            Dictionary containing search results
        """
        if not self.tavily_api_key:
            raise HTTPException(
                status_code=500,
                detail="Tavily API key not configured"
            )
            
        try:
            response = self.tavily_client.search(query=query, search_depth=search_depth)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Tavily search failed: {str(e)}"
            )
    
    def tavily_extract(self, url: str) -> Dict[str, Any]:
        """Extract content from a URL using Tavily API.
        
        Args:
            url: The URL to extract content from
            
        Returns:
            Dictionary containing extracted content
        """
        if not self.tavily_api_key:
            raise HTTPException(
                status_code=500,
                detail="Tavily API key not configured"
            )
            
        try:
            response = self.tavily_client.extract(url=url)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Tavily extraction failed: {str(e)}"
            )
    
    def tavily_crawl(self, url: str, max_pages: int = 10) -> Dict[str, Any]:
        """Crawl a website using Tavily API.
        
        Args:
            url: The URL to crawl
            max_pages: Maximum number of pages to crawl
            
        Returns:
            Dictionary containing crawled content
        """
        if not self.tavily_api_key:
            raise HTTPException(
                status_code=500,
                detail="Tavily API key not configured"
            )
            
        try:
            response = self.tavily_client.crawl(url=url, max_pages=max_pages)
            return response
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Tavily crawl failed: {str(e)}"
            )

    def _analyze_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text, structure, and tables from PDF."""
        if not HAS_PDF_SUPPORT:
            raise HTTPException(
                status_code=500,
                detail="PDF processing dependencies not installed"
            )
            
        try:
            # Get metadata with PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                metadata = {
                    "pages": len(pdf.pages),
                    "info": {k: str(v) for k, v in pdf.metadata.items()},
                    "size": os.path.getsize(file_path)
                }
            
            # Extract content with PyMuPDF
            doc = fitz.open(file_path)
            plain_text = ""
            markdown_text = ""
            images = 0
            tables = []
            
            for page in doc:
                # Text with layout
                blocks = page.get_text("dict")["blocks"]
                page_md = f"## Page {page.number + 1}\n\n"
                
                for block in blocks:
                    if block["type"] == 0:  # Text
                        for line in block["lines"]:
                            line_text = " ".join(span["text"] for span in line["spans"])
                            # Detect headings by font size
                            if block.get("size", 0) > 14:
                                page_md += f"### {line_text}\n"
                            else:
                                page_md += f"{line_text}\n"
                    elif block["type"] == 1:  # Image
                        images += 1
                        page_md += f"![Image {images}]\n"
                
                plain_text += page.get_text() + "\n\n"
                markdown_text += page_md + "\n"
                
                # Extract tables if available
                if HAS_TABLE_SUPPORT:
                    try:
                        page_tables = camelot.read_pdf(
                            file_path, 
                            pages=str(page.number + 1),
                            flavor="stream"
                        )
                        for table in page_tables:
                            tables.append({
                                "page": page.number + 1,
                                "data": table.df.to_dict("records"),
                                "accuracy": table.accuracy
                            })
                    except Exception:
                        pass
            
            return {
                "file_type": "pdf",
                "metadata": metadata,
                "extracted_text": plain_text.strip(),
                "markdown_text": markdown_text.strip(),
                "images": images,
                "tables": tables[:5]  # Limit to 5 tables
            }
            
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"PDF analysis failed: {str(e)}"
            )

    def _analyze_text(self, content: bytes, extension: str) -> Dict[str, Any]:
        """Analyze text-based files with format-specific processing."""
        try:
            text = content.decode('utf-8')
            cleaned = self._clean_text(text)
            
            if extension == ".csv":
                return self._analyze_csv(text)
            elif extension == ".json":
                return self._analyze_json(text)
            elif extension == ".md":
                return self._analyze_markdown(text)
            else:
                return self._analyze_plain_text(cleaned)
                
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Text analysis failed: {str(e)}"
            )

    def _analyze_csv(self, text: str) -> Dict[str, Any]:
        """Parse and analyze CSV data."""
        reader = csv.reader(StringIO(text))
        rows = list(reader)
        
        if not rows:
            return {
                "file_type": "csv",
                "metadata": {"rows": 0, "columns": 0},
                "extracted_text": ""
            }
            
        headers = rows[0]
        sample = rows[1:6]
        column_analysis = []
        
        if len(rows) > 1:
            for i, header in enumerate(headers):
                values = []
                for row in rows[1:]:
                    if i < len(row):
                        values.append(row[i])
                column_analysis.append(self._analyze_column(header, values))
        
        return {
            "file_type": "csv",
            "metadata": {
                "rows": len(rows),
                "columns": len(headers),
                "headers": headers,
                "sample": sample,
                "column_analysis": column_analysis
            },
            "extracted_text": "\n".join([",".join(row) for row in rows])
        }

    def _analyze_column(self, header: str, values: List[str]) -> Dict[str, Any]:
        """Analyze a single column of data."""
        if not values:
            return {"header": header, "type": "empty"}
            
        # Check for numeric values
        numeric = []
        for val in values:
            try:
                if val.strip():
                    numeric.append(float(val))
            except ValueError:
                pass
                
        if numeric:
            stats = {
                "min": min(numeric),
                "max": max(numeric),
                "mean": mean(numeric),
                "median": median(numeric)
            }
            if len(numeric) > 1:
                stats["stdev"] = stdev(numeric)
                
            return {
                "header": header,
                "type": "numeric",
                "stats": stats,
                "empty": len(numeric) / len(values)
            }
        
        # Check for dates
        date_count = sum(
            1 for val in values 
            if re.match(r'\d{4}-\d{2}-\d{2}', val.strip())
        )
        if date_count / len(values) > 0.8:
            return {
                "header": header,
                "type": "date",
                "empty": 1 - (date_count / len(values))
            }
        
        # Text analysis
        counts = Counter(values)
        unique = len(counts)
        common = counts.most_common(5)
        
        return {
            "header": header,
            "type": "text",
            "unique_values": unique,
            "most_common": common,
            "empty": sum(1 for v in values if not v.strip()) / len(values)
        }

    def _analyze_markdown(self, text: str) -> Dict[str, Any]:
        """Analyze Markdown document structure."""
        html = markdown.markdown(text)
        headings = re.findall(r'<h([1-6])>(.*?)</h\1>', html)
        code_blocks = len(re.findall(r'```.*?```', text, re.DOTALL))
        
        return {
            "file_type": "markdown",
            "metadata": {
                "headings": [{"level": int(l), "text": t} for l, t in headings],
                "code_blocks": code_blocks,
                "length": len(text)
            },
            "extracted_text": text
        }

    def _analyze_json(self, text: str) -> Dict[str, Any]:
        """Validate and analyze JSON structure."""
        try:
            data = json.loads(text)
            return {
                "file_type": "json",
                "metadata": {
                    "valid": True,
                    "structure": self._analyze_json_structure(data)
                },
                "extracted_text": text
            }
        except json.JSONDecodeError:
            return {
                "file_type": "json",
                "metadata": {"valid": False},
                "extracted_text": text
            }

    def _analyze_json_structure(self, data: Any, depth: int = 3) -> Dict[str, Any]:
        """Recursively analyze JSON structure."""
        if depth <= 0:
            return {"type": "max_depth"}
            
        if isinstance(data, dict):
            sample = {k: type(v).__name__ for k, v in list(data.items())[:5]}
            return {
                "type": "object",
                "keys": len(data),
                "sample": sample
            }
        elif isinstance(data, list):
            if not data:
                return {"type": "empty_array"}
                
            sample_types = Counter(type(x).__name__ for x in data[:100])
            return {
                "type": "array",
                "length": len(data),
                "content_types": dict(sample_types)
            }
        else:
            return {
                "type": type(data).__name__,
                "value": str(data)[:100]
            }

    def _analyze_plain_text(self, text: str) -> Dict[str, Any]:
        """Analyze generic text file."""
        words = re.findall(r'\w+', text)
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "file_type": "text",
            "metadata": {
                "length": len(text),
                "words": len(words),
                "sentences": len(sentences),
                "avg_word_length": mean(len(w) for w in words) if words else 0,
                "avg_sentence_length": mean(len(s.split()) for s in sentences if s.strip()) if sentences else 0
            },
            "extracted_text": text
        }

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace and non-printable chars."""
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _get_ai_analysis(self, text: str, filename: str) -> str:
        """Generate analysis of document content."""
        # Truncate if needed
        if len(text) > 4000:
            half = 2000
            text = f"{text[:half]}\n\n... [content truncated] ...\n\n{text[-half:]}"
        
        ext = os.path.splitext(filename)[1].lower()
        prompt = self._create_analysis_prompt(text, ext)
        return self._generate_ai_response(prompt)

    def _create_analysis_prompt(self, text: str, ext: str) -> str:
        """Create analysis prompt based on file type."""
        base = "Analyze this content and provide:\n1. Key themes/topics\n2. Important details\n3. Overall purpose\n\n"
        
        if ext == ".pdf":
            return base + "Focus on document structure and key information:\n\n" + text
        elif ext == ".csv":
            return base + "Focus on data patterns and potential insights:\n\n" + text
        elif ext == ".json":
            return "Analyze this JSON data's structure and content:\n1. Data schema\n2. Key entities\n3. Relationships\n\n" + text
        else:
            return base + text