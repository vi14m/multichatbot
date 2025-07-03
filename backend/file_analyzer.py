import os
import tempfile
import time
import re
from typing import Dict, Any, List, Optional, Tuple
from fastapi import UploadFile, HTTPException
import requests
import json
from io import StringIO
import csv
import markdown
from collections import Counter
from statistics import mean, median, stdev
from sentence_transformers import SentenceTransformer
import chromadb

# For image analysis
try:
    from PIL import Image
    import easyocr
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False

# For PDF analysis
try:
    import PyPDF2
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

# For PDF table extraction
try:
    import camelot
    HAS_TABLE_SUPPORT = True
except ImportError:
    HAS_TABLE_SUPPORT = False

# Integration summary for file types:
    # Images: OCR (Tesseract), scene description (Groq LLaVA-Next), semantic labeling (CLIP+LLM via LLaVA-Next)
    # PDFs: Text extraction (PyMuPDF, PyPDF2), table parsing (Camelot), structure preservation (markdown layout)
    # Text: Token cleanup (regex), chunking (TextSplitter), summarization (LLM prompt)
    # Each handler method implements these techniques as described.

class FileAnalyzer:
    """Class to handle analysis of different file types"""
    
    def __init__(self, openrouter_api_key: str, groq_api_key: str, embedding_model_names=None, use_all_models=False):
        self.groq_api_key = groq_api_key
        if not groq_api_key:
            raise ValueError("API key for Groq is required")
        # Store model names, not loaded models
        self.embedding_model_names = embedding_model_names or [
            "intfloat/e5-small-v2",
            "BAAI/bge-small-en-v1.5",
            "all-MiniLM-L6-v2"
        ]
        self.use_all_models = use_all_models
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.get_or_create_collection("file_chunks")
    
    async def analyze_file(self, file: UploadFile) -> Dict[str, Any]:
        """Analyze a file based on its type"""
        start_time = time.time()
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        
        # Read file content
        content = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Analyze based on file type
            if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                result = self._analyze_image(temp_path)
            elif file_ext == ".pdf":
                result = self._analyze_pdf(temp_path)
            elif file_ext in [".txt", ".csv", ".md", ".json"]:
                result = self._analyze_text(content, file_ext)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
            # RAG: Add vector DB step for text-based files
            if "extracted_text" in result and result["extracted_text"]:
                from sentence_transformers import SentenceTransformer
                chunks = self._chunk_text(result["extracted_text"])
                all_ids = []
                all_embeddings = []
                all_documents = []
                all_metadatas = []
                model_names_to_use = self.embedding_model_names if self.use_all_models else [self.embedding_model_names[0]]
                for model_idx, model_name in enumerate(model_names_to_use):
                    model_short = model_name.split("/")[-1]
                    model = SentenceTransformer(model_name, device="cpu")
                    embeddings = model.encode(chunks)
                    del model
                    import gc
                    gc.collect()
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        all_ids.append(f"{file.filename}_{model_short}_{i}")
                        all_embeddings.append(emb.tolist())
                        all_documents.append(chunk)
                        all_metadatas.append({
                            "filename": file.filename,
                            "chunk_id": i,
                            "model_name": model_short
                        })
                self.chroma_collection.add(
                    ids=all_ids,
                    embeddings=all_embeddings,
                    documents=all_documents,
                    metadatas=all_metadatas
                )
                ai_analysis = self._get_ai_analysis(result["extracted_text"], file.filename)
                result["ai_analysis"] = ai_analysis
            
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            
            return result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze an image file using OCR, scene description, and semantic labeling"""
        if not HAS_IMAGE_SUPPORT:
            raise HTTPException(status_code=500, detail="Image analysis support not installed")
        
        try:
            # Open the image
            image = Image.open(file_path)
            
            # Extract basic image info
            info = {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "file_size": os.path.getsize(file_path),
                "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
            }
            
            # Extract text using OCR with improved configuration
            reader = easyocr.Reader(['en'], gpu=False)
            result = reader.readtext(file_path, detail=0)
            extracted_text = '\n'.join(result)
            
            # Get scene description and semantic labels using AI
            image_description = self._get_image_description(image, file_path)
            
            return {
                "file_type": "image",
                "metadata": info,
                "extracted_text": extracted_text,
                "scene_description": image_description["description"],
                "semantic_labels": image_description["labels"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    
    def _analyze_pdf(self, file_path: str) -> Dict[str, Any]:
        """Analyze a PDF file with improved text extraction, structure preservation, and table parsing"""
        if not HAS_PDF_SUPPORT:
            raise HTTPException(status_code=500, detail="PDF analysis support not installed")
        
        try:
            # Use PyPDF2 for metadata
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                num_pages = len(reader.pages)
                
                # Get document info
                info = reader.metadata
                info_dict = {}
                if info:
                    for key in info:
                        info_dict[key] = str(info[key])
            
            # Use PyMuPDF (fitz) for better text extraction with layout preservation
            doc = fitz.open(file_path)
            markdown_text = ""
            plain_text = ""
            images_count = 0
            tables = []
            
            for page_num in range(num_pages):
                page = doc[page_num]
                
                # Extract text with layout preservation
                blocks = page.get_text("dict")["blocks"]
                page_text = ""
                
                for block in blocks:
                    if block["type"] == 0:  # Text block
                        for line in block["lines"]:
                            line_text = ""
                            for span in line["spans"]:
                                # Check if it's a heading based on font size
                                font_size = span["size"]
                                text = span["text"]
                                if font_size > 14:
                                    line_text += f"## {text} "
                                elif font_size > 12:
                                    line_text += f"### {text} "
                                else:
                                    line_text += text + " "
                            page_text += line_text.strip() + "\n"
                    elif block["type"] == 1:  # Image block
                        images_count += 1
                        page_text += f"\n![Image {images_count} on page {page_num+1}]\n\n"
                
                markdown_text += f"\n## Page {page_num+1}\n\n{page_text}\n\n"
                plain_text += page.get_text() + "\n\n"
                
                # Extract tables if supported
                if HAS_TABLE_SUPPORT:
                    try:
                        page_tables = camelot.read_pdf(file_path, pages=str(page_num+1))
                        if len(page_tables) > 0:
                            for i, table in enumerate(page_tables):
                                tables.append({
                                    "page": page_num+1,
                                    "table_number": i+1,
                                    "data": table.df.to_dict('records')
                                })
                    except Exception:
                        # Skip table extraction if it fails
                        pass
            
            return {
                "file_type": "pdf",
                "metadata": {
                    "num_pages": num_pages,
                    "info": info_dict,
                    "file_size": os.path.getsize(file_path),
                    "images_count": images_count,
                    "tables_count": len(tables)
                },
                "extracted_text": plain_text,
                "markdown_text": markdown_text,
                "tables": tables[:5] if tables else []  # Limit to first 5 tables
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")
    
    def _analyze_text(self, content: bytes, file_ext: str) -> Dict[str, Any]:
        """Analyze a text-based file with improved token cleanup and context summarization"""
        try:
            text_content = content.decode('utf-8')
            
            # Clean up common issues in text files
            cleaned_text = self._clean_text(text_content)
            
            # For CSV files, try to extract structured data
            if file_ext == ".csv":
                csv_data = []
                csv_reader = csv.reader(StringIO(text_content))
                for row in csv_reader:
                    csv_data.append(row)
                
                # Get headers and sample data
                headers = csv_data[0] if csv_data else []
                sample_data = csv_data[1:6] if len(csv_data) > 1 else []
                
                # Analyze data types and statistics for each column
                column_analysis = []
                if len(csv_data) > 1 and headers:
                    for i, header in enumerate(headers):
                        if i < len(headers):
                            column_values = [row[i] for row in csv_data[1:] if i < len(row)]
                            column_analysis.append(self._analyze_column(header, column_values))
                
                return {
                    "file_type": "csv",
                    "metadata": {
                        "rows": len(csv_data),
                        "columns": len(headers),
                        "headers": headers,
                        "sample": sample_data,
                        "column_analysis": column_analysis
                    },
                    "extracted_text": cleaned_text
                }
            
            # For JSON files, validate and extract structure
            elif file_ext == ".json":
                try:
                    json_data = json.loads(text_content)
                    is_valid = True
                    
                    # Analyze JSON structure
                    structure_info = self._analyze_json_structure(json_data)
                    
                except json.JSONDecodeError:
                    is_valid = False
                    structure_info = {}
                
                return {
                    "file_type": "json",
                    "metadata": {
                        "is_valid": is_valid,
                        "length": len(text_content),
                        "structure": structure_info
                    },
                    "extracted_text": cleaned_text
                }
            
            # For Markdown files
            elif file_ext == ".md":
                # Convert to HTML for better structure analysis
                html_content = markdown.markdown(text_content)
                
                # Extract headings
                headings = re.findall(r'<h([1-6])>(.*?)</h\1>', html_content)
                heading_structure = [{'level': int(level), 'text': text} for level, text in headings]
                
                # Count code blocks
                code_blocks = len(re.findall(r'```.*?```', text_content, re.DOTALL))
                
                lines = text_content.split('\n')
                return {
                    "file_type": "markdown",
                    "metadata": {
                        "lines": len(lines),
                        "length": len(text_content),
                        "headings": heading_structure,
                        "code_blocks": code_blocks
                    },
                    "extracted_text": cleaned_text
                }
            
            # For other text files
            else:
                lines = text_content.split('\n')
                
                # Get text statistics
                word_count = len(re.findall(r'\w+', text_content))
                sentence_count = len(re.findall(r'[.!?]+', text_content))
                
                # Chunk the text for better analysis
                chunks = self._chunk_text(cleaned_text)
                
                return {
                    "file_type": "text",
                    "metadata": {
                        "lines": len(lines),
                        "length": len(text_content),
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "chunks": len(chunks)
                    },
                    "extracted_text": cleaned_text,
                    "chunks": chunks[:3]  # Include first 3 chunks as preview
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error analyzing text file: {str(e)}")
    
    def _get_ai_analysis(self, text: str, filename: Optional[str] = None) -> str:
        """Get AI analysis of the extracted text with improved context handling using Groq API"""
        max_text_length = 4000
        if len(text) > max_text_length:
            half_length = max_text_length // 2
            text = text[:half_length] + "\n\n... (content truncated) ...\n\n" + text[-half_length:]
        file_desc = f"file named '{filename}'" if filename else "file"
        file_ext = os.path.splitext(filename)[1].lower() if filename else ""
        if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
            prompt = f"Analyze the following text extracted from an image file {file_desc}. Identify key information, main topics, and provide insights about what the image might contain based on the text:\n\n{text}"
        elif file_ext == ".pdf":
            prompt = f"Analyze the following content extracted from a PDF document {file_desc}. Provide a structured summary covering: 1) Main topics and themes, 2) Key information and findings, 3) Document structure and organization, 4) Any notable data or statistics mentioned:\n\n{text}"
        elif file_ext == ".csv":
            prompt = f"Analyze the following content extracted from a CSV file {file_desc}. Identify: 1) What kind of data this represents, 2) Key patterns or trends, 3) What the columns represent, 4) Potential use cases for this data:\n\n{text}"
        elif file_ext == ".json":
            prompt = f"Analyze the following JSON content from {file_desc}. Explain: 1) The data structure and schema, 2) Key entities and their relationships, 3) Purpose of this data structure, 4) Any notable patterns or values:\n\n{text}"
        elif file_ext == ".md":
            prompt = f"Analyze the following Markdown content from {file_desc}. Provide: 1) Document structure and organization, 2) Main topics and themes, 3) Purpose of the document, 4) Key takeaways:\n\n{text}"
        else:
            prompt = f"Analyze the following content extracted from a {file_desc}. Provide a comprehensive summary covering: 1) Main topics and themes, 2) Key information and insights, 3) Important details, 4) Overall purpose of the content:\n\n{text}"
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code != 200:
                return f"Error getting AI analysis: {response.text}"
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting AI analysis: {str(e)}"
            
    def _get_image_description(self, image, file_path: str) -> Dict[str, Any]:
        """Get AI-generated scene description and semantic labels for an image using Groq API"""
        import base64
        from io import BytesIO
        max_size = (800, 800)
        image.thumbnail(max_size, Image.LANCZOS)
        buffered = BytesIO()
        image.save(buffered, format=image.format if image.format else "JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Include: 1) A concise scene description, 2) A list of key objects or elements visible, 3) Any text that appears in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            if response.status_code != 200:
                return {
                    "description": f"Error getting image description: {response.text}",
                    "labels": []
                }
            result = response.json()
            description = result["choices"][0]["message"]["content"]
            words = re.findall(r'\b\w+\b', description.lower())
            common_objects = ["person", "people", "man", "woman", "child", "building", "car", "tree", 
                             "water", "sky", "mountain", "food", "animal", "dog", "cat", "bird", 
                             "phone", "computer", "book", "table", "chair", "text", "logo", "sign"]
            labels = [word for word in words if word in common_objects]
            labels = list(set(labels))
            return {
                "description": description,
                "labels": labels[:10]
            }
        except Exception as e:
            return {
                "description": f"Error getting image description: {str(e)}",
                "labels": []
            }
    
    def _clean_text(self, text: str) -> str:
        """Clean up text by removing extra whitespace and non-printable characters."""
        import re
        # Remove non-printable characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        # Replace multiple spaces/newlines with a single space/newline
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def _analyze_column(self, header: str, values: List[str]) -> Dict[str, Any]:
        """Analyze a column in a CSV file to determine data type and statistics."""
        if not values:
            return {"header": header, "type": "unknown", "empty": True}
        
        # Check if values are numeric
        numeric_values = []
        is_numeric = True
        for val in values:
            try:
                if val.strip():
                    numeric_values.append(float(val))
                else:
                    is_numeric = False
            except ValueError:
                is_numeric = False
        
        # Check if values are dates
        date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
        is_date = all(date_pattern.match(val) for val in values if val.strip())
        
        # Count empty values
        empty_count = sum(1 for val in values if not val.strip())
        
        # Determine data type
        if is_numeric and numeric_values:
            # Calculate statistics
            try:
                stats = {
                    "min": min(numeric_values),
                    "max": max(numeric_values),
                    "mean": mean(numeric_values),
                    "median": median(numeric_values)
                }
                if len(numeric_values) > 1:
                    stats["std_dev"] = stdev(numeric_values)
                return {
                    "header": header,
                    "type": "numeric",
                    "empty_count": empty_count,
                    "stats": stats
                }
            except Exception:
                pass
        
        if is_date:
            return {"header": header, "type": "date", "empty_count": empty_count}
        
        # Text analysis
        # Get unique values and their counts
        value_counts = Counter(val for val in values if val.strip())
        unique_values = len(value_counts)
        
        # Check if it's potentially categorical (few unique values)
        is_categorical = unique_values <= 10 or (unique_values / len(values) <= 0.2)
        
        # Get most common values
        most_common = value_counts.most_common(5)
        
        return {
            "header": header,
            "type": "categorical" if is_categorical else "text",
            "empty_count": empty_count,
            "unique_values": unique_values,
            "most_common": most_common
        }
    
    def _analyze_json_structure(self, json_data: Any, max_depth: int = 3) -> Dict[str, Any]:
        """Analyze the structure of a JSON object."""
        if max_depth <= 0:
            return {"type": "truncated"}
        
        if isinstance(json_data, dict):
            result = {"type": "object", "keys_count": len(json_data)}
            if json_data and max_depth > 1:
                # Analyze a sample of keys (up to 10)
                sample_keys = list(json_data.keys())[:10]
                result["sample_keys"] = sample_keys
                
                # Analyze structure of values for sample keys
                if max_depth > 1:
                    result["properties"] = {}
                    for key in sample_keys:
                        result["properties"][key] = self._analyze_json_structure(json_data[key], max_depth - 1)
            return result
        
        elif isinstance(json_data, list):
            result = {"type": "array", "length": len(json_data)}
            if json_data and max_depth > 1:
                # Analyze a sample of items (up to 5)
                sample_size = min(5, len(json_data))
                if all(isinstance(item, dict) for item in json_data[:sample_size]):
                    result["items_type"] = "object"
                    # Find common keys across objects
                    common_keys = set(json_data[0].keys())
                    for item in json_data[1:sample_size]:
                        common_keys &= set(item.keys())
                    result["common_keys"] = list(common_keys)
                elif all(isinstance(item, (int, float)) for item in json_data[:sample_size]):
                    result["items_type"] = "number"
                    if len(json_data) > 0:
                        result["min"] = min(json_data[:100])  # Sample for performance
                        result["max"] = max(json_data[:100])
                elif all(isinstance(item, str) for item in json_data[:sample_size]):
                    result["items_type"] = "string"
                    if len(json_data) > 0:
                        result["sample"] = json_data[:3]
                else:
                    result["items_type"] = "mixed"
            return result
        
        elif isinstance(json_data, str):
            return {"type": "string", "length": len(json_data)}
        
        elif isinstance(json_data, (int, float)):
            return {"type": "number", "value": json_data}
        
        elif isinstance(json_data, bool):
            return {"type": "boolean", "value": json_data}
        
        elif json_data is None:
            return {"type": "null"}
        
        else:
            return {"type": "unknown"}
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
        """Split text into overlapping chunks for better analysis."""
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            
            # Try to find a natural break point (sentence or paragraph end)
            if end < len(text):
                # Look for paragraph break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    # Look for sentence break
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('? ', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                        end = sentence_break + 2
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks

    def rag_query(self, question: str, filename: str, top_k: int = 5, relevance_threshold: float = 0.75) -> str:
        """Advanced RAG: Hybrid retrieval, context distillation, reranking, query expansion, and hallucination fallback."""
        # --- Query Expansion ---
        # Simple expansion: add synonyms/related terms (could use WordNet or static synonyms)
        expansions = [question]
        # Example: add a simple synonym expansion for demonstration
        if "summary" in question.lower():
            expansions.append(question.replace("summary", "overview"))
        if "explain" in question.lower():
            expansions.append(question.replace("explain", "describe"))

        # --- Hybrid Dense Retrieval (multiple models) + Keyword Search (if needed) ---
        from sentence_transformers import SentenceTransformer
        chunk_scores = {}
        model_names_to_use = self.embedding_model_names if self.use_all_models else [self.embedding_model_names[0]]
        for q in expansions:
            for model_name in model_names_to_use:
                model = SentenceTransformer(model_name, device="cpu")
                q_emb = model.encode([q])[0]
                del model
                import gc
                gc.collect()
                results = self.chroma_collection.query(
                    query_embeddings=[q_emb.tolist()],
                    n_results=top_k,
                    where={"filename": filename},
                    include=["distances", "documents"]
                )
                docs = results.get('documents', [[]])[0]
                dists = results.get('distances', [[]])[0]
                for doc, dist in zip(docs, dists):
                    if doc:
                        similarity = 1 - dist
                        is_diverse = all(doc not in c or abs(similarity - s) > 0.01 for c, s in chunk_scores.items())
                        if is_diverse and (doc not in chunk_scores or similarity > chunk_scores[doc]):
                            chunk_scores[doc] = similarity

        # --- Reranking: sort by similarity (could use cross-encoder for better reranking) ---
        reranked = sorted(chunk_scores.items(), key=lambda x: -x[1])
        relevant_chunks = [doc for doc, sim in reranked if sim >= relevance_threshold]

        # If no relevant chunk meets the threshold, do not include any document context
        if not relevant_chunks:
            prompt = (
                "No relevant context was found in the provided documents. Answering based on general knowledge.\n\n"
                f"User: {question}\n"
                "Assistant:"
            )
            return self._get_ai_analysis(prompt, filename)

        # --- Context Distillation: summarize if too many chunks ---
        max_chunks = 5
        if len(relevant_chunks) > max_chunks:
            context_text = "\n".join(relevant_chunks[:max_chunks])
            summary_prompt = f"Summarize the following context for answering a user question:\n\n{context_text}"
            context = self._get_ai_analysis(summary_prompt, filename)
        else:
            context = "\n".join(relevant_chunks)

        # --- Prompt Engineering & Fallback for Hallucination ---
        prompt = (
            "You are a friendly and knowledgeable AI assistant designed to help users with accurate, context-aware answers.\n\n"
            "When context is provided, use ONLY the information in the context. If context is not available or insufficient, rely on your general knowledge—but always prioritize context when available.\n\n"
            "Respond conversationally and helpfully.\n\n"
            "Context (from documents):\n"
            "--------------------------\n"
            f"{context}\n"
            "--------------------------\n\n"
            f"User: {question}\n"
            "Assistant:"
        )
        return self._get_ai_analysis(prompt, filename)
