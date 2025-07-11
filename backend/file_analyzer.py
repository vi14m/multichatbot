# import os
# import tempfile
# import time
# import re
# from typing import Dict, Any, List, Optional, Tuple
# from fastapi import UploadFile, HTTPException
# import requests
# import json
# from io import StringIO
# import csv
# import markdown
# from collections import Counter
# from statistics import mean, median, stdev
# import chromadb

# # For image analysis
# try:
#     from PIL import Image
#     import easyocr
#     HAS_IMAGE_SUPPORT = True
# except ImportError:
#     HAS_IMAGE_SUPPORT = False

# # For PDF analysis
# try:
#     import PyPDF2
#     import fitz  # PyMuPDF
#     HAS_PDF_SUPPORT = True
# except ImportError:
#     HAS_PDF_SUPPORT = False

# # For PDF table extraction
# try:
#     import camelot
#     HAS_TABLE_SUPPORT = True
# except ImportError:
#     HAS_TABLE_SUPPORT = False

# # Integration summary for file types:
#     # Images: OCR (Tesseract), scene description (Groq LLaVA-Next), semantic labeling (CLIP+LLM via LLaVA-Next)
#     # PDFs: Text extraction (PyMuPDF, PyPDF2), table parsing (Camelot), structure preservation (markdown layout)
#     # Text: Token cleanup (regex), chunking (TextSplitter), summarization (LLM prompt)
#     # Each handler method implements these techniques as described.

# class FileAnalyzer:
#     """Class to handle analysis of different file types"""
    
#     def __init__(self, openrouter_api_key: str, groq_api_key: str, embedding_model_names=None, use_all_models=False):
#         self.groq_api_key = groq_api_key
#         if not groq_api_key:
#             raise ValueError("API key for Groq is required")
#         # Store model names, not loaded models
#         # Use local model paths if not provided
#         # Use relative path for Docker and local compatibility
#         self.embedding_model_names = embedding_model_names or [
#             "intfloat/e5-small-v2",
#             "BAAI/bge-small-en-v1.5",
#             "sentence-transformers/all-MiniLM-L6-v2"
#         ]
#         self.use_all_models = use_all_models
#         self.chroma_client = chromadb.Client()
#         self.chroma_collection = self.chroma_client.get_or_create_collection("file_chunks")
    
#     async def analyze_file(self, file: UploadFile) -> Dict[str, Any]:
#         """Analyze a file based on its type"""
#         start_time = time.time()
        
#         # Get file extension
#         file_ext = os.path.splitext(file.filename)[1].lower() if file.filename else ""
        
#         # Read file content
#         content = await file.read()
        
#         # Create a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
#             temp_file.write(content)
#             temp_path = temp_file.name
        
#         try:
#             # Analyze based on file type
#             if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
#                 result = self._analyze_image(temp_path)
#             elif file_ext == ".pdf":
#                 result = self._analyze_pdf(temp_path)
#             elif file_ext in [".txt", ".csv", ".md", ".json"]:
#                 result = self._analyze_text(content, file_ext)
#             else:
#                 raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
#             # RAG: Add vector DB step for text-based files
#             if "extracted_text" in result and result["extracted_text"]:
#                 chunks = self._chunk_text(result["extracted_text"])
#                 all_ids = []
#                 all_embeddings = []
#                 all_documents = []
#                 all_metadatas = []
#                 model_names_to_use = self.embedding_model_names if self.use_all_models else [self.embedding_model_names[0]]
#                 for model_idx, model_name in enumerate(model_names_to_use):
#                     model_short = model_name.split("/")[-1]
#                     embeddings = self._get_hf_embeddings(model_name, chunks)
#                     for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
#                         all_ids.append(f"{file.filename}_{model_short}_{i}")
#                         all_embeddings.append(emb)
#                         all_documents.append(chunk)
#                         all_metadatas.append({
#                             "filename": file.filename,
#                             "chunk_id": i,
#                             "model_name": model_short
#                         })
#                 self.chroma_collection.add(
#                     ids=all_ids,
#                     embeddings=all_embeddings,
#                     documents=all_documents,
#                     metadatas=all_metadatas
#                 )
#                 ai_analysis = self._get_ai_analysis(result["extracted_text"], file.filename)
#                 result["ai_analysis"] = ai_analysis
            
#             processing_time = time.time() - start_time
#             result["processing_time"] = processing_time
            
#             return result
#         finally:
#             # Clean up the temporary file
#             if os.path.exists(temp_path):
#                 os.unlink(temp_path)
    
#     def _analyze_image(self, file_path: str) -> Dict[str, Any]:
#         """Analyze an image file using OCR, scene description, and semantic labeling"""
#         if not HAS_IMAGE_SUPPORT:
#             raise HTTPException(status_code=500, detail="Image analysis support not installed")
        
#         try:
#             # Open the image
#             image = Image.open(file_path)
            
#             # Extract basic image info
#             info = {
#                 "format": image.format,
#                 "mode": image.mode,
#                 "width": image.width,
#                 "height": image.height,
#                 "file_size": os.path.getsize(file_path),
#                 "aspect_ratio": round(image.width / image.height, 2) if image.height > 0 else 0
#             }
            
#             # Extract text using OCR with improved configuration
#             reader = easyocr.Reader(['en'], gpu=False)
#             result = reader.readtext(file_path, detail=0)
#             extracted_text = '\n'.join(result)
            
#             # Get scene description and semantic labels using AI
#             image_description = self._get_image_description(image, file_path)
            
#             return {
#                 "file_type": "image",
#                 "metadata": info,
#                 "extracted_text": extracted_text,
#                 "scene_description": image_description["description"],
#                 "semantic_labels": image_description["labels"]
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")
    
#     def _analyze_pdf(self, file_path: str) -> Dict[str, Any]:
#         """Analyze a PDF file with improved text extraction, structure preservation, and table parsing"""
#         if not HAS_PDF_SUPPORT:
#             raise HTTPException(status_code=500, detail="PDF analysis support not installed")
        
#         try:
#             # Use PyPDF2 for metadata
#             with open(file_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 num_pages = len(reader.pages)
                
#                 # Get document info
#                 info = reader.metadata
#                 info_dict = {}
#                 if info:
#                     for key in info:
#                         info_dict[key] = str(info[key])
            
#             # Use PyMuPDF (fitz) for better text extraction with layout preservation
#             doc = fitz.open(file_path)
#             markdown_text = ""
#             plain_text = ""
#             images_count = 0
#             tables = []
            
#             for page_num in range(num_pages):
#                 page = doc[page_num]
                
#                 # Extract text with layout preservation
#                 blocks = page.get_text("dict")["blocks"]
#                 page_text = ""
                
#                 for block in blocks:
#                     if block["type"] == 0:  # Text block
#                         for line in block["lines"]:
#                             line_text = ""
#                             for span in line["spans"]:
#                                 # Check if it's a heading based on font size
#                                 font_size = span["size"]
#                                 text = span["text"]
#                                 if font_size > 14:
#                                     line_text += f"## {text} "
#                                 elif font_size > 12:
#                                     line_text += f"### {text} "
#                                 else:
#                                     line_text += text + " "
#                             page_text += line_text.strip() + "\n"
#                     elif block["type"] == 1:  # Image block
#                         images_count += 1
#                         page_text += f"\n![Image {images_count} on page {page_num+1}]\n\n"
                
#                 markdown_text += f"\n## Page {page_num+1}\n\n{page_text}\n\n"
#                 plain_text += page.get_text() + "\n\n"
                
#                 # Extract tables if supported
#                 if HAS_TABLE_SUPPORT:
#                     try:
#                         page_tables = camelot.read_pdf(file_path, pages=str(page_num+1))
#                         if len(page_tables) > 0:
#                             for i, table in enumerate(page_tables):
#                                 tables.append({
#                                     "page": page_num+1,
#                                     "table_number": i+1,
#                                     "data": table.df.to_dict('records')
#                                 })
#                     except Exception:
#                         # Skip table extraction if it fails
#                         pass
            
#             return {
#                 "file_type": "pdf",
#                 "metadata": {
#                     "num_pages": num_pages,
#                     "info": info_dict,
#                     "file_size": os.path.getsize(file_path),
#                     "images_count": images_count,
#                     "tables_count": len(tables)
#                 },
#                 "extracted_text": plain_text,
#                 "markdown_text": markdown_text,
#                 "tables": tables[:5] if tables else []  # Limit to first 5 tables
#             }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error analyzing PDF: {str(e)}")
    
#     def _analyze_text(self, content: bytes, file_ext: str) -> Dict[str, Any]:
#         """Analyze a text-based file with improved token cleanup and context summarization"""
#         try:
#             text_content = content.decode('utf-8')
            
#             # Clean up common issues in text files
#             cleaned_text = self._clean_text(text_content)
            
#             # For CSV files, try to extract structured data
#             if file_ext == ".csv":
#                 csv_data = []
#                 csv_reader = csv.reader(StringIO(text_content))
#                 for row in csv_reader:
#                     csv_data.append(row)
                
#                 # Get headers and sample data
#                 headers = csv_data[0] if csv_data else []
#                 sample_data = csv_data[1:6] if len(csv_data) > 1 else []
                
#                 # Analyze data types and statistics for each column
#                 column_analysis = []
#                 if len(csv_data) > 1 and headers:
#                     for i, header in enumerate(headers):
#                         if i < len(headers):
#                             column_values = [row[i] for row in csv_data[1:] if i < len(row)]
#                             column_analysis.append(self._analyze_column(header, column_values))
                
#                 return {
#                     "file_type": "csv",
#                     "metadata": {
#                         "rows": len(csv_data),
#                         "columns": len(headers),
#                         "headers": headers,
#                         "sample": sample_data,
#                         "column_analysis": column_analysis
#                     },
#                     "extracted_text": cleaned_text
#                 }
            
#             # For JSON files, validate and extract structure
#             elif file_ext == ".json":
#                 try:
#                     json_data = json.loads(text_content)
#                     is_valid = True
                    
#                     # Analyze JSON structure
#                     structure_info = self._analyze_json_structure(json_data)
                    
#                 except json.JSONDecodeError:
#                     is_valid = False
#                     structure_info = {}
                
#                 return {
#                     "file_type": "json",
#                     "metadata": {
#                         "is_valid": is_valid,
#                         "length": len(text_content),
#                         "structure": structure_info
#                     },
#                     "extracted_text": cleaned_text
#                 }
            
#             # For Markdown files
#             elif file_ext == ".md":
#                 # Convert to HTML for better structure analysis
#                 html_content = markdown.markdown(text_content)
                
#                 # Extract headings
#                 headings = re.findall(r'<h([1-6])>(.*?)</h\1>', html_content)
#                 heading_structure = [{'level': int(level), 'text': text} for level, text in headings]
                
#                 # Count code blocks
#                 code_blocks = len(re.findall(r'```.*?```', text_content, re.DOTALL))
                
#                 lines = text_content.split('\n')
#                 return {
#                     "file_type": "markdown",
#                     "metadata": {
#                         "lines": len(lines),
#                         "length": len(text_content),
#                         "headings": heading_structure,
#                         "code_blocks": code_blocks
#                     },
#                     "extracted_text": cleaned_text
#                 }
            
#             # For other text files
#             else:
#                 lines = text_content.split('\n')
                
#                 # Get text statistics
#                 word_count = len(re.findall(r'\w+', text_content))
#                 sentence_count = len(re.findall(r'[.!?]+', text_content))
                
#                 # Chunk the text for better analysis
#                 chunks = self._chunk_text(cleaned_text)
                
#                 return {
#                     "file_type": "text",
#                     "metadata": {
#                         "lines": len(lines),
#                         "length": len(text_content),
#                         "word_count": word_count,
#                         "sentence_count": sentence_count,
#                         "chunks": len(chunks)
#                     },
#                     "extracted_text": cleaned_text,
#                     "chunks": chunks[:3]  # Include first 3 chunks as preview
#                 }
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error analyzing text file: {str(e)}")
    
#     def _get_ai_analysis(self, text: str, filename: Optional[str] = None) -> str:
#         """Get AI analysis of the extracted text with improved context handling using Groq API"""
#         max_text_length = 4000
#         if len(text) > max_text_length:
#             half_length = max_text_length // 2
#             text = text[:half_length] + "\n\n... (content truncated) ...\n\n" + text[-half_length:]
#         file_desc = f"file named '{filename}'" if filename else "file"
#         file_ext = os.path.splitext(filename)[1].lower() if filename else ""
#         if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
#             prompt = f"Analyze the following text extracted from an image file {file_desc}. Identify key information, main topics, and provide insights about what the image might contain based on the text:\n\n{text}"
#         elif file_ext == ".pdf":
#             prompt = f"Analyze the following content extracted from a PDF document {file_desc}. Provide a structured summary covering: 1) Main topics and themes, 2) Key information and findings, 3) Document structure and organization, 4) Any notable data or statistics mentioned:\n\n{text}"
#         elif file_ext == ".csv":
#             prompt = f"Analyze the following content extracted from a CSV file {file_desc}. Identify: 1) What kind of data this represents, 2) Key patterns or trends, 3) What the columns represent, 4) Potential use cases for this data:\n\n{text}"
#         elif file_ext == ".json":
#             prompt = f"Analyze the following JSON content from {file_desc}. Explain: 1) The data structure and schema, 2) Key entities and their relationships, 3) Purpose of this data structure, 4) Any notable patterns or values:\n\n{text}"
#         elif file_ext == ".md":
#             prompt = f"Analyze the following Markdown content from {file_desc}. Provide: 1) Document structure and organization, 2) Main topics and themes, 3) Purpose of the document, 4) Key takeaways:\n\n{text}"
#         else:
#             prompt = f"Analyze the following content extracted from a {file_desc}. Provide a comprehensive summary covering: 1) Main topics and themes, 2) Key information and insights, 3) Important details, 4) Overall purpose of the content:\n\n{text}"
#         headers = {
#             "Authorization": f"Bearer {self.groq_api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": "llama3-70b-8192",
#             "messages": [
#                 {"role": "user", "content": prompt}
#             ],
#             "temperature": 0.3
#         }
#         try:
#             response = requests.post(
#                 "https://api.groq.com/openai/v1/chat/completions",
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
#             if response.status_code != 200:
#                 return f"Error getting AI analysis: {response.text}"
#             result = response.json()
#             return result["choices"][0]["message"]["content"]
#         except Exception as e:
#             return f"Error getting AI analysis: {str(e)}"
            
#     def _get_image_description(self, image, file_path: str) -> Dict[str, Any]:
#         """Get AI-generated scene description and semantic labels for an image using Groq API"""
#         import base64
#         from io import BytesIO
#         max_size = (800, 800)
#         image.thumbnail(max_size, Image.LANCZOS)
#         buffered = BytesIO()
#         image.save(buffered, format=image.format if image.format else "JPEG")
#         img_str = base64.b64encode(buffered.getvalue()).decode()
#         headers = {
#             "Authorization": f"Bearer {self.groq_api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": "meta-llama/llama-4-scout-17b-16e-instruct",
#             "messages": [
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": "Describe this image in detail. Include: 1) A concise scene description, 2) A list of key objects or elements visible, 3) Any text that appears in the image."
#                         },
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/{image.format.lower() if image.format else 'jpeg'};base64,{img_str}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             "temperature": 0.2,
#             "max_tokens": 1024,
#         }
#         try:
#             response = requests.post(
#                 "https://api.groq.com/openai/v1/chat/completions",
#                 headers=headers,
#                 json=data,
#                 timeout=30
#             )
#             if response.status_code != 200:
#                 return {
#                     "description": f"Error getting image description: {response.text}",
#                     "labels": []
#                 }
#             result = response.json()
#             description = result["choices"][0]["message"]["content"]
#             words = re.findall(r'\b\w+\b', description.lower())
#             common_objects = ["person", "people", "man", "woman", "child", "building", "car", "tree", 
#                              "water", "sky", "mountain", "food", "animal", "dog", "cat", "bird", 
#                              "phone", "computer", "book", "table", "chair", "text", "logo", "sign"]
#             labels = [word for word in words if word in common_objects]
#             labels = list(set(labels))
#             return {
#                 "description": description,
#                 "labels": labels[:10]
#             }
#         except Exception as e:
#             return {
#                 "description": f"Error getting image description: {str(e)}",
#                 "labels": []
#             }
    
#     def _clean_text(self, text: str) -> str:
#         """Clean up text by removing extra whitespace and non-printable characters."""
#         import re
#         # Remove non-printable characters
#         text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
#         # Replace multiple spaces/newlines with a single space/newline
#         text = re.sub(r' +', ' ', text)
#         text = re.sub(r'\n+', '\n', text)
#         return text.strip()
    
#     def _analyze_column(self, header: str, values: List[str]) -> Dict[str, Any]:
#         """Analyze a column in a CSV file to determine data type and statistics."""
#         if not values:
#             return {"header": header, "type": "unknown", "empty": True}
        
#         # Check if values are numeric
#         numeric_values = []
#         is_numeric = True
#         for val in values:
#             try:
#                 if val.strip():
#                     numeric_values.append(float(val))
#                 else:
#                     is_numeric = False
#             except ValueError:
#                 is_numeric = False
        
#         # Check if values are dates
#         date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$')
#         is_date = all(date_pattern.match(val) for val in values if val.strip())
        
#         # Count empty values
#         empty_count = sum(1 for val in values if not val.strip())
        
#         # Determine data type
#         if is_numeric and numeric_values:
#             # Calculate statistics
#             try:
#                 stats = {
#                     "min": min(numeric_values),
#                     "max": max(numeric_values),
#                     "mean": mean(numeric_values),
#                     "median": median(numeric_values)
#                 }
#                 if len(numeric_values) > 1:
#                     stats["std_dev"] = stdev(numeric_values)
#                 return {
#                     "header": header,
#                     "type": "numeric",
#                     "empty_count": empty_count,
#                     "stats": stats
#                 }
#             except Exception:
#                 pass
        
#         if is_date:
#             return {"header": header, "type": "date", "empty_count": empty_count}
        
#         # Text analysis
#         # Get unique values and their counts
#         value_counts = Counter(val for val in values if val.strip())
#         unique_values = len(value_counts)
        
#         # Check if it's potentially categorical (few unique values)
#         is_categorical = unique_values <= 10 or (unique_values / len(values) <= 0.2)
        
#         # Get most common values
#         most_common = value_counts.most_common(5)
        
#         return {
#             "header": header,
#             "type": "categorical" if is_categorical else "text",
#             "empty_count": empty_count,
#             "unique_values": unique_values,
#             "most_common": most_common
#         }
    
#     def _analyze_json_structure(self, json_data: Any, max_depth: int = 3) -> Dict[str, Any]:
#         """Analyze the structure of a JSON object."""
#         if max_depth <= 0:
#             return {"type": "truncated"}
        
#         if isinstance(json_data, dict):
#             result = {"type": "object", "keys_count": len(json_data)}
#             if json_data and max_depth > 1:
#                 # Analyze a sample of keys (up to 10)
#                 sample_keys = list(json_data.keys())[:10]
#                 result["sample_keys"] = sample_keys
                
#                 # Analyze structure of values for sample keys
#                 if max_depth > 1:
#                     result["properties"] = {}
#                     for key in sample_keys:
#                         result["properties"][key] = self._analyze_json_structure(json_data[key], max_depth - 1)
#             return result
        
#         elif isinstance(json_data, list):
#             result = {"type": "array", "length": len(json_data)}
#             if json_data and max_depth > 1:
#                 # Analyze a sample of items (up to 5)
#                 sample_size = min(5, len(json_data))
#                 if all(isinstance(item, dict) for item in json_data[:sample_size]):
#                     result["items_type"] = "object"
#                     # Find common keys across objects
#                     common_keys = set(json_data[0].keys())
#                     for item in json_data[1:sample_size]:
#                         common_keys &= set(item.keys())
#                     result["common_keys"] = list(common_keys)
#                 elif all(isinstance(item, (int, float)) for item in json_data[:sample_size]):
#                     result["items_type"] = "number"
#                     if len(json_data) > 0:
#                         result["min"] = min(json_data[:100])  # Sample for performance
#                         result["max"] = max(json_data[:100])
#                 elif all(isinstance(item, str) for item in json_data[:sample_size]):
#                     result["items_type"] = "string"
#                     if len(json_data) > 0:
#                         result["sample"] = json_data[:3]
#                 else:
#                     result["items_type"] = "mixed"
#             return result
        
#         elif isinstance(json_data, str):
#             return {"type": "string", "length": len(json_data)}
        
#         elif isinstance(json_data, (int, float)):
#             return {"type": "number", "value": json_data}
        
#         elif isinstance(json_data, bool):
#             return {"type": "boolean", "value": json_data}
        
#         elif json_data is None:
#             return {"type": "null"}
        
#         else:
#             return {"type": "unknown"}
    
#     def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
#         """Split text into overlapping chunks for better analysis."""
#         if not text or len(text) <= chunk_size:
#             return [text] if text else []
        
#         chunks = []
#         start = 0
#         while start < len(text):
#             end = start + chunk_size
            
#             # Try to find a natural break point (sentence or paragraph end)
#             if end < len(text):
#                 # Look for paragraph break
#                 paragraph_break = text.rfind('\n\n', start, end)
#                 if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
#                     end = paragraph_break + 2
#                 else:
#                     # Look for sentence break
#                     sentence_break = max(
#                         text.rfind('. ', start, end),
#                         text.rfind('! ', start, end),
#                         text.rfind('? ', start, end)
#                     )
#                     if sentence_break != -1 and sentence_break > start + chunk_size // 2:
#                         end = sentence_break + 2
            
#             chunks.append(text[start:end])
#             start = end - overlap
        
#         return chunks

#     def rag_query(self, question: str, filename: str, top_k: int = 5, relevance_threshold: float = 0.75) -> str:
#         """Advanced RAG: Hybrid retrieval, context distillation, reranking, query expansion, and hallucination fallback using Hugging Face Inference API."""
#         # --- Query Expansion ---
#         expansions = [question]
#         if "summary" in question.lower():
#             expansions.append(question.replace("summary", "overview"))
#         if "explain" in question.lower():
#             expansions.append(question.replace("explain", "describe"))

#         chunk_scores = {}
#         model_names_to_use = self.embedding_model_names if self.use_all_models else [self.embedding_model_names[0]]
#         for q in expansions:
#             for model_name in model_names_to_use:
#                 q_emb = self._get_hf_embeddings(model_name, [q])[0]
#                 results = self.chroma_collection.query(
#                     query_embeddings=[q_emb],
#                     n_results=top_k,
#                     where={"filename": filename},
#                     include=["distances", "documents"]
#                 )
#                 docs = results.get('documents', [[]])[0]
#                 dists = results.get('distances', [[]])[0]
#                 for doc, dist in zip(docs, dists):
#                     if doc:
#                         similarity = 1 - dist
#                         is_diverse = all(doc not in c or abs(similarity - s) > 0.01 for c, s in chunk_scores.items())
#                         if is_diverse and (doc not in chunk_scores or similarity > chunk_scores[doc]):
#                             chunk_scores[doc] = similarity

#         reranked = sorted(chunk_scores.items(), key=lambda x: -x[1])
#         relevant_chunks = [doc for doc, sim in reranked if sim >= relevance_threshold]

#         if not relevant_chunks:
#             prompt = (
#                 "No relevant context was found in the provided documents. Answering based on general knowledge.\n\n"
#                 f"User: {question}\n"
#                 "Assistant:"
#             )
#             return self._get_ai_analysis(prompt, filename)

#         max_chunks = 5
#         if len(relevant_chunks) > max_chunks:
#             context_text = "\n".join(relevant_chunks[:max_chunks])
#             summary_prompt = f"Summarize the following context for answering a user question:\n\n{context_text}"
#             context = self._get_ai_analysis(summary_prompt, filename)
#         else:
#             context = "\n".join(relevant_chunks)

#         prompt = (
#             "You are a friendly and knowledgeable AI assistant designed to help users with accurate, context-aware answers.\n\n"
#             "When context is provided, use ONLY the information in the context. If context is not available or insufficient, rely on your general knowledge—but always prioritize context when available.\n\n"
#             "Respond conversationally and helpfully.\n\n"
#             "Context (from documents):\n"
#             "--------------------------\n"
#             f"{context}\n"
#             "--------------------------\n\n"
#             f"User: {question}\n"
#             "Assistant:"
#         )
#         return self._get_ai_analysis(prompt, filename)

#     def _get_hf_embeddings(self, model_name: str, sentences: list) -> list:
#         """Get embeddings from Hugging Face Inference API for a list of sentences."""
#         import os
#         import requests
#         HF_TOKEN = os.environ.get('HF_TOKEN')
#         if not HF_TOKEN:
#             raise RuntimeError("HF_TOKEN environment variable is required for Hugging Face Inference API.")
#         headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#         # Map model_name to correct API endpoint
#         if model_name == "sentence-transformers/all-MiniLM-L6-v2":
#             api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
#         elif model_name == "BAAI/bge-small-en-v1.5":
#             api_url = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction"
#         elif model_name == "intfloat/e5-small-v2":
#             api_url = "https://router.huggingface.co/hf-inference/models/intfloat/e5-small-v2/pipeline/feature-extraction"
#         else:
#             raise ValueError(f"Unknown model name for HF API: {model_name}")
#         # The API expects a single string or a list of strings
#         payload = {"inputs": sentences if len(sentences) > 1 else sentences[0]}
#         response = requests.post(api_url, headers=headers, json=payload)
#         if response.status_code != 200:
#             raise RuntimeError(f"HF API error: {response.text}")
#         result = response.json()
#         # The result is a list of embeddings (or a single embedding)
#         if isinstance(result, list) and isinstance(result[0], list):
#             return result
#         elif isinstance(result, list):
#             return [result]
#         else:
#             raise RuntimeError(f"Unexpected HF API response: {result}")

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
from io import StringIO, BytesIO
import base64

# Conditional imports for optional dependencies
try:
    from PIL import Image
    import easyocr
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False

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
    
    def __init__(self, groq_api_key: str,):
        """
        Initialize the FileAnalyzer with API keys.
        
        Args:
            groq_api_key: API key for Groq service
            hf_token: HuggingFace API token for embedding model
        """
        if not groq_api_key:
            raise ValueError("Both Groq and HuggingFace API keys are required")
            
        self.groq_api_key = groq_api_key
        
        
        # Initialize ChromaDB with optimized settings
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            "rag_collection",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None  # We'll handle embeddings ourselves
        )
        
        # Configuration
        self.embedding_model = "sentence-transformers/all-mpnet-base-v2"
        self.embedding_batch_size = 32
        self.chunk_min_size = 256
        self.chunk_max_size = 1024
        self.rag_top_k = 5
        self.rag_min_score = 0.7

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
            if file_ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]:
                result = self._analyze_image(temp_path)
            elif file_ext == ".pdf":
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

    def _analyze_image(self, file_path: str) -> Dict[str, Any]:
        """Analyze image file with OCR and visual description."""
        if not HAS_IMAGE_SUPPORT:
            raise HTTPException(
                status_code=500,
                detail="Image processing dependencies not installed"
            )
            
        try:
            with Image.open(file_path) as img:
                # Basic metadata
                info = {
                    "format": img.format,
                    "mode": img.mode,
                    "width": img.width,
                    "height": img.height,
                    "size": os.path.getsize(file_path),
                    "aspect_ratio": round(img.width / img.height, 2) if img.height else 0
                }
                
                # OCR text extraction
                reader = easyocr.Reader(['en'], gpu=False)
                ocr_result = reader.readtext(file_path, detail=0)
                extracted_text = '\n'.join(ocr_result)
                
                # Get AI description
                description = self._describe_image(file_path)
                
                return {
                    "file_type": "image",
                    "metadata": info,
                    "extracted_text": extracted_text,
                    "ai_description": description
                }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Image analysis failed: {str(e)}"
            )

    def _describe_image(self, image_path: str) -> Dict[str, Any]:
        """Get AI-generated description of image content."""
        try:
            with Image.open(image_path) as img:
                # Resize to reduce API payload
                img.thumbnail((800, 800))
                
                # Convert to base64
                buffered = BytesIO()
                img.save(buffered, format=img.format or "JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                headers = {
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image in detail."},
                            {
                                "type": "image_url",
                                "image_url": f"data:image/jpeg;base64,{img_str}"
                            }
                        ]
                    }],
                    "max_tokens": 512
                }
                
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=30
                )
                response.raise_for_status()
                
                description = response.json()["choices"][0]["message"]["content"]
                
                # Extract key elements
                elements = re.findall(r'\b\w{4,}\b', description.lower())
                common_objects = {
                    'person', 'people', 'man', 'woman', 'child', 
                    'building', 'car', 'tree', 'animal', 'text'
                }
                tags = list(set(elements) & common_objects)
                
                return {
                    "description": description,
                    "tags": tags[:10]
                }
                
        except Exception as e:
            return {
                "description": f"Could not generate description: {str(e)}",
                "tags": []
            }

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
        
        if ext in [".jpg", ".jpeg", ".png"]:
            return base + "Focus on visual elements and extracted text:\n\n" + text
        elif ext == ".pdf":
            return base + "Focus on document structure and key information:\n\n" + text
        elif ext == ".csv":
            return base + "Focus on data patterns and potential insights:\n\n" + text
        elif ext == ".json":
            return "Analyze this JSON data's structure and content:\n1. Data schema\n2. Key entities\n3. Relationships\n\n" + text
        else:
            return base + text