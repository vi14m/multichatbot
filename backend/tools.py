from typing import Dict, Any, List, Optional, Callable
import io
import contextlib
import json
import csv
import requests
import os
import time
from tavily import TavilyClient
import dotenv
dotenv.load_dotenv()

# -----------------------------
# Helper: language -> compiler
# -----------------------------
_COMPILER_ALIASES: Dict[str, str] = {
    # Python
    "python": "python-3.9.7",
    "py": "python-3.9.7",
    "python3": "python-3.9.7",
    "python-3": "python-3.9.7",
    # C++
    "cpp": "cpp-17",
    "c++": "cpp-17",
    "g++": "cpp-17",
    # C
    "c": "c-17",
    # Java
    "java": "java-17",
    # JavaScript / Node
    "javascript": "nodejs-18",
    "js": "nodejs-18",
    "node": "nodejs-18",
    "nodejs": "nodejs-18",
    # Go
    "go": "go-1.20",
    "golang": "go-1.20",
    # Rust
    "rust": "rust-1.70",
    # Ruby
    "ruby": "ruby-3.2",
    # PHP
    "php": "php-8.2",
    # Kotlin
    "kotlin": "kotlin-1.8",
    # Swift
    "swift": "swift-5.7",
    # C#
    "csharp": "dotnet-7",
    "cs": "dotnet-7",
    "dotnet": "dotnet-7",
    # Bash
    "bash": "bash",
    "sh": "bash",
    # TypeScript (Node)
    "ts": "ts-node-10",
    "typescript": "ts-node-10",
}

def _normalize_compiler(language: str) -> str:
    """
    Normalize a user/LLM-provided language string to
    an OnlineCompiler.io compiler identifier.
    """
    if not language:
        return "python-3.9.7"
    key = language.strip().lower()
    return _COMPILER_ALIASES.get(key, language)

# -----------------------------
# Tool implementations
# -----------------------------
def perform_tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> str:
    """Perform a web search using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Tavily API key not found in environment variables."

    valid_depths = ["basic", "advanced"]
    if search_depth not in valid_depths:
        return f"Error: Invalid search depth '{search_depth}'. Valid options are: {', '.join(valid_depths)}"

    client = TavilyClient(api_key=tavily_api_key)
    try:
        response = client.search(query=query, search_depth=search_depth, max_results=max_results)
    except Exception as e:
        return f"Error: Tavily search failed: {e}"

    results = []
    for result in response.get("results", []):
        title = result.get("title") or ""
        content = result.get("content") or ""
        url = result.get("url") or ""
        results.append(f"TITLE: {title}\nCONTENT: {content}\nURL: {url}")

    return "\n\n".join(results) if results else "No results found."

def perform_tavily_extract(url: str, extract_depth: str = "basic") -> str:
    """Extract content from a specific URL using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Tavily API key not found in environment variables."

    valid_depths = ["basic", "advanced"]
    if extract_depth not in valid_depths:
        return f"Error: Invalid extract depth '{extract_depth}'. Valid options are: {', '.join(valid_depths)}"

    client = TavilyClient(api_key=tavily_api_key)
    try:
        response = client.extract(url=url, extract_depth=extract_depth)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

def perform_tavily_crawl(url: str, max_pages: int = 5) -> str:
    """Crawl a website and extract content using Tavily API."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "Error: Tavily API key not found in environment variables."

    client = TavilyClient(api_key=tavily_api_key)
    try:
        response = client.crawl(url=url, max_pages=max_pages)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error crawling website: {str(e)}"

def execute_code_online(code: str, language: str, program_input: str = "", retries: int = 1, timeout: int = 60) -> str:
    """
    Execute code using OnlineCompiler.io API.

    Args:
        code (str): Source code to execute.
        language (str): Human name or compiler ID ("python", "cpp-17", etc.).
        program_input (str): Optional stdin for the program.
        retries (int): Retry count on transient failures.
        timeout (int): Request timeout in seconds.

    Returns:
        str: Program output (stdout/stderr combined if provided by API) or an error message.
    """
    api_key = os.getenv("ONLINE_COMPILER_API_KEY")
    if not api_key:
        return "Error: OnlineCompiler.io API key not found in environment variables."

    url = "https://onlinecompiler.io/api/v2/run-code/"
    headers = {
        "Accept": "*/*",
        "Authorization": api_key,
        "Content-Type": "application/json"
    }

    compiler = _normalize_compiler(language)
    payload = {
        "code": code,
        "input": program_input or "",
        "compiler": compiler
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "success":
                output = result.get("stdout", "") or ""
                stderr = result.get("stderr", "") or ""
                return f"Output:\n{output}\nStderr:\n{stderr}".strip()
            else:
                return f"Error: {result.get('message', 'Unknown error')}"

        except requests.exceptions.RequestException as e:
            if attempt < retries:
                time.sleep(1)
            else:
                return f"Error: Request to OnlineCompiler.io failed: {e}"

def format_data(data: str, input_format: str, output_format: str) -> str:
    """Convert data between formats"""
    parsed_data = None
    try:
        if input_format == "json":
            parsed_data = json.loads(data)
        elif input_format == "csv":
            csv_reader = csv.DictReader(io.StringIO(data))
            parsed_data = list(csv_reader)
        elif input_format == "txt":
            parsed_data = data.strip().split('\n')
        else:
            return f"Unsupported input format: {input_format}"

        if output_format == "json":
            return json.dumps(parsed_data, indent=2)
        elif output_format == "csv":
            if not parsed_data or not isinstance(parsed_data, list) or not isinstance(parsed_data[0], dict):
                return "Error: CSV output requires a list of dictionaries."
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=parsed_data[0].keys())
            writer.writeheader()
            writer.writerows(parsed_data)
            return output.getvalue()
        elif output_format == "txt":
            if isinstance(parsed_data, list):
                return '\n'.join(map(str, parsed_data))
            else:
                return str(parsed_data)
        else:
            return f"Unsupported output format: {output_format}"
    except Exception as e:
        return f"Error converting data: {e}"

class ToolRegistry:
    """Registry for tools that can be called by the chatbot"""
    
    def __init__(self):
        self.tools = {}
        self.register_default_tools()
    
    def register_tool(self, name: str, function: Callable, description: str, parameters: Dict[str, Any], modes: Optional[List[str]] = None):
        """Register a new tool"""
        self.tools[name] = {
            "function": function,
            "description": description,
            "parameters": parameters,
            "modes": modes if modes is not None else []
        }
    
    def get_tool(self, name: str) -> Dict[str, Any]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def get_tool_descriptions(self, mode: Optional[str] = None, selected_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get descriptions of all registered tools for function calling"""
        filtered_tools = []
        for name, tool in self.tools.items():
            mode_match = (mode is None or not tool["modes"] or mode in tool["modes"])
            selected_tools_match = (selected_tools is None or name in selected_tools)
            if mode_match and selected_tools_match:
                filtered_tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": tool["parameters"]
                    }
                })
        return filtered_tools
    
    def execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool with the given arguments"""
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool '{name}' not found")
        return tool["function"](**arguments)
    
    def register_default_tools(self):
        """Register default tools"""
        self.register_tool(
            name="web_search",
            function=perform_tavily_search,
            description="Search the web for information using Tavily API",
            modes=["chat", "research"],
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "search_depth": {"type": "string", "description": "The depth of search (basic or advanced)", "enum": ["basic", "advanced"], "default": "basic"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5}
                },
                "required": ["query"]
            }
        )
        
        self.register_tool(
            name="execute_code_online",
            function=execute_code_online,
            description="Execute code using OnlineCompiler.io API.",
            modes=["chat", "code", "math"],
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Source code to execute."},
                    "language": {"type": "string", "description": "Human name or compiler ID ('python', 'cpp-17', etc.)."},
                    "program_input": {"type": "string", "description": "Optional stdin for the program."}
                },
                "required": ["code", "language"]
            }
        )
        
        self.register_tool(
            name="format_data",
            function=format_data,
            description="Convert data between formats (CSV, JSON, etc.).",
            modes=["chat", "research", "analyze"],
            parameters={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "The data to convert."},
                    "input_format": {"type": "string", "description": "The input format (csv, json, txt).", "enum": ["csv", "json", "txt"]},
                    "output_format": {"type": "string", "description": "The output format (csv, json, txt)", "enum": ["csv", "json", "txt"]}
                },
                "required": ["data", "input_format", "output_format"]
            }
        )

        self.register_tool(
            name="tavily_extract",
            function=perform_tavily_extract,
            description="Extract content from a specific URL using Tavily API",
            modes=["chat", "research"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to extract content from"},
                    "extract_depth": {"type": "string", "description": "The depth of extraction (basic or advanced)", "enum": ["basic", "advanced"], "default": "basic"}
                },
                "required": ["url"]
            }
        )
        
        self.register_tool(
            name="tavily_crawl",
            function=perform_tavily_crawl,
            description="Crawl a website and extract content using Tavily API",
            modes=["chat", "research"],
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL of the website to crawl"},
                    "max_pages": {"type": "integer", "description": "Maximum number of pages to crawl", "default": 5}
                },
                "required": ["url"]
            }
        )

tool_registry = ToolRegistry()