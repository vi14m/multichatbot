# Function Calling Framework for MultiChatBot

This document describes the function-calling framework implemented in MultiChatBot, which allows the chatbot to trigger external tools beyond basic file reading and analysis.

## Overview

The function-calling framework enables the chatbot to:

1. Call external tools and APIs
2. Process the results of these calls
3. Generate responses based on the tool outputs

## Implemented Tools

The following tools have been implemented:

### 1. Web Search & Retrieval

Two web search options are available:

- **Google Search**: Uses the `googlesearch-python` package to perform web searches
- **Tavily Search**: Uses the Tavily API for more comprehensive search results (requires a Tavily API key)

### 2. Code Snippet Runner

- **Python Code Execution**: Executes Python code snippets and returns the output
- Includes safety measures to prevent harmful code execution

### 3. Data Formatter/Converter

- Converts data between different formats (CSV, JSON, TXT)
- Handles various data structures (lists, dictionaries, etc.)

## Implementation Details

### Backend

- `tools.py`: Defines the `ToolRegistry` class and implements the tool functions
- `app.py`: Integrates the function-calling framework with the chat endpoints

### Frontend

- Updated chat interface to display tool calls and results
- Added `ToolCall.tsx` component for rendering tool calls and their results
- Updated types in `chat.ts` to include tool calls and results

## Usage

The function-calling framework is enabled by default for all chat modes. To disable it for a specific request, set `use_tools: false` in the chat request.

### Example

```typescript
const request: ChatRequest = {
  mode: 'chat',
  message: 'Search for information about climate change',
  conversation_history: [],
  use_tools: true // Enable function calling (default)
};
```

## Adding New Tools

To add a new tool to the framework:

1. Implement the tool function in `tools.py`
2. Register the tool with the `tool_registry` using the `register_tool` method
3. Define the tool's parameters using JSON Schema

### Example

```python
def my_new_tool(param1: str, param2: int) -> str:
    # Tool implementation
    return f"Result: {param1}, {param2}"

tool_registry.register_tool(
    name="my_new_tool",
    function=my_new_tool,
    description="Description of what the tool does",
    parameters={
        "type": "object",
        "properties": {
            "param1": {
                "type": "string",
                "description": "Description of param1"
            },
            "param2": {
                "type": "integer",
                "description": "Description of param2"
            }
        },
        "required": ["param1", "param2"]
    }
)
```

## Dependencies

The function-calling framework requires the following dependencies:

- `googlesearch-python`: For Google search functionality
- `tavily-python`: For Tavily search functionality (optional)

These dependencies are listed in the `requirements.txt` file.