import { useState, useEffect, forwardRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { ChatMode, Message, ChatRequest, ToolCall, ToolResult } from '@/types/chat';
import { sendChatMessage, transcribeAudio, analyzeFile } from '@/services/api';

interface ChatInterfaceProps {
  mode: ChatMode;
}

const ChatInterface = forwardRef<HTMLDivElement, ChatInterfaceProps>(({ mode }, ref) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [useToolsEnabled, setUseToolsEnabled] = useState(false); // New state for tool usage
  const [selectedTools, setSelectedTools] = useState<string[]>([]); // New state for selected tools

  // Reset messages when mode changes
  useEffect(() => {
    setMessages([]);
  }, [mode]);

  const handleSendMessage = async (content: string) => {
    // Add user message to the chat
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: new Date(),
      mode,
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      // Prepare conversation history for the API
      const conversationHistory = messages
        .filter(msg => msg.mode === mode) // Only include messages from the current mode
        .map(msg => ({
          role: msg.role,
          content: msg.content,
        }));
      
      // Find the most recent file analysis result if in analyze mode
      let fileContext = null;
      if (mode === 'analyze') {
        const lastFileAnalysis = [...messages].reverse().find(msg => msg.fileAnalysis);
        if (
          lastFileAnalysis?.fileAnalysis &&
          typeof lastFileAnalysis.fileAnalysis.file_name === 'string' &&
          typeof lastFileAnalysis.fileAnalysis.file_type === 'string' &&
          typeof lastFileAnalysis.fileAnalysis.extracted_text === 'string'
        ) {
          fileContext = {
            file_name: lastFileAnalysis.fileAnalysis.file_name,
            file_type: lastFileAnalysis.fileAnalysis.file_type,
            extracted_text: lastFileAnalysis.fileAnalysis.extracted_text
          };
        }
      }
      
      // Create the request
      const request: ChatRequest = {
        mode,
        message: content,
        conversation_history: conversationHistory,
        file_context: fileContext,
        use_tools: useToolsEnabled, // Use the state variable for tool usage
        selected_tools: selectedTools // Pass selected tools to the backend
      };
      
      // Send the request to the API
      const response = await sendChatMessage(request);
      
      // Add assistant message to the chat
      const assistantMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        mode,
        // Only include toolCalls and toolResults if the LLM's response is empty
        // This prevents displaying raw tool output when the LLM generates a conversational response
        toolCalls: response.response ? undefined : response.tool_calls,
        toolResults: response.response ? undefined : response.tool_results
      };
      
      setMessages((prev) => [...prev, assistantMessage]);
      
      // Show token usage in toast if available
      if (response.token_count) {
        toast.info(
          `Tokens: ${response.token_count.prompt_tokens} prompt + ${response.token_count.completion_tokens} completion = ${response.token_count.total_tokens} total`,
          { autoClose: 3000, position: 'bottom-right' }
        );
      }
    } catch (error) {
      toast.error(typeof error === 'string' ? error : 'Failed to get response');
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    try {
      setIsLoading(true);
      
      if (mode === 'transcribe') {
        // Handle audio transcription
        // Add user message with file name
        const userMessage: Message = {
          id: uuidv4(),
          role: 'user',
          content: `Transcribe audio: ${file.name}`,
          timestamp: new Date(),
          mode: mode
        };
        
        setMessages(prev => [...prev, userMessage]);
        
        // Transcribe the audio
        const result = await transcribeAudio(file);
        
        // Add assistant message with transcription
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'assistant',
          content: result.text,
          timestamp: new Date(),
          mode: mode
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      } else if (mode === 'analyze') {
        // Handle file analysis
        // Add user message with file name
        const userMessage: Message = {
          id: uuidv4(),
          role: 'user',
          content: `Analyze file: ${file.name}`,
          timestamp: new Date(),
          mode: mode
        };
        
        setMessages(prev => [...prev, userMessage]);
        
        // Add a system message to inform the user they can ask questions
        const systemMessage: Message = {
          id: uuidv4(),
          role: 'assistant',
          content: `Thanks for uploading your document **${file.name}**. I'll analyze it for you. Once the analysis is complete, you can ask me any questions about the content!`,
          timestamp: new Date(),
          mode: mode
        };
        
        setMessages(prev => [...prev, systemMessage]);
        
        // Analyze the file
        const result = await analyzeFile(file);
        
        // Only show AI analysis for analyzed files (no headings, no file name, no extra sections)
        let analysisContent = result.ai_analysis ? result.ai_analysis : 'No AI analysis available.';
        
        // Add assistant message with analysis
        const assistantMessage: Message = {
          id: uuidv4(),
          role: 'assistant',
          content: analysisContent,
          timestamp: new Date(),
          mode: mode,
          fileAnalysis: {
            ...result,
            file_name: file.name // Add file name to the analysis results
          }
        };
        
        setMessages(prev => [...prev, assistantMessage]);
      }
    } catch (error) {
      console.error(`Error processing file:`, error);
      
      // Add error message
      const errorMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: `Error processing file: ${error instanceof Error ? error.message : String(error)}`,
        timestamp: new Date(),
        mode: mode
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto px-4" ref={ref}>
        <MessageList messages={messages} isLoading={isLoading} />
      </div>
      
      <div className="px-4 pb-4">
        <MessageInput 
          onSendMessage={handleSendMessage} 
          onFileUpload={handleFileUpload}
          mode={mode} 
          isLoading={isLoading}
          useToolsEnabled={useToolsEnabled}
        setUseToolsEnabled={setUseToolsEnabled}
        selectedTools={selectedTools}
        setSelectedTools={setSelectedTools}
      />
      </div>
    </div>
  );
});

ChatInterface.displayName = 'ChatInterface';

export default ChatInterface;