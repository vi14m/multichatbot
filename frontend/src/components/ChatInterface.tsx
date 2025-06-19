import { useState, useEffect, forwardRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { ChatMode, Message, ChatRequest } from '@/types/chat';
import { sendChatMessage, transcribeAudio, analyzeFile } from '@/services/api';

interface ChatInterfaceProps {
  mode: ChatMode;
}

const ChatInterface = forwardRef<HTMLDivElement, ChatInterfaceProps>(({ mode }, ref) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

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
        file_context: fileContext
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
        
        // Create content from analysis results
        let analysisContent = `# Analysis of ${file.name}\n\n`;
        
        // Add file type and metadata
        analysisContent += `**File Type:** ${result.file_type}\n\n`;
        
        // Add metadata section
        analysisContent += `## Metadata\n\n`;
        if (result.file_type === 'image') {
          analysisContent += `- Format: ${result.metadata.format}\n`;
          analysisContent += `- Dimensions: ${result.metadata.width} × ${result.metadata.height}\n`;
          analysisContent += `- Size: ${Math.round(result.metadata.file_size / 1024)} KB\n\n`;
        } else if (result.file_type === 'pdf') {
          analysisContent += `- Pages: ${result.metadata.num_pages}\n`;
          analysisContent += `- Size: ${Math.round(result.metadata.file_size / 1024)} KB\n\n`;
          
          if (result.metadata.info && Object.keys(result.metadata.info).length > 0) {
            analysisContent += `### Document Info\n\n`;
            for (const [key, value] of Object.entries(result.metadata.info)) {
              analysisContent += `- ${key}: ${value}\n`;
            }
            analysisContent += `\n`;
          }
        } else if (result.file_type === 'csv') {
          analysisContent += `- Rows: ${result.metadata.rows}\n`;
          analysisContent += `- Columns: ${result.metadata.columns}\n\n`;
          
          if (result.metadata.headers && result.metadata.headers.length > 0) {
            analysisContent += `### Headers\n\n`;
            analysisContent += `\`${result.metadata.headers.join('\`, \`')}\`\n\n`;
          }
          
          if (result.metadata.sample && result.metadata.sample.length > 0) {
            analysisContent += `### Sample Data\n\n`;
            analysisContent += `\`\`\`\n`;
            for (const row of result.metadata.sample) {
              analysisContent += `${row.join(', ')}\n`;
            }
            analysisContent += `\`\`\`\n\n`;
          }
        } else if (result.file_type === 'text' || result.file_type === 'json') {
          analysisContent += `- Lines: ${result.metadata.lines || 'N/A'}\n`;
          analysisContent += `- Length: ${result.metadata.length} characters\n\n`;
          
          if (result.file_type === 'json' && 'is_valid' in result.metadata) {
            analysisContent += `- Valid JSON: ${result.metadata.is_valid ? 'Yes' : 'No'}\n\n`;
          }
        }
        
        // Add AI analysis if available
        if (result.ai_analysis) {
          analysisContent += `## AI Analysis\n\n${result.ai_analysis}\n\n`;
        }
        
        // Add extracted text preview if available
        if (result.extracted_text) {
          const previewText = result.extracted_text.length > 500 
            ? result.extracted_text.substring(0, 500) + '... (text truncated)' 
            : result.extracted_text;
          
          analysisContent += `## Extracted Text Preview\n\n\`\`\`\n${previewText}\n\`\`\`\n`;
        }
        
        // Add processing time
        analysisContent += `\n*Processing time: ${result.processing_time.toFixed(2)} seconds*`;
        
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
        />
      </div>
    </div>
  );
});

ChatInterface.displayName = 'ChatInterface';

export default ChatInterface;