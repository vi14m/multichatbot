import { useState, useEffect, forwardRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { toast } from 'react-toastify';
import MessageList from './MessageList';
import MessageInput from './MessageInput';
import { ChatMode, Message, ChatRequest } from '@/types/chat';
import { sendChatMessage, transcribeAudio } from '@/services/api';

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
      
      // Create the request
      const request: ChatRequest = {
        mode,
        message: content,
        conversation_history: conversationHistory,
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
    // Add user message indicating file upload
    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content: `Uploaded file: ${file.name}`,
      timestamp: new Date(),
      mode,
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    
    try {
      // Send the file to the API for transcription
      const response = await transcribeAudio(file);
      
      // Add assistant message with the transcription
      const assistantMessage: Message = {
        id: uuidv4(),
        role: 'assistant',
        content: `**Transcription:**\n\n${response.text}`,
        timestamp: new Date(),
        mode,
      };
      
      setMessages((prev) => [...prev, assistantMessage]);
      
      toast.success(`Transcription completed in ${response.processing_time.toFixed(2)}s`);
    } catch (error) {
      toast.error(typeof error === 'string' ? error : 'Failed to transcribe audio');
      console.error('Error transcribing audio:', error);
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