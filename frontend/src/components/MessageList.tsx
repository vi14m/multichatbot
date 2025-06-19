import { Message as MessageType } from '@/types/chat';
import Message from './Message';

interface MessageListProps {
  messages: MessageType[];
  isLoading: boolean;
}

const MessageList = ({ messages, isLoading }: MessageListProps) => {
  if (messages.length === 0 && !isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-8">
        <div className="text-6xl mb-4">👋</div>
        <h2 className="text-2xl font-bold text-gray-700 dark:text-gray-300 mb-2">
          Welcome to MultiChatBot!
        </h2>
        <p className="text-gray-500 dark:text-gray-400 max-w-md mb-4">
          Select a mode from the dropdown above and start chatting. Each mode is powered by a specialized AI model.
        </p>
        
        {/* File Analysis Feature Highlight */}
        <div className="bg-primary-50 dark:bg-primary-900/20 border border-primary-100 dark:border-primary-800 rounded-lg p-4 mt-4 max-w-md">
          <h3 className="text-lg font-medium text-primary-700 dark:text-primary-300 mb-2 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            Try Document Analysis
          </h3>
          <p className="text-primary-600 dark:text-primary-400 text-sm">
            Select the "Analyze Files" mode to upload documents (PDF, images, text) and chat with AI about their contents.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4 py-4">
      {messages.map((message) => (
        <Message key={message.id} message={message} />
      ))}
      
      {isLoading && (
        <div className="flex justify-start mb-4">
          <div className="max-w-[80%]">
            <div className="rounded-lg p-4 bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600">
              <div className="typing-animation flex space-x-1">
                <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500"></span>
                <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500"></span>
                <span className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500"></span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MessageList;