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
        <p className="text-gray-500 dark:text-gray-400 max-w-md">
          Select a mode from the dropdown above and start chatting. Each mode is powered by a specialized AI model.
        </p>
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