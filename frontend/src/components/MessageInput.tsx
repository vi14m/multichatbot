import { useState, useRef, FormEvent, ChangeEvent } from 'react';
import { PaperAirplaneIcon, MicrophoneIcon, XMarkIcon } from '@heroicons/react/24/solid';
import { ChatMode } from '@/types/chat';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  onFileUpload?: (file: File) => void;
  mode: ChatMode;
  isLoading: boolean;
}

const MessageInput = ({ onSendMessage, onFileUpload, mode, isLoading }: MessageInputProps) => {
  const [message, setMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isTranscribeMode = mode === 'transcribe';

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    if (isTranscribeMode && selectedFile) {
      onFileUpload?.(selectedFile);
      setSelectedFile(null);
      return;
    }
    
    if (message.trim()) {
      onSendMessage(message);
      setMessage('');
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      if (file.type !== 'audio/wav') {
        alert('Only .wav files are supported');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4">
      <div className="relative">
        {isTranscribeMode ? (
          <div className="bg-white dark:bg-dark-700 rounded-lg border border-gray-300 dark:border-dark-600 p-4">
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".wav"
              className="hidden"
            />
            
            {selectedFile ? (
              <div className="flex items-center justify-between bg-gray-100 dark:bg-dark-600 p-3 rounded">
                <div className="flex items-center">
                  <MicrophoneIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />
                  <span className="text-sm truncate">{selectedFile.name}</span>
                </div>
                <button
                  type="button"
                  onClick={handleRemoveFile}
                  className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <XMarkIcon className="w-5 h-5" />
                </button>
              </div>
            ) : (
              <div className="text-center">
                <MicrophoneIcon className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-2" />
                <p className="text-gray-600 dark:text-gray-300 mb-2">Upload a .wav file for transcription</p>
                <button
                  type="button"
                  onClick={handleBrowseClick}
                  className="bg-primary-600 hover:bg-primary-700 text-white px-4 py-2 rounded-md transition-colors"
                >
                  Browse Files
                </button>
              </div>
            )}
          </div>
        ) : (
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={`Type your message for ${mode} mode...`}
            className="w-full p-4 pr-16 rounded-lg border border-gray-300 dark:border-dark-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100"
            disabled={isLoading}
          />
        )}
        
        <button
          type="submit"
          className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-full ${isLoading ? 'bg-gray-300 dark:bg-dark-600 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700 text-white'} transition-colors`}
          disabled={isLoading || (isTranscribeMode && !selectedFile) || (!isTranscribeMode && !message.trim())}
        >
          <PaperAirplaneIcon className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
};

export default MessageInput;