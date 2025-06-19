import { useState, useRef, FormEvent, ChangeEvent } from 'react';
import { 
  PaperAirplaneIcon, 
  XMarkIcon, 
  DocumentIcon, 
  PhotoIcon,
  DocumentTextIcon,
  TableCellsIcon
} from '@heroicons/react/24/solid';
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
  const isAnalyzeMode = mode === 'analyze';

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    
    if ((isTranscribeMode || isAnalyzeMode) && selectedFile) {
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
      if (isTranscribeMode && file.type !== 'audio/wav') {
        alert('Only .wav files are supported for transcription');
        return;
      }
      
      if (isAnalyzeMode) {
        // Check if file type is supported for analysis
        const fileType = file.type.toLowerCase();
        const fileExt = file.name.split('.').pop()?.toLowerCase() || '';
        
        const supportedTypes = [
          // Images
          'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
          // Documents
          'application/pdf', 'text/plain', 'text/csv', 'application/json',
          'text/markdown'
        ];
        
        const supportedExts = [
          'jpg', 'jpeg', 'png', 'gif', 'bmp',
          'pdf', 'txt', 'csv', 'json', 'md'
        ];
        
        if (!supportedTypes.includes(fileType) && !supportedExts.includes(fileExt)) {
          alert('Unsupported file type. Please upload an image, PDF, or text file.');
          return;
        }
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

  // Function to determine which icon to show based on file type
  const fileTypeIcon = (file: File) => {
    const fileType = file.type.toLowerCase();
    const fileExt = file.name.split('.').pop()?.toLowerCase() || '';
    
    // Image types
    if (fileType.startsWith('image/') || ['jpg', 'jpeg', 'png', 'gif', 'bmp'].includes(fileExt)) {
      return <PhotoIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />;
    }
    
    // PDF
    if (fileType === 'application/pdf' || fileExt === 'pdf') {
      return <DocumentIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />;
    }
    
    // CSV
    if (fileType === 'text/csv' || fileExt === 'csv') {
      return <TableCellsIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />;
    }
    
    // Text files
    if (fileType === 'text/plain' || fileExt === 'txt' || fileExt === 'md' || fileExt === 'json') {
      return <DocumentTextIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />;
    }
    
    // Default
    return <DocumentIcon className="w-5 h-5 text-primary-600 dark:text-primary-400 mr-2" />;
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4">
      <div className="relative flex flex-col gap-2">
        {/* Show selected file as a chip above the input only in analyze mode */}
        {isAnalyzeMode && selectedFile && (
          <div className="flex items-center bg-gray-200 dark:bg-dark-600 px-2 py-1 rounded text-xs mb-1 ml-10 w-fit">
            {fileTypeIcon(selectedFile)}
            <span className="truncate max-w-[120px]">{selectedFile.name}</span>
            <button type="button" onClick={handleRemoveFile} className="ml-1 text-gray-500 hover:text-red-500">
              <XMarkIcon className="w-4 h-4" />
            </button>
          </div>
        )}
        <div className="flex items-center relative">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={`Type your message${isAnalyzeMode ? ' or upload a document' : ''}...`}
            className="w-full p-4 pl-12 pr-16 rounded-lg border border-gray-300 dark:border-dark-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100"
            disabled={isLoading}
          />
          {/* Upload icon only visible for file upload in analyze mode */}
          {isAnalyzeMode && (
            <>
              <button
                type="button"
                onClick={handleBrowseClick}
                className="absolute left-2 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-gray-100 hover:bg-primary-100 dark:bg-dark-600 dark:hover:bg-primary-900 transition-colors"
                title="Upload document"
              >
                <DocumentIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept={".jpg,.jpeg,.png,.gif,.bmp,.pdf,.txt,.csv,.json,.md"}
                className="hidden"
              />
            </>
          )}
          <button
            type="submit"
            className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-full ${isLoading ? 'bg-gray-300 dark:bg-dark-600 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700 text-white'} transition-colors`}
            disabled={isLoading || (isAnalyzeMode && selectedFile == null && !message.trim()) || (!isAnalyzeMode && !message.trim())}
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
      </div>
    </form>
  );
};

export default MessageInput;