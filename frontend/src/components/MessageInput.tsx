import { useState, useRef, FormEvent, ChangeEvent, Dispatch, SetStateAction } from 'react';
import { 
  PaperAirplaneIcon, 
  XMarkIcon, 
  DocumentIcon, 
  PhotoIcon,
  DocumentTextIcon,
  TableCellsIcon
} from '@heroicons/react/24/solid';
import { ChatMode } from '@/types/chat';

import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  onFileUpload?: (file: File) => void;
  mode: ChatMode;
  isLoading: boolean;
  useToolsEnabled: boolean; // Added
  setUseToolsEnabled: Dispatch<SetStateAction<boolean>>; // Added
  selectedTools: string[];
  setSelectedTools: (tools: string[]) => void;
}

const MessageInput = ({ onSendMessage, onFileUpload, mode, isLoading, useToolsEnabled, setUseToolsEnabled, selectedTools, setSelectedTools }: MessageInputProps) => {
  const [message, setMessage] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [showToolDialog, setShowToolDialog] = useState(false);


  const availableTools = [
    { id: 'web_search', name: 'Web Search' },
    { id: 'run_python_code', name: 'Code Runner' },
    // Add more tools as they are defined in the backend
  ];

  const handleToolToggle = (toolId: string) => {
    const newSelectedTools = selectedTools.includes(toolId)
      ? selectedTools.filter(id => id !== toolId)
      : [...selectedTools, toolId];
    setSelectedTools(newSelectedTools);
  };

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
        {/* Tool selection button */}
        <div className="absolute bottom-full left-0 mb-2 flex space-x-2">
          {/* Tool selection button - always visible now */}

          <button
            type="button"
            onClick={() => {
              setShowToolDialog(true);
              // Automatically enable tools when dialog is opened
            }}
            className="px-3 py-1 rounded-full text-sm font-medium bg-gray-200 text-gray-700 hover:bg-gray-300 dark:bg-dark-600 dark:text-gray-300 dark:hover:bg-dark-500 transition-colors"
            title="Select specific tools to use"
          >
            Select Tools
          </button>
        </div>

        {/* Show selected file as a chip above the input only in analyze mode */}
        {isAnalyzeMode && selectedFile && (
          <div className="flex items-center bg-gray-100 dark:bg-dark-600 px-3 py-2 rounded-full text-sm mb-2 ml-10 w-fit shadow-sm border border-gray-200 dark:border-dark-500 animate-fadeIn">
            {fileTypeIcon(selectedFile)}
            <span className="truncate max-w-[180px] font-medium">{selectedFile.name}</span>
            <button type="button" onClick={handleRemoveFile} className="ml-2 text-gray-500 hover:text-red-500 p-1 rounded-full hover:bg-gray-200 dark:hover:bg-dark-500 transition-colors">
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
            className="w-full p-4 pl-12 pr-16 rounded-full border border-gray-300 dark:border-dark-600 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-dark-700 text-gray-900 dark:text-gray-100 shadow-sm transition-all hover:shadow-md"
            disabled={isLoading}
          />
          {/* Upload icon only visible for file upload in analyze mode */}
          {isAnalyzeMode && (
            <>
              <button
                type="button"
                onClick={handleBrowseClick}
                className="absolute left-3 top-1/2 transform -translate-y-1/2 p-2 rounded-full bg-gray-100 hover:bg-primary-100 dark:bg-dark-600 dark:hover:bg-primary-900/30 transition-colors hover:shadow-sm"
                title="Upload document"
              >
                <DocumentIcon className="w-5 h-5 text-primary-600 dark:text-primary-400" />
              </button>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept={isTranscribeMode ? ".wav" : ".jpg,.jpeg,.png,.gif,.bmp,.pdf,.txt,.csv,.json,.md"}
                className="hidden"
              />
            </>
          )}
          <button
            type="submit"
            className={`absolute right-3 top-1/2 transform -translate-y-1/2 p-2.5 rounded-full ${isLoading ? 'bg-gray-300 dark:bg-dark-600 cursor-not-allowed' : 'bg-primary-600 hover:bg-primary-700 text-white shadow-sm hover:shadow-md'} transition-all`}
            disabled={isLoading || (isAnalyzeMode && selectedFile == null && !message.trim()) || (!isAnalyzeMode && !message.trim())}
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Tool Selection Dialog */}

    <Transition appear show={showToolDialog} as={Fragment}>
        <Dialog as="div" className="relative z-10" onClose={() => setShowToolDialog(false)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black bg-opacity-25" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4 text-center">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-md transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all dark:bg-dark-700">
                  <Dialog.Title
                    as="h3"
                    className="text-lg font-medium leading-6 text-gray-900 dark:text-gray-100"
                  >
                    Select Tools
                  </Dialog.Title>
                  <div className="mt-2">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Choose which tools the AI can use for this conversation.
                    </p>
                  </div>

                  <div className="mt-4 space-y-2">
                    {availableTools.map(tool => (
                      <label key={tool.id} className="flex items-center space-x-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={selectedTools.includes(tool.id)}
                          onChange={() => handleToolToggle(tool.id)}
                          className="form-checkbox h-5 w-5 text-primary-600 rounded dark:bg-dark-600 dark:border-dark-500"
                        />
                        <span className="text-gray-900 dark:text-gray-100">{tool.name}</span>
                      </label>
                    ))}
                  </div>

                  <div className="mt-4">
                    <button
                      type="button"
                      className="inline-flex justify-center rounded-md border border-transparent bg-primary-100 px-4 py-2 text-sm font-medium text-primary-900 hover:bg-primary-200 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary-500 focus-visible:ring-offset-2 dark:bg-primary-900 dark:text-primary-100 dark:hover:bg-primary-800"
                      onClick={() => setShowToolDialog(false)}
                    >
                      Done
                    </button>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </form>
  );
};

export default MessageInput;
