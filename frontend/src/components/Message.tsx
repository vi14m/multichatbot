import { useState } from 'react';
import { Message as MessageType } from '@/types/chat';
import ReactMarkdown from 'react-markdown';
import { SpeakerWaveIcon, ClipboardIcon, CheckIcon, DocumentMagnifyingGlassIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { convertTextToSpeech } from '@/services/api';

interface MessageProps {
  message: MessageType;
}

const Message = ({ message }: MessageProps) => {
  const [isCopied, setIsCopied] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(message.audioUrl || null);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  const [showExtractedText, setShowExtractedText] = useState(false);
  const [showAiAnalysis, setShowAiAnalysis] = useState(true);
  
  const isUser = message.role === 'user';
  const formattedTime = new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: 'numeric',
  }).format(new Date(message.timestamp));

  const handleCopyToClipboard = () => {
    navigator.clipboard.writeText(message.content);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleTextToSpeech = async () => {
    if (audioUrl) {
      // If we already have audio, just play it
      const audio = new Audio(audioUrl);
      audio.onplay = () => setIsPlayingAudio(true);
      audio.onended = () => setIsPlayingAudio(false);
      audio.play();
    } else {
      // Otherwise, generate new audio
      try {
        setIsLoadingAudio(true);
        const audioBlob = await convertTextToSpeech({ text: message.content });
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        
        const audio = new Audio(url);
        audio.onplay = () => setIsPlayingAudio(true);
        audio.onended = () => setIsPlayingAudio(false);
        audio.play();
      } catch (error) {
        console.error('Error generating audio:', error);
      } finally {
        setIsLoadingAudio(false);
      }
    }
  };

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div className={`max-w-[80%] ${isUser ? 'order-2' : 'order-1'}`}>
        <div className={`rounded-lg p-4 ${isUser ? 'bg-primary-600 text-white' : 'bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600'}`}>
          {message.fileAnalysis ? (
            <div className="file-analysis">
              <div className="flex items-center mb-2">
                <DocumentMagnifyingGlassIcon className="w-5 h-5 mr-2" />
                <span className="font-medium">File Analysis: {message.fileAnalysis.file_name || 'Unknown File'}</span>
              </div>
              
              <div className="file-analysis-details bg-gray-50 dark:bg-dark-600 rounded p-3 mb-4 text-sm border border-gray-200 dark:border-dark-500">
                <div className="flex flex-wrap gap-3 mb-2">
                  <div className="file-type px-2 py-1 bg-primary-100 dark:bg-primary-900/30 rounded-full text-primary-700 dark:text-primary-300 font-medium">
                    {message.fileAnalysis.file_type.toUpperCase()}
                  </div>
                  
                  {message.fileAnalysis.processing_time && (
                    <div className="processing-time px-2 py-1 bg-gray-100 dark:bg-dark-700 rounded-full">
                      <span className="font-medium">Time:</span> {message.fileAnalysis.processing_time.toFixed(2)}s
                    </div>
                  )}
                  
                  {message.fileAnalysis.file_type === 'image' && message.fileAnalysis.metadata && (
                    <div className="image-dimensions px-2 py-1 bg-gray-100 dark:bg-dark-700 rounded-full">
                      <span className="font-medium">Size:</span> {message.fileAnalysis.metadata.width} × {message.fileAnalysis.metadata.height}
                    </div>
                  )}
                  
                  {message.fileAnalysis.file_type === 'pdf' && message.fileAnalysis.metadata && (
                    <div className="pdf-pages px-2 py-1 bg-gray-100 dark:bg-dark-700 rounded-full">
                      <span className="font-medium">Pages:</span> {message.fileAnalysis.metadata.num_pages}
                    </div>
                  )}
                </div>
                
                <p className="text-gray-600 dark:text-gray-300 text-xs italic">
                  Analysis complete. You can now ask questions about this document.
                </p>
              </div>
              
              {/* AI Analysis Section */}
              {message.fileAnalysis.ai_analysis && (
                <div className="ai-analysis mb-3">
                  <button 
                    onClick={() => setShowAiAnalysis(!showAiAnalysis)}
                    className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {showAiAnalysis ? (
                      <ChevronUpIcon className="w-4 h-4 mr-1" />
                    ) : (
                      <ChevronDownIcon className="w-4 h-4 mr-1" />
                    )}
                    AI Analysis
                  </button>
                  
                  {showAiAnalysis && (
                    <div className="bg-white dark:bg-dark-700 rounded p-3 border border-gray-200 dark:border-dark-600">
                      <ReactMarkdown>
                        {message.fileAnalysis.ai_analysis}
                      </ReactMarkdown>
                    </div>
                  )}
                </div>
              )}
              
              {/* Extracted Text Section */}
              {message.fileAnalysis.extracted_text && (
                <div className="extracted-text mb-3">
                  <button 
                    onClick={() => setShowExtractedText(!showExtractedText)}
                    className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 hover:text-primary-600 dark:hover:text-primary-400 transition-colors"
                  >
                    {showExtractedText ? (
                      <ChevronUpIcon className="w-4 h-4 mr-1" />
                    ) : (
                      <ChevronDownIcon className="w-4 h-4 mr-1" />
                    )}
                    Extracted Text
                  </button>
                  
                  {showExtractedText && (
                    <div className="bg-white dark:bg-dark-700 rounded p-3 border border-gray-200 dark:border-dark-600 max-h-60 overflow-y-auto">
                      <pre className="text-xs whitespace-pre-wrap">
                        {message.fileAnalysis.extracted_text}
                      </pre>
                    </div>
                  )}
                </div>
              )}
              
              {/* Main Content */}
              <div className="message-content">
                <ReactMarkdown>
                  {message.content}
                </ReactMarkdown>
              </div>
            </div>
          ) : (
            <div className="message-content">
              <ReactMarkdown>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>
        
        <div className="flex items-center mt-1 text-xs text-gray-500 dark:text-gray-400">
          <span>{formattedTime}</span>
          
          {!isUser && (
            <div className="flex ml-2 space-x-2">
              <button 
                onClick={handleCopyToClipboard}
                className="flex items-center hover:text-gray-700 dark:hover:text-gray-300 transition-colors"
                aria-label="Copy to clipboard"
              >
                {isCopied ? (
                  <CheckIcon className="w-4 h-4 text-green-500" />
                ) : (
                  <ClipboardIcon className="w-4 h-4" />
                )}
              </button>
              
              <button 
                onClick={handleTextToSpeech}
                disabled={isLoadingAudio}
                className={`flex items-center hover:text-gray-700 dark:hover:text-gray-300 transition-colors ${isLoadingAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
                aria-label="Text to speech"
              >
                <SpeakerWaveIcon className={`w-4 h-4 ${isPlayingAudio ? 'text-primary-500' : ''}`} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Message;