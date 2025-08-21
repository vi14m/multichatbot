import { useState } from 'react';
import { Message as MessageType } from '@/types/chat';
import ReactMarkdown from 'react-markdown';
import { SpeakerWaveIcon, ClipboardIcon, CheckIcon, DocumentMagnifyingGlassIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import { convertTextToSpeech } from '@/services/api';
import ToolCall from './ToolCall';

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
        <div className={`rounded-lg p-4 shadow-sm ${isUser ? 'bg-primary-600 text-white' : 'bg-white dark:bg-dark-700 border border-gray-200 dark:border-dark-600'}`}>
          {message.fileAnalysis ? (
            <div className="file-analysis">
              <div className="flex items-center mb-2">
                <DocumentMagnifyingGlassIcon className="w-5 h-5 mr-2" />
                <span className="font-medium">File Analysis: {message.fileAnalysis.file_name || 'Unknown File'}</span>
              </div>
              
              <div className="file-analysis-details bg-gradient-to-r from-gray-50 to-white dark:from-dark-600 dark:to-dark-700 rounded-lg p-4 mb-4 text-sm border border-gray-200 dark:border-dark-500 shadow-sm">
                <div className="flex flex-wrap gap-3 mb-3">
                  <div className="file-type px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 rounded-full text-primary-700 dark:text-primary-300 font-medium shadow-sm">
                    {message.fileAnalysis.file_type.toUpperCase()}
                  </div>
                  
                  {message.fileAnalysis.processing_time && (
                    <div className="processing-time px-3 py-1.5 bg-gray-100 dark:bg-dark-700 rounded-full shadow-sm hover:shadow-md transition-shadow">
                      <span className="font-medium">Time:</span> {message.fileAnalysis.processing_time.toFixed(2)}s
                    </div>
                  )}
                  
                  {message.fileAnalysis.file_type === 'image' && message.fileAnalysis.metadata && (
                    <div className="image-dimensions px-3 py-1.5 bg-gray-100 dark:bg-dark-700 rounded-full shadow-sm hover:shadow-md transition-shadow">
                      <span className="font-medium">Size:</span> {message.fileAnalysis.metadata.width} × {message.fileAnalysis.metadata.height}
                    </div>
                  )}
                  
                  {message.fileAnalysis.file_type === 'pdf' && message.fileAnalysis.metadata && (
                    <div className="pdf-pages px-3 py-1.5 bg-gray-100 dark:bg-dark-700 rounded-full shadow-sm hover:shadow-md transition-shadow">
                      <span className="font-medium">Pages:</span> {message.fileAnalysis.metadata.num_pages}
                    </div>
                  )}
                </div>
                
                <p className="text-gray-600 dark:text-gray-300 text-sm bg-blue-50 dark:bg-blue-900/10 p-2 rounded-md border border-blue-100 dark:border-blue-800/30 flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                  </svg>
                  Analysis complete. You can now ask questions about this document.
                </p>
              </div>
              
              {/* AI Analysis Section */}
              {message.fileAnalysis.ai_analysis && (
                <div className="ai-analysis mb-3">
                  <button 
                    onClick={() => setShowAiAnalysis(!showAiAnalysis)}
                    className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 hover:text-primary-600 dark:hover:text-primary-400 transition-colors bg-gray-100 dark:bg-dark-600 px-3 py-2 rounded-md w-full justify-between group"
                  >
                    <span className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-primary-500" viewBox="0 0 20 20" fill="currentColor">
                        <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                        <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
                      </svg>
                      AI Analysis
                    </span>
                    <span className="bg-primary-100 dark:bg-primary-900/30 rounded-full p-1 group-hover:bg-primary-200 dark:group-hover:bg-primary-800/30 transition-colors">
                      {showAiAnalysis ? (
                        <ChevronUpIcon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                      ) : (
                        <ChevronDownIcon className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                      )}
                    </span>
                  </button>
                  
                  {showAiAnalysis && (
                    <div className="bg-white dark:bg-dark-700 rounded-md p-4 border border-gray-200 dark:border-dark-600 shadow-sm animate-fadeIn">
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
                    className="flex items-center text-sm font-medium text-gray-700 dark:text-gray-300 mb-2 hover:text-primary-600 dark:hover:text-primary-400 transition-colors bg-gray-100 dark:bg-dark-600 px-3 py-2 rounded-md w-full justify-between group"
                  >
                    <span className="flex items-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-secondary-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                      </svg>
                      Extracted Text
                    </span>
                    <span className="bg-secondary-100 dark:bg-secondary-900/30 rounded-full p-1 group-hover:bg-secondary-200 dark:group-hover:bg-secondary-800/30 transition-colors">
                      {showExtractedText ? (
                        <ChevronUpIcon className="w-4 h-4 text-secondary-600 dark:text-secondary-400" />
                      ) : (
                        <ChevronDownIcon className="w-4 h-4 text-secondary-600 dark:text-secondary-400" />
                      )}
                    </span>
                  </button>
                  
                  {showExtractedText && (
                    <div className="bg-white dark:bg-dark-700 rounded-md p-4 border border-gray-200 dark:border-dark-600 max-h-60 overflow-y-auto shadow-sm animate-fadeIn">
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
              
              {/* Consistency Check Information - Display when available */}
              {message.consistencyInfo && (
                <div className="consistency-info mt-4 pt-4 border-t border-gray-200 dark:border-dark-600">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-green-500" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Consistency Check
                  </h4>
                  <div className="bg-green-50 dark:bg-green-900/10 rounded-md p-3 border border-green-100 dark:border-green-800/30 mb-3">
                    <div className="text-sm text-gray-700 dark:text-gray-300">
                      <div className="font-medium mb-1">{message.consistencyInfo.method === 'comparison' ? 'Response Comparison' : 'Self-Reflection'}</div>
                      <div className="text-gray-600 dark:text-gray-400">
                        {message.consistencyInfo.method === 'comparison' 
                          ? 'Two responses were generated and compared for consistency.' 
                          : 'The response was analyzed through self-reflection.'}
                      </div>
                    </div>
                  </div>
                  
                  {message.consistencyInfo.method === 'comparison' && message.consistencyInfo.similarity_score && (
                    <div className="mb-3">
                      <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Similarity Score</div>
                      <div className="w-full bg-gray-200 dark:bg-dark-600 rounded-full h-2.5">
                        <div 
                          className={`h-2.5 rounded-full ${message.consistencyInfo.similarity_score > 0.8 ? 'bg-green-500' : message.consistencyInfo.similarity_score > 0.5 ? 'bg-yellow-500' : 'bg-red-500'}`}
                          style={{ width: `${Math.round(message.consistencyInfo.similarity_score * 100)}%` }}
                        ></div>
                      </div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 text-right">
                        {Math.round(message.consistencyInfo.similarity_score * 100)}% similar
                      </div>
                    </div>
                  )}
                  
                  {message.consistencyInfo.differences && message.consistencyInfo.differences.length > 0 && (
                    <div className="mb-3">
                      <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Key Differences</div>
                      <ul className="list-disc list-inside text-sm text-gray-600 dark:text-gray-400">
                        {message.consistencyInfo.differences.map((diff, index) => (
                          <li key={index}>{diff}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {message.consistencyInfo.reflection && (
                    <div className="mb-3">
                      <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Reflection</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400 bg-white dark:bg-dark-700 p-3 rounded-md border border-gray-200 dark:border-dark-600">
                        <ReactMarkdown>
                          {message.consistencyInfo.reflection}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Tool Calls Section - Only display if there's no main content */}
              {!message.content && message.toolCalls && message.toolCalls.length > 0 && (
                <div className="tool-calls mt-4 pt-4 border-t border-gray-200 dark:border-dark-600">
                  <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3 flex items-center">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-primary-500" viewBox="0 0 20 20" fill="currentColor">
                      <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd" />
                    </svg>
                    Tool Calls
                  </h4>
                  <div className="space-y-3">
                    {message.toolCalls.map((toolCall) => (
                      <ToolCall 
                        key={toolCall.id}
                        toolCall={toolCall}
                        toolResult={message.toolResults?.find(result => result.tool_call_id === toolCall.id)}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          </div>
          
          <div className="flex items-center mt-2 text-xs text-gray-500 dark:text-gray-400">
          <span className="bg-gray-100 dark:bg-dark-600 px-2 py-1 rounded-full">{formattedTime}</span>
          
          {!isUser && (
            <div className="flex ml-3 space-x-3">
              <button 
                onClick={handleCopyToClipboard}
                className="flex items-center hover:text-gray-700 dark:hover:text-gray-300 transition-colors bg-gray-100 dark:bg-dark-600 p-1.5 rounded-full hover:bg-gray-200 dark:hover:bg-dark-500"
                aria-label="Copy to clipboard"
                title="Copy to clipboard"
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
                className={`flex items-center hover:text-gray-700 dark:hover:text-gray-300 transition-colors bg-gray-100 dark:bg-dark-600 p-1.5 rounded-full hover:bg-gray-200 dark:hover:bg-dark-500 ${isLoadingAudio ? 'opacity-50 cursor-not-allowed' : ''}`}
                aria-label="Text to speech"
                title="Text to speech"
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