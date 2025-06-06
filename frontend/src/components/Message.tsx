import { useState } from 'react';
import { Message as MessageType } from '@/types/chat';
import ReactMarkdown from 'react-markdown';
import { SpeakerWaveIcon, ClipboardIcon, CheckIcon } from '@heroicons/react/24/outline';
import { convertTextToSpeech } from '@/services/api';

interface MessageProps {
  message: MessageType;
}

const Message = ({ message }: MessageProps) => {
  const [isCopied, setIsCopied] = useState(false);
  const [isPlayingAudio, setIsPlayingAudio] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(message.audioUrl || null);
  const [isLoadingAudio, setIsLoadingAudio] = useState(false);
  
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
          <div className="message-content">
            <ReactMarkdown>
              {message.content}
            </ReactMarkdown>
          </div>
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