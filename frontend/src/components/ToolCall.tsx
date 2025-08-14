import React, { useState } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface ToolCallProps {
  toolCall: {
    id: string;
    function: {
      name: string;
      arguments: string;
    };
  };
  toolResult?: {
    tool_call_id: string;
    name: string;
    result?: string;
    error?: string;
  };
}

const ToolCall: React.FC<ToolCallProps> = ({ toolCall, toolResult }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  // Parse the arguments as JSON for display
  let parsedArgs = {};
  try {
    parsedArgs = JSON.parse(toolCall.function.arguments);
  } catch (e) {
    parsedArgs = { error: 'Could not parse arguments' };
  }
  
  // Determine if there was an error with the tool call
  const hasError = toolResult?.error !== undefined;
  
  return (
    <div className="tool-call border border-gray-200 dark:border-dark-600 rounded-lg mb-2 overflow-hidden">
      <div 
        className="tool-call-header flex items-center justify-between p-2 cursor-pointer bg-gray-50 dark:bg-dark-600"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center">
          <div className={`tool-call-badge px-2 py-0.5 rounded-full text-xs font-medium mr-2 ${hasError ? 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'}`}>
            {hasError ? 'Error' : 'Tool'}
          </div>
          <span className="font-medium">{toolCall.function.name}</span>
        </div>
        {isExpanded ? (
          <ChevronUpIcon className="w-4 h-4 text-gray-500" />
        ) : (
          <ChevronDownIcon className="w-4 h-4 text-gray-500" />
        )}
      </div>
      
      {isExpanded && (
        <div className="tool-call-details p-3 text-sm">
          <div className="mb-3">
            <h4 className="text-xs text-gray-500 dark:text-gray-400 mb-1">Arguments</h4>
            <pre className="bg-gray-100 dark:bg-dark-700 p-2 rounded-md overflow-auto text-xs">
              {JSON.stringify(parsedArgs, null, 2)}
            </pre>
          </div>
          
          {toolResult && (
            <div>
              <h4 className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                {hasError ? 'Error' : 'Result'}
              </h4>
              <pre className={`p-2 rounded-md overflow-auto text-xs ${hasError ? 'bg-red-50 dark:bg-red-900/10 text-red-700 dark:text-red-300' : 'bg-gray-100 dark:bg-dark-700'}`}>
                {hasError ? toolResult.error : toolResult.result}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ToolCall;