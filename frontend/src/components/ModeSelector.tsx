import { Fragment } from 'react';
import { Listbox, Transition } from '@headlessui/react';
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid';
import { ChatMode } from '@/types/chat';

interface ModeSelectorProps {
  selectedMode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
}

const modes: { id: ChatMode; name: string; description: string; icon: string }[] = [
  { id: 'chat', name: 'Chat', description: 'General conversation and Q&A', icon: '💬' },
  { id: 'code', name: 'Code', description: 'Programming help and code generation', icon: '💻' },
  { id: 'write', name: 'Write', description: 'Long-form writing, storytelling', icon: '✍️' },
  { id: 'brainstorm', name: 'Brainstorm', description: 'Idea generation and creative prompts', icon: '🧠' },
  { id: 'math', name: 'Math', description: 'Solve math problems with explanations', icon: '🔢' },
  { id: 'research', name: 'Research', description: 'Deep info gathering, summarization, Q&A', icon: '🔍' },
  { id: 'email', name: 'Email', description: 'Draft professional or casual emails', icon: '📧' },
  { id: 'text-to-speech', name: 'Text-to-Speech', description: 'Convert chatbot replies to audio', icon: '🔊' },
  { id: 'transcribe', name: 'Transcribe', description: 'Audio file (.wav) transcription', icon: '🎤' },
  { id: 'moderate', name: 'Moderate', description: 'Content safety and policy moderation', icon: '🔒' },
];

const ModeSelector = ({ selectedMode, onModeChange }: ModeSelectorProps) => {
  const selectedModeObj = modes.find(mode => mode.id === selectedMode) || modes[0];

  return (
    <div className="w-full max-w-md mx-auto">
      <Listbox value={selectedMode} onChange={onModeChange}>
        <div className="relative">
          <Listbox.Button className="relative w-full cursor-pointer rounded-lg bg-white dark:bg-dark-700 py-3 pl-3 pr-10 text-left shadow-md focus:outline-none focus-visible:border-primary-500 focus-visible:ring-2 focus-visible:ring-white/75 focus-visible:ring-offset-2 focus-visible:ring-offset-primary-300 sm:text-sm border border-gray-200 dark:border-dark-600">
            <div className="flex items-center">
              <span className="text-2xl mr-2">{selectedModeObj.icon}</span>
              <div>
                <span className="block truncate font-medium">{selectedModeObj.name}</span>
                <span className="block truncate text-xs text-gray-500 dark:text-gray-400">{selectedModeObj.description}</span>
              </div>
            </div>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon
                className="h-5 w-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </Listbox.Button>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Listbox.Options className="absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md bg-white dark:bg-dark-700 py-1 text-base shadow-lg ring-1 ring-black/5 focus:outline-none sm:text-sm">
              {modes.map((mode) => (
                <Listbox.Option
                  key={mode.id}
                  className={({ active }) =>
                    `relative cursor-pointer select-none py-2 pl-10 pr-4 ${active ? 'bg-primary-100 dark:bg-primary-900/20 text-primary-900 dark:text-primary-100' : 'text-gray-900 dark:text-gray-100'}`
                  }
                  value={mode.id}
                >
                  {({ selected }) => (
                    <>
                      <div className="flex items-center">
                        <span className="text-xl mr-2">{mode.icon}</span>
                        <div>
                          <span
                            className={`block truncate ${selected ? 'font-medium' : 'font-normal'}`}
                          >
                            {mode.name}
                          </span>
                          <span className="block truncate text-xs text-gray-500 dark:text-gray-400">
                            {mode.description}
                          </span>
                        </div>
                      </div>
                      {selected ? (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-600 dark:text-primary-400">
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
                        </span>
                      ) : null}
                    </>
                  )}
                </Listbox.Option>
              ))}
            </Listbox.Options>
          </Transition>
        </div>
      </Listbox>
    </div>
  );
};

export default ModeSelector;