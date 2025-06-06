import { useState, useRef, useEffect } from 'react'
import Header from '@components/Header'
import ChatInterface from '@components/ChatInterface'
import ModeSelector from '@components/ModeSelector'
import { ChatMode } from '@/types/chat'
import { ThemeProvider } from '@/context/ThemeContext'

function App() {
  const [selectedMode, setSelectedMode] = useState<ChatMode>('chat')
  const chatInterfaceRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom of chat interface when new messages are added
  useEffect(() => {
    if (chatInterfaceRef.current) {
      chatInterfaceRef.current.scrollTop = chatInterfaceRef.current.scrollHeight
    }
  }, [])

  return (
    <ThemeProvider>
      <div className="flex flex-col h-screen bg-gray-50 dark:bg-dark-800 text-gray-900 dark:text-gray-100">
        <Header />
        
        <main className="flex-1 container mx-auto px-4 py-6 overflow-hidden flex flex-col">
          <ModeSelector selectedMode={selectedMode} onModeChange={setSelectedMode} />
          
          <div className="flex-1 overflow-hidden mt-4">
            <ChatInterface 
              mode={selectedMode} 
              ref={chatInterfaceRef}
            />
          </div>
        </main>
        
        <footer className="py-4 text-center text-sm text-gray-500 dark:text-gray-400 border-t border-gray-200 dark:border-dark-600">
          <p>Powered by Groq & OpenRouter APIs</p>
        </footer>
      </div>
    </ThemeProvider>
  )
}

export default App