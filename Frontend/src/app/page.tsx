'use client'

import { useCallback, useState } from 'react'

interface Message {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');

  const sendMessage = useCallback(async () => {
    if (!inputValue.trim()) return;
    
    const userMessage: Message = {
      id: Date.now().toString(),
      text: inputValue.trim(),
      isUser: true,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    
    try {
      const response = await fetch('http://localhost:8000/api/v1/chat/message', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: inputValue.trim() })
      });
      
      const data = await response.json();
      
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: data.reply || 'Sorry, I could not process your request.',
        isUser: false,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        text: 'Sorry, there was an error connecting to the server.',
        isUser: false,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    }
  }, [inputValue]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      sendMessage();
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: '#0B0219', 
      color: 'white',
      display: 'flex',
      flexDirection: 'column'
    }}>
      {/* Header */}
      <header style={{ 
        height: '72px', 
        backgroundColor: 'rgba(11, 2, 25, 0.7)', 
        borderBottom: '1px solid rgba(255, 255, 255, 0.05)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 24px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ 
            height: '36px', 
            width: '36px', 
            borderRadius: '50%', 
            background: 'linear-gradient(135deg, #d946ef, #6366f1)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }} />
          <div style={{ 
            letterSpacing: '0.1em', 
            fontWeight: '600', 
            fontSize: '14px', 
            opacity: '0.9' 
          }}>AI ESTATE</div>
        </div>
      </header>

      {/* Chat Area */}
      <div style={{ 
        flex: 1, 
        overflowY: 'auto', 
        padding: '24px',
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {messages.length === 0 ? (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            alignItems: 'center', 
            height: '100%',
            flexDirection: 'column',
            gap: '16px'
          }}>
            <div style={{ 
              width: '80px', 
              height: '80px', 
              borderRadius: '50%', 
              background: 'linear-gradient(135deg, #d946ef, #6366f1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: '32px'
            }}>
              🤖
            </div>
            <h1 style={{ 
              fontSize: '32px', 
              fontWeight: '600', 
              margin: 0,
              textAlign: 'center'
            }}>
              Real-Estate Agent
            </h1>
            <p style={{ 
              color: 'rgba(255, 255, 255, 0.75)', 
              fontSize: '16px',
              textAlign: 'center',
              maxWidth: '500px',
              lineHeight: '1.5',
              margin: 0
            }}>
              Get comprehensive advice on various aspects of real estate, from legalities to client management, tailored to your needs.
            </p>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} style={{ 
              display: 'flex', 
              justifyContent: message.isUser ? 'flex-end' : 'flex-start' 
            }}>
              {message.isUser ? (
                <div style={{ 
                  backgroundColor: 'rgba(139, 92, 246, 0.7)', 
                  borderRadius: '16px', 
                  padding: '16px 24px', 
                  maxWidth: '500px' 
                }}>
                  <p style={{ color: 'white', fontSize: '14px', margin: 0 }}>{message.text}</p>
                </div>
              ) : (
                <div style={{ maxWidth: '500px' }}>
                  <p style={{ color: 'rgba(255, 255, 255, 0.85)', lineHeight: '1.75', margin: 0 }}>
                    {message.text}
                  </p>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Input Area */}
      <div style={{ 
        padding: '24px', 
        borderTop: '1px solid rgba(255, 255, 255, 0.05)' 
      }}>
        <div style={{ maxWidth: '800px', margin: '0 auto' }}>
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '12px', 
            borderRadius: '16px', 
            border: '1px solid rgba(255, 255, 255, 0.15)', 
            backgroundColor: 'rgba(255, 255, 255, 0.05)', 
            padding: '6px 16px',
            transition: 'border-color 0.3s ease'
          }}>
            <div style={{ 
              width: '20px', 
              height: '20px', 
              opacity: '0.6',
              color: 'white'
            }}>💬</div>
            <input 
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Message Chatbot.." 
              style={{ 
                flex: 1,
                background: 'transparent',
                border: 'none',
                outline: 'none',
                color: 'white',
                fontSize: '15px',
                padding: '12px 0'
              }}
            />
            <button 
              onClick={sendMessage}
              style={{ 
                width: '48px', 
                height: '48px', 
                borderRadius: '50%', 
                background: 'linear-gradient(135deg, #7c3aed, #a855f7)', 
                border: '1px solid rgba(255, 255, 255, 0.15)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'transform 0.3s ease'
              }}
              onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.03)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              <span style={{ color: 'white', fontSize: '20px' }}>→</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}