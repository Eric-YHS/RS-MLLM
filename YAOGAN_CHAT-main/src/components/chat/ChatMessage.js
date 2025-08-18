import React, { useState, useEffect } from 'react';
import Markdown from './Markdown';
import './ChatMessage.css';

const ChatMessage = ({ message, onViewInCanvas }) => {
  const { sender, text, error, image, thinking, isObjectMark, objectCoordinates } = message;
  const [showThinking, setShowThinking] = useState(false);

  const isUser = sender === 'user';
  const isSystem = sender === 'system';

  const getAvatar = () => {
    if (isUser) return <div className="user-avatar">U</div>;
    if (isSystem) return <div className="system-avatar">S</div>;
    return <div className="ai-avatar">AI</div>;
  };

  const formatImageUrl = (url) => {
    if (!url) return '';

    if (url.startsWith('http') || url.startsWith('blob:')) {
      return url;
    }

    const API_BASE_URL = 'http://localhost:8000';

    if (!url.startsWith('/')) {
      url = '/' + url;
    }

    console.log('Formatting image URL:', `${API_BASE_URL}${url}`);
    return `${API_BASE_URL}${url}`;
  };

  const toggleThinking = () => {
    setShowThinking(!showThinking);
  };

  useEffect(() => {
    if (isObjectMark && objectCoordinates && window.updateCanvasDisplay) {
      console.log('Object mark message loaded, automatically updating canvas state');
      window.updateCanvasDisplay();
    }
  }, [isObjectMark, objectCoordinates]);

  return (
    <div className={`message-wrapper ${isUser ? 'user-message' : isSystem ? 'system-message' : 'ai-message'
      }`}>
      <div className="avatar">
        {getAvatar()}
      </div>

      <div className={`message-bubble ${error ? 'error-message' : ''}`}>
        {image && (
          <div className="message-image-container">
            <img src={formatImageUrl(image)} alt="Uploaded image" className="message-image" />
          </div>
        )}

        <div className="message-content">
          {isObjectMark ? "Marking completed on canvas" : <Markdown content={text} />}
        </div>

        {isObjectMark && onViewInCanvas && (
          <div className="canvas-view-section">
            <button
              className="canvas-view-button toggle-canvas-button"
              onClick={() => {
                if (window.updateCanvasDisplay) {
                  window.updateCanvasDisplay();
                }
                onViewInCanvas();
              }}
            >
              View in Canvas
            </button>
          </div>
        )}

        {thinking && (
          <div className="thinking-section">
            <button
              className="thinking-toggle"
              onClick={toggleThinking}
            >
              {showThinking ? 'Hide thinking process' : 'View thinking process'}
            </button>

            {showThinking && (
              <div className="thinking-content">
                <pre>{thinking}</pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatMessage;