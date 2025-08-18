import React, { useState, useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { uploadAndAnalyzeImage, pollTaskResult, getChatMessages, processTextMessageAsync, cancelTask, updateChatTitle } from '../../services/api';
import './ChatContainer.css';
import './FunctionButtons.css';

const API_BASE_URL = 'http://localhost:8000';

const ChatContainer = ({ activeChat, onFunctionSelect }) => {
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [currentImage, setCurrentImage] = useState(null);
  const [markObjectMode, setMarkObjectMode] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const [imageUploaded, setImageUploaded] = useState(false);
  const [currentTaskId, setCurrentTaskId] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  useEffect(() => {
    const loadChatMessages = async () => {
      if (activeChat) {
        setIsTyping(true);
        try {
          const chatMessages = await getChatMessages(activeChat.id);
          if (chatMessages && chatMessages.length > 0) {
            const formattedMessages = chatMessages.map(msg => ({
              id: msg.id,
              text: msg.text,
              sender: msg.sender,
              timestamp: new Date(msg.timestamp),
              image: msg.image_path ? `${API_BASE_URL}${msg.image_path}` : null,
              thinking: msg.thinking,
              error: msg.error,
              objectCoordinates: msg.object_coordinates,
              isObjectMark: msg.is_object_mark,
              serverMessage: true
            }));

            setMessages(formattedMessages);

            const lastImageMessage = [...formattedMessages]
              .reverse()
              .find(msg => msg.sender === 'system' && msg.image);

            if (lastImageMessage) {
              setCurrentImage({
                isExisting: true,
                url: lastImageMessage.image
              });
              setImageUploaded(true);
            } else {
              setCurrentImage(null);
              setImageUploaded(false);
            }
          } else {
            setMessages([{
              id: 'welcome',
              text: 'Hello, welcome to the YAOGAN Chat System! Please start a conversation or upload a remote sensing image for analysis.',
              sender: 'ai',
              timestamp: new Date()
            }]);
          }
        } catch (error) {
          console.error('Failed to load chat messages:', error);
          if (error.message !== 'AUTHENTICATION_FAILED') {
            setMessages([{
              id: 'error',
              text: `Failed to load messages: ${error.message}`,
              sender: 'system',
              error: true,
              timestamp: new Date()
            }]);
          }
        } finally {
          setIsTyping(false);
        }
      } else {
        setMessages([]);
      }
    };

    loadChatMessages();
  }, [activeChat]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setCurrentImage(file);
    setImageUploaded(true);

    const blobUrl = URL.createObjectURL(file);
    console.log('Created local Blob URL:', blobUrl);

    const imageMessage = {
      id: Date.now(),
      text: 'Image uploaded. Please enter the question you want to ask.',
      sender: 'system',
      image: blobUrl,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, imageMessage]);
  };

  const handleUploadButtonClick = () => {
    fileInputRef.current.click();
  };

  const handleSendMessage = async (text) => {
    if (!text.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: markObjectMode ? `Mark object: ${text}` : text,
      sender: 'user',
      timestamp: new Date()
    };

    if (markObjectMode) {
      setMarkObjectMode(false);
    }

    const isFirstUserMessage = !messages.some(msg => msg.sender === 'user');
    if (isFirstUserMessage && activeChat) {
      try {
        await updateChatTitle(activeChat.id, text);
        if (window.updateChatTitle) {
          window.updateChatTitle(activeChat.id, text);
        }
      } catch (error) {
        console.error('Failed to update chat title:', error);
      }
    }

    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);
    setIsGenerating(true);

    let processingMessageId = null;

    try {
      if (currentImage) {
        const processingMessage = {
          id: Date.now() + 1,
          text: 'Analyzing image, please wait...',
          sender: 'system',
          timestamp: new Date()
        };

        processingMessageId = processingMessage.id;
        setMessages(prev => [...prev, processingMessage]);

        let response;

        if (currentImage.isExisting) {
          console.log('Analyzing with existing uploaded image...');

          const taskType = markObjectMode ? 'mark_object' : 'description';
          console.log(`Using task type: ${taskType}`);
          response = await processTextMessageAsync(text, activeChat.id, taskType);
          console.log('Text processing task submitted successfully, got task ID:', response.task_id);

          setCurrentTaskId(response.task_id);

          console.log('Starting to poll for task result...');
          const result = await pollTaskResult(
            response.task_id,
            3000,
            30,
            (attempts, maxAttempts) => {
              console.log(`Polling progress: ${attempts}/${maxAttempts}`);
              if (processingMessageId) {
                setMessages(prev => prev.map(msg =>
                  msg.id === processingMessageId
                    ? { ...msg, text: `Analyzing image, please wait... (${attempts}/${maxAttempts})` }
                    : msg
                ));
              }
            }
          );

          setCurrentTaskId(null);
          console.log('Polling complete, got result:', result);

          const aiMessage = {
            id: Date.now() + 2,
            text: result.result || 'Analysis complete, but no result was returned',
            sender: 'ai',
            timestamp: new Date(),
            thinking: result.thinking,
            objectCoordinates: result.object_coordinates,
            isObjectMark: result.is_object_mark
          };

          setMessages(prev => prev.filter(msg => msg.id !== processingMessageId).concat([aiMessage]));

          if (result.is_object_mark || result.object_coordinates) {
            console.log('Object mark message added, triggering canvas update');
            if (window.updateCanvasDisplay) {
              window.updateCanvasDisplay();
            }
          }
        } else {
          console.log('Starting image upload...');
          const taskType = markObjectMode ? 'detection' : 'description';
          console.log(`Using task type: ${taskType}`);
          response = await uploadAndAnalyzeImage(currentImage, text, taskType, activeChat?.id);
          console.log('Image upload successful, got task ID:', response.task_id);

          setCurrentTaskId(response.task_id);

          console.log('Starting to poll for task result...');
          const result = await pollTaskResult(
            response.task_id,
            3000,
            30,
            (attempts, maxAttempts) => {
              console.log(`Polling progress: ${attempts}/${maxAttempts}`);
              if (processingMessageId) {
                setMessages(prev => prev.map(msg =>
                  msg.id === processingMessageId
                    ? { ...msg, text: `Analyzing image, please wait... (${attempts}/${maxAttempts})` }
                    : msg
                ));
              }
            }
          );

          setCurrentTaskId(null);
          console.log('Polling complete, got result:', result);

          const aiMessage = {
            id: Date.now() + 2,
            text: result.result || 'Analysis complete, but no result was returned',
            sender: 'ai',
            timestamp: new Date(),
            thinking: result.thinking,
            objectCoordinates: result.object_coordinates || (markObjectMode ? result.result : null),
            isObjectMark: result.is_object_mark || markObjectMode
          };

          setMessages(prev => prev.filter(msg => msg.id !== processingMessageId).concat([aiMessage]));

          if (markObjectMode && window.updateCanvasDisplay) {
            console.log('Object marking mode detected, triggering canvas update with a delay');
            setTimeout(() => {
              window.updateCanvasDisplay();
            }, 1000);
          }

          if (response && response.chat_id) {
            const chatMessages = await getChatMessages(response.chat_id);
            const imageMessage = chatMessages.find(msg => msg.image_path && msg.sender === 'system');

            if (imageMessage) {
              let imagePath = imageMessage.image_path;
              if (imagePath && !imagePath.startsWith('/')) {
                imagePath = '/' + imagePath;
              }

              setCurrentImage({
                isExisting: true,
                url: `${API_BASE_URL}${imagePath}`
              });
              console.log('Image URL set to:', `${API_BASE_URL}${imagePath}`);
            }
          }
        }
      } else {
        if (activeChat) {
          try {
            const processingMessage = {
              id: Date.now() + 1,
              text: 'Processing your request, please wait...',
              sender: 'system',
              timestamp: new Date()
            };

            processingMessageId = processingMessage.id;
            setMessages(prev => [...prev, processingMessage]);

            const taskType = markObjectMode ? 'mark_object' : 'description';
            const response = await processTextMessageAsync(text, activeChat.id, taskType);
            console.log('Text processing task submitted successfully, got task ID:', response.task_id);

            setCurrentTaskId(response.task_id);

            console.log('Starting to poll for task result...');
            const result = await pollTaskResult(
              response.task_id,
              3000,
              30,
              (attempts, maxAttempts) => {
                console.log(`Polling progress: ${attempts}/${maxAttempts}`);
                if (processingMessageId) {
                  setMessages(prev => prev.map(msg =>
                    msg.id === processingMessageId
                      ? { ...msg, text: `Processing your request, please wait... (${attempts}/${maxAttempts})` }
                      : msg
                  ));
                }
              }
            );

            setCurrentTaskId(null);
            console.log('Polling complete, got result:', result);

            const aiMessage = {
              id: Date.now() + 2,
              text: result.result || 'Analysis complete, but no result was returned',
              sender: 'ai',
              timestamp: new Date(),
              thinking: result.thinking,
              objectCoordinates: result.object_coordinates,
              isObjectMark: result.is_object_mark
            };

            setMessages(prev => prev.filter(msg => msg.id !== processingMessageId).concat([aiMessage]));

            if (result.is_object_mark || result.object_coordinates) {
              console.log('Object mark message added, triggering canvas update');
              if (window.updateCanvasDisplay) {
                window.updateCanvasDisplay();
              }
            }
          } catch (error) {
            if (processingMessageId) {
              setMessages(prev => prev.filter(msg => msg.id !== processingMessageId));
            }

            const aiMessage = {
              id: Date.now() + 1,
              text: `Please upload a remote sensing image first so I can analyze it.`,
              sender: 'ai',
              timestamp: new Date()
            };
            setMessages(prev => [...prev, aiMessage]);
          }
        } else {
          const aiMessage = {
            id: Date.now() + 1,
            text: `Please upload a remote sensing image first so I can analyze it.`,
            sender: 'ai',
            timestamp: new Date()
          };
          setMessages(prev => [...prev, aiMessage]);
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);

      if (processingMessageId) {
        setMessages(prev => prev.filter(msg => msg.id !== processingMessageId));
      }

      const errorMessage = error.message || JSON.stringify(error);
      const isCanceled = errorMessage.includes('User canceled') || errorMessage.includes('Task canceled');

      if (!isCanceled) {
        setMessages(prev => [...prev, {
          id: Date.now() + 1,
          text: `Sorry, an error occurred while processing your message: ${errorMessage}`,
          sender: 'ai',
          error: true,
          timestamp: new Date()
        }]);
      }
    } finally {
      setIsTyping(false);
      setIsGenerating(false);
      setCurrentTaskId(null);
    }
  };

  const handleCancelGeneration = async () => {
    if (currentTaskId) {
      try {
        console.log(`Canceling task: ${currentTaskId}`);
        await cancelTask(currentTaskId);
        console.log(`Task canceled: ${currentTaskId}`);
      } catch (error) {
        console.error('Error during cancellation:', error);
      }
    }
  };

  return (
    <div className="chat-container">
      <div className="messages-container">
        {messages.map(message => (
          <ChatMessage
            key={message.id}
            message={message}
            onViewInCanvas={message.isObjectMark ? () => onFunctionSelect('canvas') : undefined}
          />
        ))}
        {isTyping && (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <div className="function-controls">
          <div className="function-buttons">
            <button
              className="function-button toggle-canvas-button"
              onClick={() => {
                if (window.updateCanvasDisplay) {
                  window.updateCanvasDisplay();
                }
                onFunctionSelect('canvas');
              }}
            >
              Show Canvas
            </button>
            <button
              className="function-button"
              onClick={handleUploadButtonClick}
              title="Upload remote sensing image"
              disabled={imageUploaded}
            >
              Upload Image
            </button>
            {!markObjectMode && (
              <button
                className={`function-button ${markObjectMode ? 'active' : ''}`}
                onClick={() => setMarkObjectMode(true)}
                title="Mark an object"
                disabled={!currentImage}
              >
                Mark Object
              </button>
            )}
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept="image/*"
              onChange={handleImageUpload}
            />
          </div>
        </div>
        <ChatInput
          onSendMessage={handleSendMessage}
          placeholder={
            markObjectMode
              ? "Please enter the name of the object to mark..."
              : currentImage
                ? "Please enter a question about the image..."
                : "Enter a message..."
          }
          markObjectMode={markObjectMode}
          onCancelMarkObject={() => setMarkObjectMode(false)}
          isGenerating={isGenerating}
          onCancelGeneration={handleCancelGeneration}
        />
      </div>
    </div>
  );
};

export default ChatContainer;