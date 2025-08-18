import React, { useState, useEffect } from 'react';
import Canvas from '../Canvas';
import ImageCanvas from '../ImageCanvas';
import { getChatMessages } from '../../services/api';
import './CanvasDisplay.css';

const canvasUpdateEvent = new CustomEvent('canvasUpdate');

window.updateCanvasDisplay = function () {
  document.dispatchEvent(canvasUpdateEvent);
};

const CanvasDisplay = ({ width, height, onClose, activeChat }) => {
  const [canvasMode, setCanvasMode] = useState('image');
  const [objectCoordinates, setObjectCoordinates] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [updateTrigger, setUpdateTrigger] = useState(0);

  const API_BASE_URL = 'http://localhost:8000';

  useEffect(() => {
    const handleCanvasUpdate = () => {
      setUpdateTrigger(prev => prev + 1);
    };

    document.addEventListener('canvasUpdate', handleCanvasUpdate);

    return () => {
      document.removeEventListener('canvasUpdate', handleCanvasUpdate);
    };
  }, []);

  useEffect(() => {
    const loadObjectData = async () => {
      if (activeChat) {
        try {
          console.log('Loading messages for chat ID:', activeChat.id);
          const messages = await getChatMessages(activeChat.id);
          console.log('Fetched messages:', messages);

          const lastImageMessage = [...messages]
            .reverse()
            .find(msg => (msg.sender === 'system' && msg.image_path) ||
              (msg.sender === 'system' && msg.text && msg.text.includes('Image uploaded')));

          const allObjectMarkMessages = messages
            .filter(msg => {
              const hasExplicitMark = (msg.is_object_mark === true || msg.object_coordinates) && msg.sender === 'ai';

              if (!hasExplicitMark && msg.sender === 'ai' && msg.text) {
                const containsBbox = msg.text.includes('"bbox"') || msg.text.includes("'bbox'");
                const containsLabel = msg.text.includes('"label"') || msg.text.includes("'label'");
                const containsJsonBraces = msg.text.includes('{') && msg.text.includes('}');

                return containsBbox && containsLabel && containsJsonBraces;
              }

              return hasExplicitMark;
            })
            .reverse();

          console.log('Latest image message:', lastImageMessage);
          console.log('Number of object mark messages found:', allObjectMarkMessages.length);

          if (lastImageMessage) {
            if (lastImageMessage.image_path) {
              const imagePath = lastImageMessage.image_path;
              const fullImageUrl = `${API_BASE_URL}${imagePath.startsWith('/') ? imagePath : '/' + imagePath}`;
              console.log('Setting image URL:', fullImageUrl);
              setImageUrl(fullImageUrl);
            } else {
              const systemMessages = messages.filter(msg => msg.sender === 'system');
              for (const msg of systemMessages) {
                if (msg.image_path) {
                  const imagePath = msg.image_path;
                  const fullImageUrl = `${API_BASE_URL}${imagePath.startsWith('/') ? imagePath : '/' + imagePath}`;
                  console.log('Found image URL from other system messages:', fullImageUrl);
                  setImageUrl(fullImageUrl);
                  break;
                }
              }
            }
          }

          if (allObjectMarkMessages.length > 0) {
            const allCoordinates = [];

            for (const markMessage of allObjectMarkMessages) {
              console.log('Processing mark message:', markMessage.id);

              let messageCoordinates = null;

              if (markMessage.object_coordinates) {
                messageCoordinates = markMessage.object_coordinates;
                console.log('Getting coordinates from object property');
              } else if (markMessage.is_object_mark) {
                const text = markMessage.text || '';

                try {
                  const jsonRegex = /(\{.*\}|\[.*\])/s;
                  const match = text.match(jsonRegex);
                  if (match) {
                    messageCoordinates = match[0];
                    console.log('Extracting coordinates from text');
                  }
                } catch (e) {
                  console.error('Error extracting coordinates from text:', e);
                }
              }

              if (messageCoordinates) {
                try {
                  let parsedCoordinates;

                  if (typeof messageCoordinates === 'string') {
                    try {
                      parsedCoordinates = JSON.parse(messageCoordinates);
                    } catch (jsonError) {
                      console.log('Direct string parse failed, attempting to extract JSON part');
                      const jsonRegex = /(\{.*\}|\[.*\])/s;
                      const match = messageCoordinates.match(jsonRegex);
                      if (match) {
                        parsedCoordinates = JSON.parse(match[0]);
                        console.log('Successfully extracted and parsed JSON part');
                      } else {
                        console.error('Could not extract JSON part');
                      }
                    }
                  } else {
                    parsedCoordinates = messageCoordinates;
                  }

                  if (Array.isArray(parsedCoordinates)) {
                    if (parsedCoordinates.length >= 4 && typeof parsedCoordinates[0] === 'number') {
                      allCoordinates.push({
                        bbox: parsedCoordinates,
                        label: `Object ${allCoordinates.length + 1}`
                      });
                    } else {
                      for (const coord of parsedCoordinates) {
                        if (coord) allCoordinates.push(coord);
                      }
                    }
                  } else if (parsedCoordinates && typeof parsedCoordinates === 'object') {
                    allCoordinates.push(parsedCoordinates);
                  }

                  console.log(`Added coordinate data, current total: ${allCoordinates.length}`);
                } catch (parseError) {
                  console.error('Failed to parse coordinate data:', parseError);
                }
              }
            }

            if (allCoordinates.length > 0) {
              console.log(`Setting merged ${allCoordinates.length} coordinates:`, allCoordinates);
              setObjectCoordinates(allCoordinates);
            } else {
              console.warn('No valid coordinate data found');
              setObjectCoordinates(null);
            }
          } else {
            console.log('No object mark messages found');
            setObjectCoordinates(null);
          }
        } catch (error) {
          console.error('Failed to load object data:', error);
        }
      }
    };

    loadObjectData();
  }, [activeChat, API_BASE_URL, updateTrigger]);

  return (
    <div className="canvas-display">
      <div className="canvas-content">
        {canvasMode === 'basic' ? (
          <Canvas width={width} height={height} />
        ) : (
          <ImageCanvas
            width={width}
            height={height}
            initialImageUrl={imageUrl}
            objectCoordinates={objectCoordinates}
          />
        )}
      </div>
    </div>
  );
};

export default CanvasDisplay;