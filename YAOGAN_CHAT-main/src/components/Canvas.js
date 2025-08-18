import React, { useState } from 'react';
import { Stage, Layer, Rect, Circle } from 'react-konva';

const Canvas = ({ width, height }) => {
  const [selectedId, selectShape] = useState(null);
  const [shapes, setShapes] = useState([]);

  const addShape = (type) => {
    const newShape = {
      id: Date.now().toString(),
      type,
      x: Math.random() * width * 0.8,
      y: Math.random() * height * 0.8,
      width: 100,
      height: 100,
      radius: 50,
      fill: getRandomColor(),
      isDragging: false,
    };

    setShapes([...shapes, newShape]);
  };

  const getRandomColor = () => {
    const letters = '0123456789ABCDEF';
    let color = '#';
    for (let i = 0; i < 6; i++) {
      color += letters[Math.floor(Math.random() * 16)];
    }
    return color;
  };

  const checkDeselect = (e) => {
    const clickedOnEmpty = e.target === e.target.getStage();
    if (clickedOnEmpty) {
      selectShape(null);
    }
  };

  return (
    <div className="canvas-container">
      <div className="canvas-tools">
        <button onClick={() => addShape('rect')}>Add Rectangle</button>
        <button onClick={() => addShape('circle')}>Add Circle</button>
        {selectedId && <button onClick={() => {
          setShapes(shapes.filter(shape => shape.id !== selectedId));
          selectShape(null);
        }}>Delete Selected</button>}
      </div>

      <Stage
        width={width}
        height={height}
        onMouseDown={checkDeselect}
        onTouchStart={checkDeselect}
        style={{ border: '1px solid #ccc' }}
      >
        <Layer>
          {shapes.map((shape) => {
            const isSelected = shape.id === selectedId;

            if (shape.type === 'rect') {
              return (
                <Rect
                  key={shape.id}
                  id={shape.id}
                  x={shape.x}
                  y={shape.y}
                  width={shape.width}
                  height={shape.height}
                  fill={shape.fill}
                  stroke={isSelected ? '#0078d7' : null}
                  strokeWidth={isSelected ? 2 : 0}
                  draggable
                  onClick={() => selectShape(shape.id)}
                  onTap={() => selectShape(shape.id)}
                  onDragStart={() => {
                    const updatedShapes = shapes.map(s => {
                      if (s.id === shape.id) {
                        return { ...s, isDragging: true };
                      }
                      return s;
                    });
                    setShapes(updatedShapes);
                  }}
                  onDragEnd={() => {
                    const updatedShapes = shapes.map(s => {
                      if (s.id === shape.id) {
                        return { ...s, isDragging: false };
                      }
                      return s;
                    });
                    setShapes(updatedShapes);
                  }}
                />
              );
            } else if (shape.type === 'circle') {
              return (
                <Circle
                  key={shape.id}
                  id={shape.id}
                  x={shape.x}
                  y={shape.y}
                  radius={shape.radius}
                  fill={shape.fill}
                  stroke={isSelected ? '#0078d7' : null}
                  strokeWidth={isSelected ? 2 : 0}
                  draggable
                  onClick={() => selectShape(shape.id)}
                  onTap={() => selectShape(shape.id)}
                  onDragStart={() => {
                    const updatedShapes = shapes.map(s => {
                      if (s.id === shape.id) {
                        return { ...s, isDragging: true };
                      }
                      return s;
                    });
                    setShapes(updatedShapes);
                  }}
                  onDragEnd={() => {
                    const updatedShapes = shapes.map(s => {
                      if (s.id === shape.id) {
                        return { ...s, isDragging: false };
                      }
                      return s;
                    });
                    setShapes(updatedShapes);
                  }}
                />
              );
            }
            return null;
          })}
        </Layer>
      </Stage>
    </div>
  );
};

export default Canvas;