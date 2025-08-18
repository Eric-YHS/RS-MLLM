import React, { useState, useRef, useEffect } from 'react';
import { Stage, Layer, Image, Transformer, Rect, Text } from 'react-konva';
import Konva from 'konva';

const DraggableImage = ({ image, isSelected, onSelect, onChange }) => {
  const shapeRef = useRef();
  const trRef = useRef();

  useEffect(() => {
    if (isSelected) {
      trRef.current.nodes([shapeRef.current]);
      trRef.current.getLayer().batchDraw();
    }
  }, [isSelected]);

  return (
    <>
      <Image
        ref={shapeRef}
        image={image.imageObj}
        x={image.x}
        y={image.y}
        width={image.width}
        height={image.height}
        rotation={image.rotation}
        draggable
        onClick={onSelect}
        onTap={onSelect}
        onDragEnd={(e) => {
          onChange({
            ...image,
            x: e.target.x(),
            y: e.target.y(),
          });
        }}
        onTransformEnd={() => {
          const node = shapeRef.current;
          const scaleX = node.scaleX();
          const scaleY = node.scaleY();

          node.scaleX(1);
          node.scaleY(1);

          onChange({
            ...image,
            x: node.x(),
            y: node.y(),
            width: Math.max(5, node.width() * scaleX),
            height: Math.max(5, node.height() * scaleY),
            rotation: node.rotation(),
          });
        }}
      />
      {isSelected && (
        <Transformer
          ref={trRef}
          boundBoxFunc={(oldBox, newBox) => {
            if (newBox.width < 5 || newBox.height < 5) {
              return oldBox;
            }
            return newBox;
          }}
        />
      )}
    </>
  );
};

const BoundingBox = ({ box, color = 'red', label }) => {
  const { x, y, width, height } = box;

  return (
    <>
      <Rect
        x={x}
        y={y}
        width={width}
        height={height}
        stroke={color}
        strokeWidth={2}
        dash={[5, 2]}
        fill="transparent"
      />
      {label && (
        <Text
          x={x}
          y={y - 20}
          text={label}
          fontSize={16}
          fill={color}
          padding={2}
          background="#ffffff88"
        />
      )}
    </>
  );
};

const ImageCanvas = ({ width, height, initialImageUrl, objectCoordinates }) => {
  const [images, setImages] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [boundingBoxes, setBoundingBoxes] = useState([]);
  const stageRef = useRef();

  useEffect(() => {
    if (initialImageUrl) {
      console.log('Loading initial image:', initialImageUrl);
      const img = new window.Image();
      img.crossOrigin = 'anonymous';

      img.onerror = (error) => {
        console.error('Image loading failed:', error);
        console.log('Attempting to load image without crossOrigin');
        const fallbackImg = new window.Image();
        fallbackImg.src = initialImageUrl;

        fallbackImg.onload = () => handleImageLoaded(fallbackImg);
        fallbackImg.onerror = (fallbackError) => {
          console.error('Fallback image loading also failed:', fallbackError);
        };
      };

      const handleImageLoaded = (loadedImg) => {
        let imgWidth = loadedImg.width;
        let imgHeight = loadedImg.height;

        const maxWidth = width * 0.8;
        const maxHeight = height * 0.8;

        if (imgWidth > maxWidth) {
          const scale = maxWidth / imgWidth;
          imgWidth = maxWidth;
          imgHeight *= scale;
        }

        if (imgHeight > maxHeight) {
          const scale = maxHeight / imgHeight;
          imgHeight = maxHeight;
          imgWidth *= scale;
        }

        const newImage = {
          id: 'main-image',
          imageObj: loadedImg,
          x: (width - imgWidth) / 2,
          y: (height - imgHeight) / 2,
          width: imgWidth,
          height: imgHeight,
          rotation: 0,
        };

        setImages([newImage]);

        console.log('Image loaded, checking for coordinate data:', !!objectCoordinates);

        setTimeout(() => {
          if (objectCoordinates) {
            console.log('Force processing coordinates after image load');
            processCoordinatesData(objectCoordinates, newImage);
          }
        }, 100);
      };

      img.onload = () => handleImageLoaded(img);

      img.src = initialImageUrl;

      if (img.complete) {
        handleImageLoaded(img);
      }
    }
  }, [initialImageUrl, width, height, objectCoordinates]);

  const processCoordinatesData = (coordinates, mainImage = null) => {
    if (!coordinates) return;

    const imageToUse = mainImage || images.find(img => img.id === 'main-image');
    if (!imageToUse) {
      console.log('Cannot process coordinates: main image not found');
      return;
    }

    console.log('Starting to process coordinate data');

    const boxes = [];

    try {
      let parsedCoordinates;

      if (typeof coordinates === 'string') {
        try {
          parsedCoordinates = JSON.parse(coordinates);
        } catch (e) {
          console.error('Failed to parse JSON coordinate string:', e);

          try {
            const jsonRegex = /(\{.*\}|\[.*\])/s;
            const match = coordinates.match(jsonRegex);
            if (match) {
              parsedCoordinates = JSON.parse(match[0]);
            }
          } catch (extractError) {
            console.error('Failed to extract JSON part:', extractError);
          }
        }
      } else {
        parsedCoordinates = coordinates;
      }

      if (parsedCoordinates) {
        if (Array.isArray(parsedCoordinates)) {
          if (parsedCoordinates.length >= 4 && typeof parsedCoordinates[0] === 'number') {
            const boxData = {
              x: parsedCoordinates[0] * imageToUse.width + imageToUse.x,
              y: parsedCoordinates[1] * imageToUse.height + imageToUse.y,
              width: (parsedCoordinates[2] - parsedCoordinates[0]) * imageToUse.width,
              height: (parsedCoordinates[3] - parsedCoordinates[1]) * imageToUse.height,
              label: 'Object 1'
            };
            boxes.push(boxData);
          } else {
            parsedCoordinates.forEach((coord, index) => {
              if (!coord) return;

              if (coord.bbox && Array.isArray(coord.bbox)) {
                const boxData = {
                  x: coord.bbox[0] * imageToUse.width + imageToUse.x,
                  y: coord.bbox[1] * imageToUse.height + imageToUse.y,
                  width: (coord.bbox[2] - coord.bbox[0]) * imageToUse.width,
                  height: (coord.bbox[3] - coord.bbox[1]) * imageToUse.height,
                  label: coord.label || `Object ${index + 1}`
                };
                boxes.push(boxData);
              } else if (coord.label && typeof coord.label === 'string') {
                let x1, y1, x2, y2, width, height;

                if ('x' in coord) x1 = parseFloat(coord.x);
                if ('y' in coord) y1 = parseFloat(coord.y);
                if ('width' in coord) width = parseFloat(coord.width);
                if ('height' in coord) height = parseFloat(coord.height);
                if ('x2' in coord) x2 = parseFloat(coord.x2);
                if ('y2' in coord) y2 = parseFloat(coord.y2);

                if (x1 !== undefined && x2 !== undefined && width === undefined) {
                  width = x2 - x1;
                }
                if (y1 !== undefined && y2 !== undefined && height === undefined) {
                  height = y2 - y1;
                }

                if (x1 !== undefined && y1 !== undefined && width !== undefined && height !== undefined) {
                  const boxData = {
                    x: x1 * imageToUse.width + imageToUse.x,
                    y: y1 * imageToUse.height + imageToUse.y,
                    width: width * imageToUse.width,
                    height: height * imageToUse.height,
                    label: coord.label
                  };
                  boxes.push(boxData);
                }
              }
            });
          }
        } else if (typeof parsedCoordinates === 'object') {
          if (parsedCoordinates.label && parsedCoordinates.bbox) {
            const boxData = {
              x: parsedCoordinates.bbox[0] * imageToUse.width + imageToUse.x,
              y: parsedCoordinates.bbox[1] * imageToUse.height + imageToUse.y,
              width: (parsedCoordinates.bbox[2] - parsedCoordinates.bbox[0]) * imageToUse.width,
              height: (parsedCoordinates.bbox[3] - parsedCoordinates.bbox[1]) * imageToUse.height,
              label: parsedCoordinates.label
            };
            boxes.push(boxData);
          }
        }
      }

      console.log(`Processing complete, created ${boxes.length} bounding boxes`);
      if (boxes.length > 0) {
        setBoundingBoxes(boxes);
      }
    } catch (error) {
      console.error('Error occurred while processing coordinates:', error);
    }
  };

  useEffect(() => {
    console.log('ImageCanvas: objectCoordinates effect triggered', {
      hasCoordinates: !!objectCoordinates,
      imagesLength: images.length,
      objectCoordinatesType: typeof objectCoordinates,
      isArray: Array.isArray(objectCoordinates),
      objectCoordinatesValue: objectCoordinates
    });

    if (objectCoordinates && images.length > 0) {
      const mainImage = images.find(img => img.id === 'main-image');
      if (!mainImage) {
        console.log('Main image not found');
        return;
      }

      const boxes = [];

      try {
        console.log('Processing object coordinate data:',
          typeof objectCoordinates === 'string' ?
            objectCoordinates.substring(0, 100) + '...' :
            objectCoordinates
        );

        let parsedCoordinates;
        if (typeof objectCoordinates === 'string') {
          try {
            parsedCoordinates = JSON.parse(objectCoordinates);
            console.log('Successfully parsed JSON string');
          } catch (e) {
            console.error('Failed to parse JSON coordinate string:', e);

            try {
              const jsonRegex = /(\{.*\}|\[.*\])/s;
              const match = objectCoordinates.match(jsonRegex);
              if (match) {
                parsedCoordinates = JSON.parse(match[0]);
                console.log('Extracted JSON part');
              }
            } catch (extractError) {
              console.error('Failed to extract JSON part:', extractError);
            }
          }
        } else {
          parsedCoordinates = objectCoordinates;
          console.log('Using non-string coordinate data, type:', Array.isArray(parsedCoordinates) ? 'Array' : typeof parsedCoordinates);
        }

        if (parsedCoordinates) {
          console.log('Parsed coordinate data type:', Array.isArray(parsedCoordinates) ? 'Array' : typeof parsedCoordinates);

          if (Array.isArray(parsedCoordinates)) {
            if (parsedCoordinates.length === 0) {
              console.log('Coordinate array is empty');
            } else if (parsedCoordinates.length >= 4 &&
              typeof parsedCoordinates[0] === 'number') {
              console.log('Processing single coordinate array');
              processCoordinates([parsedCoordinates]);
            } else {
              console.log('Processing object array with', parsedCoordinates.length, 'items');
              processCoordinates(parsedCoordinates);
            }
          } else if (typeof parsedCoordinates === 'object') {
            console.log('Processing single object');
            processCoordinates([parsedCoordinates]);
          }
        } else {
          console.warn('Could not parse coordinate data');
        }
      } catch (error) {
        console.error('Error while processing coordinate data:', error);
      }

      function processCoordinates(coords) {
        if (!coords) return;

        if (!Array.isArray(coords)) {
          coords = [coords];
        }

        console.log(`Starting to process ${coords.length} coordinate items`);

        coords.forEach((coord, index) => {
          console.log(`Processing coordinate item ${index}:`, coord);
          let boxData;

          if (Array.isArray(coord) && coord.length >= 4) {
            boxData = {
              x: coord[0] * mainImage.width + mainImage.x,
              y: coord[1] * mainImage.height + mainImage.y,
              width: (coord[2] - coord[0]) * mainImage.width,
              height: (coord[3] - coord[1]) * mainImage.height,
              label: `Object ${index + 1}`
            };
            console.log(`Format [x1,y1,x2,y2], created bounding box:`, boxData);
          } else if (coord.bbox && Array.isArray(coord.bbox)) {
            boxData = {
              x: coord.bbox[0] * mainImage.width + mainImage.x,
              y: coord.bbox[1] * mainImage.height + mainImage.y,
              width: (coord.bbox[2] - coord.bbox[0]) * mainImage.width,
              height: (coord.bbox[3] - coord.bbox[1]) * mainImage.height,
              label: coord.label || `Object ${index + 1}`
            };
            console.log(`Format {bbox:[]}, created bounding box:`, boxData);
          } else if (coord.x !== undefined && coord.y !== undefined &&
            coord.width !== undefined && coord.height !== undefined) {
            boxData = {
              x: coord.x * mainImage.width + mainImage.x,
              y: coord.y * mainImage.height + mainImage.y,
              width: coord.width * mainImage.width,
              height: coord.height * mainImage.height,
              label: coord.label || `Object ${index + 1}`
            };
            console.log(`Format {x,y,width,height}, created bounding box:`, boxData);
          } else if (typeof coord === 'object') {
            console.log('Attempting to extract coordinates from object:', coord);

            let x1, y1, x2, y2, label;

            const possibleXKeys = ['x', 'x1', 'left', 'startX'];
            const possibleYKeys = ['y', 'y1', 'top', 'startY'];
            const possibleWidthKeys = ['width', 'w', 'dx'];
            const possibleHeightKeys = ['height', 'h', 'dy'];
            const possibleX2Keys = ['x2', 'right', 'endX'];
            const possibleY2Keys = ['y2', 'bottom', 'endY'];
            const possibleLabelKeys = ['label', 'name', 'class', 'type', 'category'];

            for (const key of possibleXKeys) {
              if (coord[key] !== undefined) {
                x1 = parseFloat(coord[key]);
                console.log(`Found x1 value: ${x1}, using field: ${key}`);
                break;
              }
            }

            for (const key of possibleYKeys) {
              if (coord[key] !== undefined) {
                y1 = parseFloat(coord[key]);
                console.log(`Found y1 value: ${y1}, using field: ${key}`);
                break;
              }
            }

            let width;
            for (const key of possibleWidthKeys) {
              if (coord[key] !== undefined) {
                width = parseFloat(coord[key]);
                console.log(`Found width: ${width}, using field: ${key}`);
                break;
              }
            }

            for (const key of possibleX2Keys) {
              if (coord[key] !== undefined) {
                x2 = parseFloat(coord[key]);
                console.log(`Found x2 value: ${x2}, using field: ${key}`);
                break;
              }
            }

            let height;
            for (const key of possibleHeightKeys) {
              if (coord[key] !== undefined) {
                height = parseFloat(coord[key]);
                console.log(`Found height: ${height}, using field: ${key}`);
                break;
              }
            }

            for (const key of possibleY2Keys) {
              if (coord[key] !== undefined) {
                y2 = parseFloat(coord[key]);
                console.log(`Found y2 value: ${y2}, using field: ${key}`);
                break;
              }
            }

            for (const key of possibleLabelKeys) {
              if (coord[key] !== undefined) {
                label = coord[key];
                console.log(`Found label: ${label}, using field: ${key}`);
                break;
              }
            }

            if (!label) {
              label = `Object ${index + 1}`;
            }

            if (x1 !== undefined && x2 !== undefined && width === undefined) {
              width = x2 - x1;
              console.log(`Calculated width from x1 and x2: ${width}`);
            }

            if (y1 !== undefined && y2 !== undefined && height === undefined) {
              height = y2 - y1;
              console.log(`Calculated height from y1 and y2: ${height}`);
            }

            if (x1 !== undefined && y1 !== undefined &&
              (width !== undefined || x2 !== undefined) &&
              (height !== undefined || y2 !== undefined)) {

              boxData = {
                x: x1 * mainImage.width + mainImage.x,
                y: y1 * mainImage.height + mainImage.y,
                width: width !== undefined ? width * mainImage.width : (x2 - x1) * mainImage.width,
                height: height !== undefined ? height * mainImage.height : (y2 - y1) * mainImage.height,
                label: label
              };
              console.log(`Extracted from object fields, created bounding box:`, boxData);
            } else {
              console.log('Extracted coordinate info is incomplete, cannot create bounding box');
            }
          }

          if (boxData) {
            console.log('Adding bounding box to list:', boxData);
            boxes.push(boxData);
          }
        });
      }

      console.log(`Created a total of ${boxes.length} bounding boxes`);
      setBoundingBoxes(boxes);
    } else {
      if (boundingBoxes.length > 0 && (!objectCoordinates || images.length === 0)) {
        console.log('Clearing bounding boxes');
        setBoundingBoxes([]);
      }
    }
  }, [objectCoordinates, images, boundingBoxes.length]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new window.Image();
      img.src = event.target.result;

      img.onload = () => {
        let imgWidth = img.width;
        let imgHeight = img.height;

        const maxWidth = width * 0.8;
        const maxHeight = height * 0.8;

        if (imgWidth > maxWidth) {
          const scale = maxWidth / imgWidth;
          imgWidth = maxWidth;
          imgHeight *= scale;
        }

        if (imgHeight > maxHeight) {
          const scale = maxHeight / imgHeight;
          imgHeight = maxHeight;
          imgWidth *= scale;
        }

        const newImage = {
          id: Date.now().toString(),
          imageObj: img,
          x: (width - imgWidth) / 2,
          y: (height - imgHeight) / 2,
          width: imgWidth,
          height: imgHeight,
          rotation: 0,
        };

        setImages([newImage]);
        setBoundingBoxes([]);

        if (window.updateCanvasDisplay) {
          console.log('Triggering canvas update after new image upload to get latest markers');
          setTimeout(() => window.updateCanvasDisplay(), 500);
        }
      };

      img.onerror = (error) => {
        console.error('Local image loading failed:', error);
        alert('Image failed to load, please try another one');
      };
    };

    reader.onerror = () => {
      console.error('Failed to read file');
      alert('Failed to read image file, please try another one');
    };

    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const saveCanvas = () => {
    try {
      if (images.length === 0) {
        console.error('No image available to export');
        return;
      }

      const pixelRatio = 2;

      const offscreenStage = new Konva.Stage({
        container: document.createElement('div'),
        width: stageRef.current.width(),
        height: stageRef.current.height(),
      });

      const offscreenLayer = new Konva.Layer();
      offscreenStage.add(offscreenLayer);

      images.forEach(image => {
        const img = new Konva.Image({
          image: image.imageObj,
          x: image.x,
          y: image.y,
          width: image.width,
          height: image.height,
          rotation: image.rotation,
        });
        offscreenLayer.add(img);
      });

      boundingBoxes.forEach((box, index) => {
        const rect = new Konva.Rect({
          x: box.x,
          y: box.y,
          width: box.width,
          height: box.height,
          stroke: ['red', 'blue', 'green', 'orange', 'purple'][index % 5],
          strokeWidth: 2,
          dash: [5, 2],
          fill: 'transparent',
        });
        offscreenLayer.add(rect);

        if (box.label) {
          const text = new Konva.Text({
            x: box.x,
            y: box.y - 20,
            text: box.label,
            fontSize: 16,
            fill: ['red', 'blue', 'green', 'orange', 'purple'][index % 5],
            padding: 2,
            background: '#ffffff88',
          });
          offscreenLayer.add(text);
        }
      });

      offscreenLayer.draw();

      const dataURL = offscreenStage.toDataURL({
        pixelRatio: pixelRatio,
        mimeType: 'image/png',
        quality: 1
      });

      const link = document.createElement('a');
      link.download = 'canvas-export.png';
      link.href = dataURL;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      offscreenStage.destroy();
      console.log('Image exported successfully');
    } catch (error) {
      console.error('Error while exporting canvas:', error);
      alert('Failed to export image, please check the console for more information');
    }
  };

  const checkDeselect = (e) => {
    if (e.target === e.target.getStage()) {
      setSelectedId(null);
    }
  };

  return (
    <div className="image-canvas-container">
      <div className="canvas-main-container">
        <Stage
          ref={stageRef}
          width={width}
          height={height}
          onMouseDown={checkDeselect}
          onTouchStart={checkDeselect}
          style={{
            border: 'none',
            background: '#fff'
          }}
        >
          <Layer>
            {images.map((image) => (
              <DraggableImage
                key={image.id}
                image={image}
                isSelected={image.id === selectedId}
                onSelect={() => setSelectedId(image.id)}
                onChange={(newAttrs) => {
                  const updatedImages = images.map((img) => {
                    if (img.id === image.id) {
                      return newAttrs;
                    }
                    return img;
                  });
                  setImages(updatedImages);
                }}
              />
            ))}

            {boundingBoxes.map((box, index) => (
              <BoundingBox
                key={index}
                box={box}
                color={['red', 'blue', 'green', 'orange', 'purple'][index % 5]}
                label={box.label}
              />
            ))}
          </Layer>
        </Stage>
      </div>

      <div className="download-button-container">
        <button className="download-button" onClick={saveCanvas}>
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
            <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z" />
            <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z" />
          </svg>
          Download
        </button>
      </div>
    </div>
  );
};

export default ImageCanvas;