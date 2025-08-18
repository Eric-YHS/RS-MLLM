const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'src/components/ImageCanvas.js');
let content = fs.readFileSync(filePath, 'utf8');

const oldCode = `        try {
          parsedCoordinates = JSON.parse(objectCoordinates);
          console.log('Successfully parsed JSON string');
        } catch (e) {
          console.error('Failed to parse JSON coordinate string:', e);
          
          try {
            const jsonRegex = /(\\{.*\\}|\\[.*\\])/s;
            const match = objectCoordinates.match(jsonRegex);
            if (match) {
              parsedCoordinates = JSON.parse(match[1]);
              console.log('Extracted and parsed JSON from string');
            }
          } catch (extractError) {
            console.error('Failed to extract JSON part:', extractError);
          }`;

const newCode = `        try {
          parsedCoordinates = JSON.parse(objectCoordinates);
          console.log('Successfully parsed JSON string');
        } catch (e) {
          console.error('Failed to parse JSON coordinate string:', e);
          
          try {
            const jsonRegex = /(\\{.*?\\}|\\[.*?\\])/gs;
            const matches = objectCoordinates.match(jsonRegex);
            
            if (matches) {
              for (const match of matches) {
                try {
                  const parsed = JSON.parse(match);
                  if (Array.isArray(parsed) || 
                      (parsed && typeof parsed === 'object' && 
                       (parsed.bbox || (parsed.x !== undefined && parsed.y !== undefined)))) {
                    parsedCoordinates = parsed;
                    console.log('Found valid JSON coordinates in string:', match);
                    break;
                  }
                } catch (err) {
                  // Continue to next match
                }
              }
            }
          } catch (extractError) {
            console.error('Failed to extract JSON part:', extractError);
          }`;

const updatedContent = content.replace(oldCode, newCode);

fs.writeFileSync(filePath, updatedContent, 'utf8');
console.log('File has been updated');