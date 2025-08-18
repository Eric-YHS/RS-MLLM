from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import asyncio
import uuid
import time
import json

from app.db.database import get_db
from app.services.user_service import MessageService, ChatService
from app.api.api_v1.endpoints.users import get_current_user
from app.db.models import User, Message
from app.services.zhipuai_service import zhipuai_service
from app.worker.tasks import process_text_task

router = APIRouter()

@router.post("/text")
async def process_text_message(
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if "prompt" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: prompt"
        )
    if "chat_id" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: chat_id"
        )
    
    prompt = data["prompt"]
    chat_id = data["chat_id"]
    task_type = data.get("task_type", "description")
    
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or does not belong to the current user"
        )
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text=prompt,
        sender="user"
    )
    
    messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp.desc()).limit(10).all()
    messages = sorted(messages, key=lambda x: x.timestamp)
    
    context_messages = []
    has_image = False
    image_path = None
    
    for msg in messages:
        if msg.sender == "system" and msg.image_path:
            has_image = True
            image_path = msg.image_path
        
        if msg.sender == "user":
            context_messages.append({
                "role": "user", 
                "content": msg.text
            })
        elif msg.sender == "ai":
            context_messages.append({
                "role": "assistant", 
                "content": msg.text
            })
    
    if not has_image:
        MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text="Please upload a remote sensing image first for me to analyze",
            sender="ai"
        )
        
        return {
            "status": "error",
            "message": "No uploaded image in the chat"
        }
        
    import os
    from app.core.config import UPLOAD_FOLDER
    
    image_filename = os.path.basename(image_path.replace("/api/uploads/", ""))
    local_image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    
    if not os.path.isfile(local_image_path):
        error_msg = "The historical image file is no longer accessible, please re-upload the image"
        MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text=error_msg,
            sender="ai",
            error=True
        )
        
        return {
            "status": "error",
            "message": error_msg
        }
    
    processing_message = MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text="Processing your request, please wait...",
        sender="system"
    )
    
    try:
        import base64
        
        api_task_type = task_type
        if task_type == "mark_object":
            api_task_type = "detection"
        
        try:
            with open(local_image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
            
            result = await zhipuai_service.analyze_image(
                image_base64=image_base64,
                prompt=prompt,
                task_type=api_task_type,
                context_messages=context_messages
            )
        except Exception as img_error:
            image_url = f"http://localhost:8000{image_path.replace('/api', '')}"
            result = await zhipuai_service.analyze_image(
                image_url=image_url,
                prompt=prompt,
                task_type=api_task_type,
                context_messages=context_messages
            )
        
        if hasattr(result, "content"):
            content = result.content
        else:
            content = str(result)
            
        db.delete(processing_message)
        
        if content and hasattr(zhipuai_service, "_clean_special_tags"):
            content = zhipuai_service._clean_special_tags(content)
        
        object_coordinates = None
        if task_type == "mark_object":
            try:
                import re
                import json
                
                try:
                    full_json = json.loads(content)
                    if isinstance(full_json, dict) and ('bbox' in full_json or 
                            ('x' in full_json and 'y' in full_json and 'width' in full_json and 'height' in full_json)):
                        object_coordinates = content
                    elif isinstance(full_json, list) and len(full_json) > 0:
                        if isinstance(full_json[0], dict) and ('bbox' in full_json[0] or 
                                ('x' in full_json[0] and 'y' in full_json[0] and 'width' in full_json[0] and 'height' in full_json[0])):
                            object_coordinates = content
                        elif len(full_json) >= 4 and all(isinstance(item, (int, float)) for item in full_json[:4]):
                            object_coordinates = content
                except:
                    bbox_json_pattern = r'\{"label":[^}]+,"bbox":\[[^\]]+\]\}'
                    bbox_matches = re.findall(bbox_json_pattern, content)
                    
                    if bbox_matches:
                        object_coordinates = bbox_matches[0]
                    else:
                        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
                        json_matches = re.findall(json_pattern, content)
                        
                        array_pattern = r'\[(?:[^\[\]]|\[[^\[\]]*\])*\]'
                        array_matches = re.findall(array_pattern, content)
                        
                        if json_matches:
                            for match in json_matches:
                                try:
                                    obj = json.loads(match)
                                    if isinstance(obj, dict) and ('bbox' in obj or 
                                        ('x' in obj and 'y' in obj and 'width' in obj and 'height' in obj) or
                                        'label' in obj):
                                        object_coordinates = match
                                        break
                                except:
                                    pass
                                    
                        if not object_coordinates and array_matches:
                            for match in array_matches:
                                try:
                                    arr = json.loads(match)
                                    if isinstance(arr, list):
                                        if len(arr) >= 4 and all(isinstance(item, (int, float)) for item in arr[:4]):
                                            object_coordinates = match
                                            break
                                        elif len(arr) > 0 and isinstance(arr[0], dict):
                                            if ('bbox' in arr[0] or 
                                                ('x' in arr[0] and 'y' in arr[0] and 'width' in arr[0] and 'height' in arr[0])):
                                                object_coordinates = match
                                                break
                                except:
                                    pass
                
                if not object_coordinates:
                    print(f"Could not extract coordinate information from content: {content}")
            except Exception as e:
                print(f"Error extracting coordinate information: {e}")
        
        MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text=content,
            sender="ai",
            thinking=result.thinking if hasattr(result, "thinking") else None,
            object_coordinates=object_coordinates,
            is_object_mark=(task_type == "mark_object")
        )
        
        db.commit()
        
        return {
            "status": "success",
            "result": content,
            "thinking": result.thinking if hasattr(result, "thinking") else None,
            "object_coordinates": object_coordinates,
            "is_object_mark": (task_type == "mark_object")
        }
        
    except Exception as e:
        db.delete(processing_message)
        
        error_message = str(e)
        if "Image input format/parsing error" in error_message:
            user_friendly_message = "The historical image may have expired or be in an incompatible format, please re-upload the image"
        else:
            user_friendly_message = f"Error processing message: {error_message}"
        
        MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text=user_friendly_message,
            sender="ai",
            error=True
        )
        
        db.commit()
        
        return {
            "status": "error",
            "message": user_friendly_message,
            "original_error": error_message
        }

@router.post("/text-async")
async def process_text_message_async(
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if "prompt" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: prompt"
        )
    if "chat_id" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: chat_id"
        )
    
    prompt = data["prompt"]
    chat_id = data["chat_id"]
    task_type = data.get("task_type", "description")
    
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=404,
            detail="Chat session not found or does not belong to the current user"
        )
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text=prompt,
        sender="user"
    )
    
    task_id = str(uuid.uuid4())
    
    from redis import Redis
    from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
    
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    task_info = {
        "task_id": task_id,
        "chat_id": chat_id,
        "prompt": prompt,
        "task_type": task_type,
        "user_id": current_user.id,
        "status": "submitted",
        "submitted_at": time.time()
    }
    
    redis_client.setex(
        f"task_result:{task_id}",
        86400,
        json.dumps(task_info)
    )
    
    try:
        process_text_task.delay(task_id, prompt, chat_id, task_type)
        
        return {
            "task_id": task_id,
            "status": "submitted", 
            "message": "Text processing task has been submitted"
        }
        
    except Exception as e:
        redis_client.delete(f"task_result:{task_id}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit task: {str(e)}"
        )