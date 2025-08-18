from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional
import asyncio

from app.db.database import get_db
from app.services.user_service import MessageService, ChatService
from app.api.api_v1.endpoints.users import get_current_user
from app.db.models import User
from app.services.zhipuai_service import zhipuai_service

router = APIRouter()

@router.post("/text", response_model=Dict[str, Any])
async def process_text_message(
    data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    prompt = data.get("prompt")
    chat_id = data.get("chat_id")
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing required parameter: prompt")
    if not chat_id:
        raise HTTPException(status_code=400, detail="Missing required parameter: chat_id")
    
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found or does not belong to the current user")
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text=prompt,
        sender="user"
    )
    
    messages = MessageService.get_chat_messages(db, chat_id)
    
    image_messages = [msg for msg in messages if msg.image_path and msg.sender == "system"]
    last_image_message = image_messages[-1] if image_messages else None
    
    if not last_image_message:
        error_message = MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text="Please upload a remote sensing image first for analysis",
            sender="ai",
            error=True
        )
        return {
            "status": "error",
            "message": "Please upload a remote sensing image first for analysis"
        }
    
    processing_message = MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text="Analyzing, please wait...",
        sender="system"
    )
    
    context_messages = []
    for msg in messages:
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
    
    try:
        image_path = last_image_message.image_path
        if image_path.startswith('/api/uploads/'):
            image_path = image_path.replace('/api/uploads/', '')
        
        import os
        from app.core.config import UPLOAD_FOLDER
        import base64
        
        full_image_path = os.path.join(UPLOAD_FOLDER, image_path)
        
        with open(full_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        loop = asyncio.get_event_loop()
        result = await zhipuai_service.analyze_image(
            image_base64=image_base64,
            prompt=prompt,
            task_type="description",
            context_messages=context_messages
        )
        
        if hasattr(result, "content"):
            content = result.content
            thinking = getattr(result, "thinking", None)
        else:
            content = str(result)
            thinking = None
            
        db.delete(processing_message)
        db.commit()
        
        ai_message = MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text=content,
            sender="ai",
            thinking=thinking
        )
        
        return {
            "status": "success",
            "message": "Text processed successfully",
            "result": content,
            "thinking": thinking
        }
        
    except Exception as e:
        db.delete(processing_message)
        db.commit()
        
        error_message = MessageService.create_message(
            db=db,
            chat_id=chat_id,
            text=f"Processing error: {str(e)}",
            sender="ai",
            error=True
        )
        
        raise HTTPException(status_code=500, detail=f"Error processing text message: {str(e)}")