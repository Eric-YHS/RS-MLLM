from fastapi import APIRouter, File, UploadFile, Form, HTTPException, BackgroundTasks, Query, Body, Depends
from fastapi.responses import JSONResponse
import uuid
import os
import time
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.core.config import UPLOAD_FOLDER
from app.worker.tasks import process_image_task
from app.models.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.zhipuai_service import zhipuai_service
from app.db.database import get_db
from app.services.user_service import MessageService, ChatService
from app.api.api_v1.endpoints.users import get_current_user
from app.db.models import User

router = APIRouter()

@router.post("/image", response_model=AnalyzeResponse)
@router.post("/image/", response_model=AnalyzeResponse)
async def analyze_image(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    task_type: str = Form("description"),
    model: str = Form("glm-4.5v"),
    chat_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if chat_id:
        chat = ChatService.get_chat_by_id(db, chat_id)
        if not chat or chat.user_id != current_user.id:
            raise HTTPException(
                status_code=404,
                detail="Chat session not found or does not belong to the current user"
            )
    else:
        chat = ChatService.create_chat(db, current_user.id, f"Analysis of {prompt[:20]}")
        chat_id = chat.id
    
    task_id = str(uuid.uuid4())
    
    content_type = file.content_type
    if not content_type or not content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Only image files are supported"
        )
    
    file_extension = os.path.splitext(file.filename)[1] if file.filename else ".jpg"
    image_filename = f"{task_id}{file_extension}"
    image_path = os.path.join(UPLOAD_FOLDER, image_filename)
    
    try:
        contents = await file.read()
        with open(image_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text=prompt,
        sender="user"
    )
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text="Image uploaded",
        sender="system",
        image_path=f"/api/uploads/{image_filename}"
    )
    
    MessageService.create_message(
        db=db,
        chat_id=chat_id,
        text="Analyzing image, please wait...",
        sender="system"
    )
    
    process_image_task.delay(task_id, image_path, prompt, task_type, chat_id)
    
    return {
        "task_id": task_id,
        "chat_id": chat_id,
        "status": "processing",
        "message": "Image successfully uploaded, analysis is in progress"
    }

@router.post("/image/url", response_model=AnalyzeResponse)
async def analyze_image_url(
    data: Dict[str, Any] = Body(...),
):
    if "image_url" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: image_url"
        )
    if "prompt" not in data:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: prompt"
        )
    
    image_url = data["image_url"]
    prompt = data["prompt"]
    task_type = data.get("task_type", "description")
    model = data.get("model", "glm-4.5v")
    
    task_id = str(uuid.uuid4())
    
    try:
        result = await zhipuai_service.analyze_image(
            image_url=image_url, 
            prompt=prompt, 
            task_type=task_type, 
            model=model
        )
        
        if hasattr(result, "content"):
            content = result.content
        elif isinstance(result, dict) and "error" in result:
            raise HTTPException(
                status_code=500,
                detail=f"Error analyzing image: {result['error']}"
            )
        else:
            content = str(result)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": content,
            "completed_at": time.time()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image URL: {str(e)}"
        )