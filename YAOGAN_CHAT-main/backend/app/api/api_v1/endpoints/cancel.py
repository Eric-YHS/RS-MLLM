from fastapi import APIRouter, HTTPException, Depends
from redis import Redis
import json
from typing import Dict, Any
import logging

from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
from app.api.api_v1.endpoints.users import get_current_user
from app.db.models import User

router = APIRouter()
logger = logging.getLogger(__name__)

def get_redis_client():
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    try:
        yield redis
    finally:
        redis.close()

@router.post("/{task_id}")
@router.post("/{task_id}/")
async def cancel_task(
    task_id: str, 
    redis: Redis = Depends(get_redis_client),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    if not redis.exists(f"task_processing:{task_id}"):
        return {
            "task_id": task_id,
            "status": "not_processing",
            "message": "Task is not being processed or is already complete"
        }
    
    try:
        redis.setex(f"task_cancel:{task_id}", 3600, "1")
        
        logger.info(f"User {current_user.username} has requested to cancel task {task_id}")
        
        import time
        from app.db.database import get_db
        from app.services.user_service import MessageService
        from sqlalchemy.orm import Session
        from fastapi import Depends
        
        canceled_result = {
            "task_id": task_id,
            "status": "canceled",
            "result": "Task canceled by user",
            "canceled_at": time.time()
        }
        
        task_data_raw = redis.get(f"task_result:{task_id}")
        if task_data_raw:
            try:
                task_data = json.loads(task_data_raw)
                chat_id = task_data.get("chat_id")
                
                if chat_id:
                    db = next(get_db())
                    try:
                        processing_msgs = db.query(MessageService.Message).filter(
                            MessageService.Message.chat_id == chat_id,
                            MessageService.Message.sender == "system",
                            MessageService.Message.text.like("Analyzing%")
                        ).all()
                        for msg in processing_msgs:
                            db.delete(msg)
                        
                        MessageService.create_message(
                            db=db,
                            chat_id=chat_id,
                            text="Generation canceled by user",
                            sender="system"
                        )
                        
                        db.commit()
                    except Exception as e:
                        logger.error(f"Error saving cancellation message to database: {str(e)}")
                        db.rollback()
                    finally:
                        db.close()
            except Exception as e:
                logger.error(f"Error processing task data: {str(e)}")
        
        redis.setex(
            f"task_result:{task_id}", 
            86400,
            json.dumps(canceled_result)
        )
        
        return {
            "task_id": task_id,
            "status": "canceling",
            "message": "Canceling task"
        }
    except Exception as e:
        logger.error(f"Error while canceling task: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error while canceling task: {str(e)}")