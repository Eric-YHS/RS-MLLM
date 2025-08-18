from fastapi import APIRouter, HTTPException, Depends
from redis import Redis
import json
from typing import Dict, Any, Optional

from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB

router = APIRouter()

def get_redis_client():
    redis = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    try:
        yield redis
    finally:
        redis.close()

@router.get("/{task_id}")
@router.get("/{task_id}/")
async def get_task_status(task_id: str, redis: Redis = Depends(get_redis_client)) -> Dict[str, Any]:
    task_key = f"task_result:{task_id}"
    task_result = redis.get(task_key)
    
    if not task_result:
        if redis.exists(f"task_processing:{task_id}"):
            return {
                "task_id": task_id,
                "status": "processing",
                "message": "Analysis is in progress"
            }
        else:
            raise HTTPException(status_code=404, detail=f"Task ID not found: {task_id}")
    
    try:
        result = json.loads(task_result)
        return result
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing task result")