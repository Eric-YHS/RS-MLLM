from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
import time


class TaskType(str, Enum):
    """任务类型枚举"""
    DESCRIPTION = "description"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CUSTOM = "custom"


class ModelType(str, Enum):
    """模型类型枚举"""


class AnalyzeRequest(BaseModel):
    """图像分析请求模型"""
    prompt: str = Field(..., description="分析提示或问题")
    task_type: TaskType = Field(TaskType.DESCRIPTION, description="分析任务类型")
    model: ModelType = Field(ModelType.GLM_45V, description="使用的模型")


class AnalyzeResponse(BaseModel):
    """图像分析响应模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: Optional[str] = Field(None, description="任务消息")
    result: Optional[Any] = Field(None, description="分析结果")
    error: Optional[str] = Field(None, description="错误信息(如果有)")
    completed_at: Optional[float] = Field(None, description="完成时间戳")
    thinking: Optional[str] = Field(None, description="模型思考过程")


class TaskResult(BaseModel):
    """任务结果模型"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    result: Optional[Any] = Field(None, description="分析结果")
    error: Optional[str] = Field(None, description="错误信息(如果有)")
    completed_at: Optional[float] = Field(None, description="完成时间戳")
    thinking: Optional[str] = Field(None, description="模型思考过程")
