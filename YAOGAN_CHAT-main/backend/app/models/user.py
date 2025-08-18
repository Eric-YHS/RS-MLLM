from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

class Message(BaseModel):
    id: str
    text: str
    sender: str  # 'user', 'ai', 'system'
    timestamp: datetime
    image: Optional[str] = None  # 可选图像路径或URL
    thinking: Optional[str] = None  # AI的思考过程
    error: Optional[bool] = False  # 是否为错误消息

class Chat(BaseModel):
    id: str
    user_id: str
    title: str
    messages: List[Message] = []
    created_at: datetime
    last_updated: datetime
    
class UserBase(BaseModel):
    username: str
    display_name: Optional[str] = None
    avatar: Optional[str] = None

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: str
    created_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class UserInDB(User):
    hashed_password: str
    
    class Config:
        orm_mode = True
