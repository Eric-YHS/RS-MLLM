from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import Any, List

from app.db.database import get_db
from app.services.user_service import UserService, ChatService, MessageService
from app.core.security import create_access_token
from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from app.models.user import UserCreate, User
from jose import JWTError, jwt
from app.core.config import SECRET_KEY, ALGORITHM

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

router = APIRouter()

async def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = UserService.get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user

@router.post("/register", response_model=User)
async def register_user(user_create: UserCreate, db: Session = Depends(get_db)):
    db_user = UserService.get_user_by_username(db, user_create.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already in use"
        )
    
    user = UserService.create_user(
        db=db,
        username=user_create.username,
        password=user_create.password,
        display_name=user_create.display_name
    )
    
    ChatService.create_chat(db=db, user_id=user.id, title="New Chat")
    
    return user

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = UserService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    UserService.update_last_login(db, user.id)
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "display_name": user.display_name
    }

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@router.get("/chats", response_model=List[dict])
async def get_user_chats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chats = ChatService.get_user_chats(db, current_user.id)
    
    result = []
    for chat in chats:
        messages = MessageService.get_last_messages(db, chat.id, 1)
        last_message = None
        if messages:
            msg = messages[0]
            last_message = {
                "id": msg.id,
                "text": msg.text,
                "sender": msg.sender,
                "timestamp": msg.timestamp.isoformat()
            }
        
        chat_data = {
            "id": chat.id,
            "title": chat.title,
            "created_at": chat.created_at.isoformat(),
            "last_updated": chat.last_updated.isoformat(),
            "last_message": last_message
        }
        result.append(chat_data)
    
    return result

@router.post("/chats", response_model=dict)
async def create_new_chat(
    title: str = "New Chat", 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    chat = ChatService.create_chat(db, current_user.id, title)
    
    welcome_message = MessageService.create_message(
        db=db,
        chat_id=chat.id,
        text="Hello, welcome to the YAOGAN Chat System! Please start a conversation or upload a remote sensing image for analysis.",
        sender="ai"
    )
    
    return {
        "id": chat.id,
        "title": chat.title,
        "created_at": chat.created_at.isoformat(),
        "last_updated": chat.last_updated.isoformat()
    }

@router.get("/chats/{chat_id}/messages", response_model=List[dict])
async def get_chat_messages(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    messages = MessageService.get_chat_messages(db, chat_id)
    
    result = []
    for msg in messages:
        message_data = {
            "id": msg.id,
            "text": msg.text,
            "sender": msg.sender,
            "timestamp": msg.timestamp.isoformat(),
            "image_path": msg.image_path,
            "thinking": msg.thinking,
            "error": msg.error,
            "object_coordinates": msg.object_coordinates,
            "is_object_mark": msg.is_object_mark
        }
        result.append(message_data)
    
    return result

@router.put("/chats/{chat_id}", response_model=dict)
async def update_chat(
    chat_id: str,
    title: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    updated_chat = ChatService.update_chat_title(db, chat_id, title)
    
    return {
        "id": updated_chat.id,
        "title": updated_chat.title,
        "created_at": updated_chat.created_at.isoformat(),
        "last_updated": updated_chat.last_updated.isoformat()
    }

@router.put("/chats/{chat_id}/title", response_model=dict)
async def update_chat_title(
    chat_id: str,
    title_data: dict,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    title = title_data.get("title", "")
    if not title:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Title cannot be empty"
        )
    
    updated_chat = ChatService.update_chat_title(db, chat_id, title)
    
    return {
        "id": updated_chat.id,
        "title": updated_chat.title,
        "created_at": updated_chat.created_at.isoformat(),
        "last_updated": updated_chat.last_updated.isoformat()
    }

@router.delete("/chats/{chat_id}", response_model=dict)
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    chat = ChatService.get_chat_by_id(db, chat_id)
    if not chat or chat.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )
    
    success = ChatService.delete_chat(db, chat_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat session"
        )
    
    return {"message": "Chat session deleted"}