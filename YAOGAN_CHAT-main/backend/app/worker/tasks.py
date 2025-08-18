import base64
import time
import json
import os
import logging
import subprocess
import sys
from PIL import Image
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.worker.celery_app import celery_app
from app.core.config import UPLOAD_FOLDER, DATABASE_URL
from app.services.user_service import MessageService, ChatService
from app.db.models import Message, Chat
from app.utils.image_utils import preprocess_image

logger = logging.getLogger(__name__)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def call_multi_lora_model(image_path: str, question: str, task_type: str = "description"):
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        app_dir = os.path.dirname(current_dir)
        backend_dir = os.path.dirname(app_dir)
        yaogan_dir = os.path.dirname(backend_dir)
        project_root = os.path.dirname(yaogan_dir)
        
        model_script = os.path.join(project_root, "mul_lora_systems", "demo.py")
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Model script path: {model_script}")
        
        if not os.path.exists(model_script):
            return {
                "error": f"Model script does not exist: {model_script}",
                "success": False
            }
        
        cmd = [
            sys.executable,
            model_script,
            "--model_path", ".",
            "--image_path", image_path,
            "--question", question,
            "--mode", "single"
        ]
        
        logger.info(f"Calling multi-LoRA model: {' '.join(cmd)}")
        
        original_cwd = os.getcwd()
        os.chdir(project_root)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8'
            )
            
            if result.returncode == 0:
                output_lines = result.stdout.split('\n')
                response = None
                
                for i, line in enumerate(output_lines):
                    if "💬 模型回答:" in line or "模型回答:" in line:
                        if i + 1 < len(output_lines):
                            response_line = output_lines[i + 1].strip()
                            response = response_line.lstrip()
                        break
                
                if not response:
                    for line in reversed(output_lines):
                        line = line.strip()
                        if line and not line.startswith(('✅', '⏱️', '🎉', '📊', '=')):
                            response = line
                            break
                
                if not response:
                    response = "Model executed successfully but no valid response was found"
                
                return {
                    "content": response,
                    "task_type": task_type,
                    "success": True
                }
            else:
                error_msg = f"Model execution failed: {result.stderr}"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "success": False
                }
                
        finally:
            os.chdir(original_cwd)
            
    except subprocess.TimeoutExpired:
        error_msg = "Model execution timed out"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "success": False
        }
    except Exception as e:
        error_msg = f"Error calling multi-LoRA model: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "success": False
        }

def get_db():
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()

@celery_app.task(name="process_image_task")
def process_image_task(task_id, image_path, prompt, task_type, chat_id=None):
    logger.info(f"Starting to process task {task_id}, task type: {task_type}, chat ID: {chat_id}")
    
    from redis import Redis
    from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
    
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    redis_client.setex(f"task_processing:{task_id}", 3600, "1")
    
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        context_messages = []
        if chat_id:
            db = get_db()
            messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp.desc()).limit(10).all()
            
            messages = sorted(messages, key=lambda x: x.timestamp)
            
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
            
            db.close()
        
        if redis_client.exists(f"task_cancel:{task_id}"):
            logger.info(f"Task {task_id} was canceled by user, terminating.")
            redis_client.delete(f"task_cancel:{task_id}")
            return {
                "task_id": task_id,
                "chat_id": chat_id,
                "status": "canceled",
                "result": "Task was canceled by user",
                "completed_at": time.time()
            }
        
        result = call_multi_lora_model(
            image_path=image_path, 
            question=prompt, 
            task_type=task_type
        )
        
        if isinstance(result, dict):
            if "error" in result:
                content = f"Model execution error: {result['error']}"
                if chat_id:
                    db = get_db()
                    MessageService.create_message(
                        db=db,
                        chat_id=chat_id,
                        text=content,
                        sender="system"
                    )
                    db.close()
                
                return {
                    "task_id": task_id,
                    "chat_id": chat_id,
                    "status": "error",
                    "result": content,
                    "completed_at": time.time()
                }
            elif "content" in result:
                content = result["content"]
            else:
                content = str(result)
        elif hasattr(result, "content"):
            content = result.content
        else:
            content = str(result)
        
        object_coordinates = None
        is_object_mark = task_type == "detection"
        
        if is_object_mark:
            try:
                import re
                
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
                
                if not object_coordinates and is_object_mark:
                    logger.warning(f"Could not extract coordinate information from content: {content}")
                    object_coordinates = content
            except Exception as e:
                logger.error(f"Error extracting coordinate information: {e}")
        
        formatted_result = {
            "task_id": task_id,
            "chat_id": chat_id,
            "status": "completed",
            "result": content,
            "completed_at": time.time(),
            "is_object_mark": is_object_mark,
            "object_coordinates": object_coordinates
        }
        
        if hasattr(result, "thinking"):
            formatted_result["thinking"] = result.thinking
        
        if chat_id:
            db = get_db()
            try:
                processing_msgs = db.query(Message).filter(
                    Message.chat_id == chat_id,
                    Message.sender == "system",
                    Message.text.like("Analyzing%")
                ).all()
                for msg in processing_msgs:
                    db.delete(msg)
                
                MessageService.create_message(
                    db=db,
                    chat_id=chat_id,
                    text=content,
                    sender="ai",
                    thinking=formatted_result.get("thinking"),
                    is_object_mark=formatted_result.get("is_object_mark", False),
                    object_coordinates=formatted_result.get("object_coordinates")
                )
                
                db.commit()
            except Exception as e:
                logger.error(f"Error saving message to database: {str(e)}")
                db.rollback()
            finally:
                db.close()
        
        redis_client.setex(
            f"task_result:{task_id}", 
            86400,
            json.dumps(formatted_result)
        )
        
        redis_client.delete(f"task_processing:{task_id}")
        redis_client.delete(f"task_cancel:{task_id}")
        
        logger.info(f"Task {task_id} completed and saved to Redis and database")
        return formatted_result
        
    except Exception as e:
        error_result = {
            "task_id": task_id,
            "chat_id": chat_id,
            "status": "failed",
            "error": str(e),
            "completed_at": time.time(),
        }
        
        if chat_id:
            db = get_db()
            try:
                processing_msgs = db.query(Message).filter(
                    Message.chat_id == chat_id,
                    Message.sender == "system",
                    Message.text.like("Analyzing%")
                ).all()
                for msg in processing_msgs:
                    db.delete(msg)
                
                MessageService.create_message(
                    db=db,
                    chat_id=chat_id,
                    text=f"Analysis error: {str(e)}",
                    sender="ai",
                    error=True
                )
                
                db.commit()
            except Exception as db_error:
                logger.error(f"Error saving error message to database: {str(db_error)}")
                db.rollback()
            finally:
                db.close()
        
        redis_client.setex(
            f"task_result:{task_id}", 
            86400,
            json.dumps(error_result)
        )
        
        redis_client.delete(f"task_processing:{task_id}")
        redis_client.delete(f"task_cancel:{task_id}")
        
        logger.error(f"Error processing task {task_id}: {str(e)}")
        return error_result

@celery_app.task(name="process_text_task")
def process_text_task(task_id, prompt, chat_id, task_type="description"):
    logger.info(f"Starting to process text task {task_id}, task type: {task_type}, chat ID: {chat_id}")
    
    from redis import Redis
    from app.core.config import REDIS_HOST, REDIS_PORT, REDIS_DB
    
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    redis_client.setex(f"task_processing:{task_id}", 3600, "1")
    
    try:
        db = get_db()
        
        chat = db.query(Chat).filter(Chat.id == chat_id).first()
        if not chat:
            raise Exception("Chat session does not exist")
        
        messages = db.query(Message).filter(Message.chat_id == chat_id).order_by(Message.timestamp.desc()).limit(20).all()
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
            raise Exception("No uploaded image in the chat")
        
        image_filename = os.path.basename(image_path.replace("/api/uploads/", ""))
        local_image_path = os.path.join(UPLOAD_FOLDER, image_filename)
        
        if not os.path.isfile(local_image_path):
            raise Exception("Historical image file is no longer accessible, please re-upload the image")
        
        if redis_client.exists(f"task_cancel:{task_id}"):
            logger.info(f"Text task {task_id} was canceled by user, terminating.")
            redis_client.delete(f"task_cancel:{task_id}")
            return {
                "task_id": task_id,
                "chat_id": chat_id,
                "status": "canceled",
                "result": "Task was canceled by user",
                "completed_at": time.time()
            }
        
        with open(local_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode('utf-8')
        
        api_task_type = task_type
        if task_type == "mark_object":
            api_task_type = "detection"
        
        result = call_multi_lora_model(
            image_path=local_image_path,
            question=prompt,
            task_type=api_task_type
        )
        
        if not result.get("success", False):
            raise Exception(result.get("error", "Model invocation failed"))
        
        content = result.get("content", "")
        thinking = None
        
        object_coordinates = None
        if task_type == "mark_object":
            try:
                import re
                
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
            except Exception as e:
                logger.warning(f"Error extracting coordinate information: {e}")
        
        try:
            MessageService.create_message(
                db=db,
                chat_id=chat_id,
                text=content,
                sender="ai",
                thinking=thinking,
                object_coordinates=object_coordinates,
                is_object_mark=(task_type == "mark_object")
            )
            
            db.commit()
        except Exception as db_error:
            logger.error(f"Error saving message to database: {str(db_error)}")
            db.rollback()
        finally:
            db.close()
        
        success_result = {
            "task_id": task_id,
            "chat_id": chat_id,
            "status": "completed",
            "result": content,
            "thinking": thinking,
            "object_coordinates": object_coordinates,
            "is_object_mark": (task_type == "mark_object"),
            "completed_at": time.time()
        }
        
        redis_client.setex(
            f"task_result:{task_id}", 
            86400,
            json.dumps(success_result)
        )
        
        redis_client.delete(f"task_processing:{task_id}")
        redis_client.delete(f"task_cancel:{task_id}")
        
        logger.info(f"Text task {task_id} processing complete")
        return success_result
        
    except Exception as e:
        error_result = {
            "task_id": task_id,
            "chat_id": chat_id,
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        }
        
        if chat_id:
            try:
                db = get_db()
                
                if "Image input format/parsing error" in str(e):
                    user_friendly_message = "The historical image may have expired or be in an incompatible format, please re-upload the image"
                elif "No uploaded image in the chat" in str(e):
                    user_friendly_message = "Please upload a remote sensing image first for me to analyze"
                elif "Historical image file is no longer accessible" in str(e):
                    user_friendly_message = "The historical image file is no longer accessible, please re-upload the image"
                else:
                    user_friendly_message = f"Processing error: {str(e)}"
                
                MessageService.create_message(
                    db=db,
                    chat_id=chat_id,
                    text=user_friendly_message,
                    sender="ai",
                    error=True
                )
                
                db.commit()
            except Exception as db_error:
                logger.error(f"Error saving error message to database: {str(db_error)}")
                db.rollback()
            finally:
                db.close()
        
        redis_client.setex(
            f"task_result:{task_id}", 
            86400,
            json.dumps(error_result)
        )
        
        redis_client.delete(f"task_processing:{task_id}")
        redis_client.delete(f"task_cancel:{task_id}")
        
        logger.error(f"Error processing text task {task_id}: {str(e)}")
        return error_result