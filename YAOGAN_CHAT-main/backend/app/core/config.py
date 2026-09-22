import ast
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_PREFIX = os.getenv("API_PREFIX", "/api")


def _parse_origins(raw: str):
    """解析 CORS_ORIGINS：优先按 JSON 数组，兼容 .env 里写 Python 列表字面量的旧写法。"""
    try:
        value = json.loads(raw)
    except ValueError:
        value = ast.literal_eval(raw)
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("CORS_ORIGINS 需要是一个字符串数组")
    return value


CORS_ORIGINS = _parse_origins(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]'))

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./yaogan_chat.db")

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60 * 24))

ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "uploads"))
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16777216))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)