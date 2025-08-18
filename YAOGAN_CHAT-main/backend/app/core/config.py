import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

API_PREFIX = os.getenv("API_PREFIX", "/api")

CORS_ORIGINS = eval(os.getenv("CORS_ORIGINS", '["http://localhost:3000"]'))

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