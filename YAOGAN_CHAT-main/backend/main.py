from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.core.config import API_PREFIX, UPLOAD_FOLDER
from app.api.api_v1.api import api_router
from starlette.middleware.base import BaseHTTPMiddleware

app = FastAPI(
    title="Remote Sensing Image Analysis API",
    description="A service for remote sensing image analysis",
    version="0.1.0",
    redirect_slashes=False
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(api_router, prefix=API_PREFIX)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class CORSStaticFilesMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        
        if request.url.path.startswith(f"{API_PREFIX}/uploads"):
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
            
        return response

app.mount(f"{API_PREFIX}/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")

app.add_middleware(CORSStaticFilesMiddleware)

@app.get("/")
async def root():
    return {"message": "Welcome to the Remote Sensing Image Analysis API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)