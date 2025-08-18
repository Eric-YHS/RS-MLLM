#!/bin/bash

echo "==================================="
echo "Starting Remote Sensing Image Analysis Backend Service"
echo "==================================="

pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies, please check the error messages"
    exit 1
fi

echo "Starting Redis service..."
redis-server --daemonize yes
sleep 2

if [ ! -f yaogan_chat.db ]; then
    echo "Initializing database..."
    python init_db.py
fi

echo "Starting FastAPI server..."
uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

sleep 3

echo "Starting Celery Worker..."
celery -A app.worker.celery_app worker --loglevel=info &
CELERY_PID=$!

echo "==================================="
echo "Services have been started:"
echo "- FastAPI: http://localhost:8000"
echo "- API Docs: http://localhost:8000/docs"
echo "==================================="

trap "kill $FASTAPI_PID $CELERY_PID; exit" SIGINT
wait