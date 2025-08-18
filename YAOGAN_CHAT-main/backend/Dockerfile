FROM python:3.10-slim

WORKDIR /app



RUN apt update && \
    apt install -y --no-install-recommends gdal-bin libgdal-dev build-essential

COPY . .

RUN pip install --no-cache-dir -r requirements.txt && \
    mkdir -p uploads

EXPOSE 8000

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
