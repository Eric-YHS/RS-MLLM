FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=1
ENV TOKENIZERS_PARALLELISM=false


RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


COPY . .

# 注意：模型权重不入库也不进镜像（.gitignore 已排除 pytorch_model.bin）。
# 运行时需要把放好权重的目录挂载进容器，例如：
#   docker build -t rs-mllm .
#   docker run --rm --gpus all -v /path/to/ninegrids-4b:/app/weights rs-mllm \
#     python mul_lora_systems/demo.py --model_path /app/weights --image_path 1.jpg --question "..."
# 网页版（FastAPI 后端 + React 前端）在 YAOGAN_CHAT-main/ 下，有各自的 Dockerfile，
# 后端监听 8000 端口。
EXPOSE 8000

CMD ["python", "mul_lora_systems/demo.py", "--help"]
