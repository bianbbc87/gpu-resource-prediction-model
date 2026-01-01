FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# PyTorch 먼저 (CUDA 11.8 고정)
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 나머지 라이브러리
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# train.sh 등록
COPY scripts/train.sh /workspace/train.sh
RUN chmod +x /workspace/train.sh

ENV PYTHONPATH=/workspace

# 핵심: 실행 엔트리는 train.sh
ENTRYPOINT ["/workspace/train.sh"]
