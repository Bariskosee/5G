FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    python3 \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/data/input /app/data/output /app/models /app/src

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel

# Install CUDA 12.1 compatible PyTorch before Ultralytics to avoid CPU-only wheels.
RUN python3 -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1 \
    torchvision==0.18.1

RUN sed '/^torch==/d;/^torchvision==/d' /app/requirements.txt > /tmp/requirements-no-torch.txt \
    && python3 -m pip install --no-cache-dir -r /tmp/requirements-no-torch.txt

# Pre-download EasyOCR model files at build time.
# Runtime has no internet. Turkish plates need both 'tr' and 'en' language packs.
# Use /app/easyocr_models to avoid default-path double-slash bug in EasyOCR 1.7.x.
RUN mkdir -p /app/easyocr_models && \
    python3 -c "import easyocr; easyocr.Reader(['tr', 'en'], gpu=False, model_storage_directory='/app/easyocr_models', verbose=True)"

# Pre-download MediaPipe FaceLandmarker model at build time (no internet at runtime).
RUN mkdir -p /app/models/mediapipe && \
    python3 -c "\
import urllib.request; \
urllib.request.urlretrieve(\
'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', \
'/app/models/mediapipe/face_landmarker.task'); \
print('face_landmarker.task downloaded')"

# Pre-download YOLOv8m COCO weights for vehicle type detection.
# File is saved to CWD (/app) so it is found at runtime without internet.
RUN python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt'); print('yolov8m.pt ready')"

COPY main.py /app/main.py
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY models/ /app/models/

CMD ["python3", "main.py"]
