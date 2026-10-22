FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision

RUN grep -v -E '^(torch|torchvision)$' requirements.txt \
    > /tmp/requirements-camera.txt \
    && sed -i "s/^opencv-python==/opencv-python-headless==/" /tmp/requirements-camera.txt \
    && pip install -r /tmp/requirements-camera.txt \
    && pip uninstall -y opencv-python \
    && pip install --force-reinstall --no-deps opencv-python-headless==5.0.0.93

COPY . .

CMD ["python", "camera.py"]
