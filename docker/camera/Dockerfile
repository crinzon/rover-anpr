FROM python:3.11-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOME=/home/anpr
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd --gid 10001 anpr && useradd --uid 10001 --gid anpr --create-home anpr
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch==2.14.0 torchvision==0.29.0
COPY docker/camera/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && pip check
COPY camera.py video_stream.py event_sender.py plate_validator.py ./
COPY full_stand/ ./full_stand/
COPY ocr_model/ ./ocr_model/
RUN mkdir -p /app/captures && chown -R anpr:anpr /app
USER anpr
CMD ["python", "camera.py"]
