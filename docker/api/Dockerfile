FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10001 anpr && useradd --uid 10001 --gid anpr --create-home anpr
COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY api.py database.py current_plates.py live_preview.py anpr_diagnostics.py plate_validator.py plate_formats.py plate_evidence.py vehicle_registry.py ./
COPY video_demo.py process_video.py evaluate_videos.py ./
COPY static/live.html static/live_dual.html ./static/
RUN mkdir -p /app/captures /app/data && chown -R anpr:anpr /app
USER anpr
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
