FROM python:3.11-slim-bookworm
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10001 anpr && useradd --uid 10001 --gid anpr --create-home anpr
COPY docker/api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY api.py database.py ./
RUN mkdir -p /app/captures /app/data && chown -R anpr:anpr /app
USER anpr
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
