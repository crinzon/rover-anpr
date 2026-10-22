FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart

COPY api.py database.py ./

RUN mkdir -p /app/captures /app/data

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
