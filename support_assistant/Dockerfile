# Task 6 - Dockerfile for the Zepto Support Assistant FastAPI app.
# Graded baseline: builds and runs locally, serving POST /ask in offline mock mode.
FROM python:3.11-slim

# Keep Python lean and predictable in a container.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MOCK_LLM=1 \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# Install CPU-only PyTorch first (small wheel) so sentence-transformers does not
# pull the large CUDA build, then the rest of the requirements.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + corpus.
COPY . .

# Bake the ChromaDB index into the image so the container is self-contained.
RUN python ingest.py

EXPOSE 8000

# Serve the API. MOCK_LLM defaults to 1 (graded offline path).
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
