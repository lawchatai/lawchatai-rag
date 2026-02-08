# ---- Base image ----
FROM python:3.12-slim

# ---- System deps (needed for torch / faiss / pdf parsing) ----
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# HuggingFace cache (model preloading + runtime)
ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/root/.cache/huggingface

# ---- Workdir ----
WORKDIR /app

# ---- Install Python deps first (better layer caching) ----
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- 🔥 Pre-download embedding model (OPTION 3) ----
RUN python - <<EOF
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
print("✅ Embedding model cached")
EOF

# ---- Copy application code ----
COPY . .

# ---- Expose port (Render ignores, local use only) ----
EXPOSE 8000

# ---- Start FastAPI (Render-compatible) ----
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
