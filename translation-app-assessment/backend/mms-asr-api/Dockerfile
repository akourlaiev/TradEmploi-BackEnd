FROM python:3.11-slim

# Dépendances système minimales + zstd pour extraire .tar.zst
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ca-certificates tini zstd \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    MMS_MODEL_ID=facebook/mms-1b-all

WORKDIR /app

# --- PyTorch + torchaudio depuis l'index officiel (choix CPU/GPU au build) ---
# PT_DEVICE=cu121 pour GPU (L4/T4 - CUDA 12.1), PT_DEVICE=cpu pour build local CPU
ARG PT_DEVICE=cu121
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/${PT_DEVICE} \
    torch==2.5.1 torchaudio==2.5.1

# --- Le reste des deps Python ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Application ---
COPY app.py .
COPY run.sh .
RUN chmod +x /app/run.sh && mkdir -p /models

COPY models/ /models/

# Nettoyage léger (facultatif)
RUN find /usr/local -type d -name "__pycache__" -prune -exec rm -rf {} + || true

EXPOSE 8080
ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["/app/run.sh"]
