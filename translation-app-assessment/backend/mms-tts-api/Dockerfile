# ---------- builder ----------
    FROM python:3.11-slim AS builder
    ENV PIP_NO_CACHE_DIR=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    WORKDIR /app
    
    # Outils pour compiler ce qui n'a pas de wheel (monotonic-align, etc.)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc g++ make pkg-config \
     && rm -rf /var/lib/apt/lists/*
    
    COPY requirements.txt .
    
    # Astuce PyTorch CPU: index officiel CPU pour éviter CUDA
    RUN pip install --upgrade pip setuptools wheel
    RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
    
    # On “gèle” les deps dans un dossier portable
    RUN python -c "import site,sys,shutil,os; p=[site.getsitepackages()[0], site.getusersitepackages()]; d='/opt/venv'; os.makedirs(d, exist_ok=True); [shutil.copytree(s, d, dirs_exist_ok=True) for s in p if os.path.exists(s)]"
    
    # ---------- final ----------
    FROM python:3.11-slim
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PORT=8080
    WORKDIR /app
    
    # Copie uniquement les paquets Python (sans gcc & co)
    COPY --from=builder /opt/venv /usr/local/lib/python3.11/site-packages
    
    # Si tu n’utilises plus soundfile/libsndfile, rien à ajouter.
    # (Sinon: RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ...)
    
    # Code app
    COPY . .
    
    # Cloud Run impose la variable $PORT
    ENV PORT=8080

    EXPOSE 8080
    #CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
    #CMD ["sh","-c","python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
    CMD ["sh","-c","python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 2 --proxy-headers --forwarded-allow-ips='*'"]

