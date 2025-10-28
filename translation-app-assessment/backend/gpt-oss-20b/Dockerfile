# Image CUDA de llama.cpp (serveur + outils)
# Variantes possibles : full-cuda / server-cuda selon tags publiés
#FROM ghcr.io/ggml-org/llama.cpp:full-cuda
FROM ghcr.io/ggml-org/llama.cpp:server-cuda

# Dossier où le bucket GCS sera monté (volume Cloud Run)
ENV PATH="/app:${PATH}"
ENV MODEL_DIR=/models
ENV PORT=8080

# Santé basique (Cloud Run lit bien les codes HTTP)
#HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -sf http://127.0.0.1:${PORT}/health || exit 1

# Entrypoint (OpenAI-compatible via llama-server)
# -ngl 999 : offload max couches sur GPU (L4)
# -c 8192  : contexte (ajustez selon VRAM)
# -np 4    : requêtes parallèles
# --api-key : protège l'API (clé arbitraire)
COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/bin/bash","/entrypoint.sh"]
