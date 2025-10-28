# app.py
import io
import os
import time
import logging
import torch
import numpy as np
import soundfile as sf
import torchaudio
import psutil
import requests

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from transformers import AutoProcessor, AutoModelForCTC, logging as hf_logging

# ---------------------------
# Logging setup
# ---------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

log = logging.getLogger("mms-asr")
hf_logging.set_verbosity_error()

# Optionnel: réduire le sur-threading CPU
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

APP_TITLE = "MMS ASR API"
MODEL_ID = os.getenv("MMS_MODEL_ID", "facebook/mms-1b-all")  # multilingual ASR

HF_HOME = os.getenv("HF_HOME", "/tmp/models")
os.environ["HF_HOME"] = HF_HOME
os.makedirs(HF_HOME, exist_ok=True)

MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", os.path.join(HF_HOME, "facebook", "mms-1b-all"))

ENABLE_LLM_CORRECTION = os.getenv("ENABLE_LLM_CORRECTION", "0") == "1"  # Activer/désactiver la correction LLM
LLM_MODEL_ID = "gpt-oss-20B"  # Modèle LLM pour la correction
GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY", "secretemploi")  # Clé API pour GPT-OSS
LLM_MODEL_ENDPOINT = os.getenv("LLM_MODEL_ENDPOINT", "https://gpt-oss-20-b-615733745472.europe-west1.run.app/v1/chat/completions")

#LLM_MODEL_ID = "ggml-gpt4all-j-v1.3-groovy"  # Modèle LLM pour la correction

app = FastAPI(title=APP_TITLE)
_model = None
_processor = None
_current_lang = None

def mem_info():
    """Retourne un dict avec la RAM/CPU et (si dispo) mémoire GPU."""
    p = psutil.Process(os.getpid())
    rss_mb = p.memory_info().rss / (1024**2)
    cpu_pct = psutil.cpu_percent(interval=None)
    info = {"rss_mb": round(rss_mb, 1), "cpu_pct": cpu_pct}
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            info.update({
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_mem_alloc_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 1),
                "gpu_mem_reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 1),
            })
        except Exception as e:
            info.update({"gpu": f"error: {e}"})
    return info

class StepTimer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.perf_counter()
        log.info(f"[START] {self.name} | {mem_info()}")
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt = (time.perf_counter() - self.t0) * 1000.0
        if exc_type:
            log.exception(f"[ERROR] {self.name} failed after {dt:.1f} ms")
        else:
            log.info(f"[END] {self.name} | {dt:.1f} ms | {mem_info()}")

# ---------------------------
# Request/response timing middleware
# ---------------------------
@app.middleware("http")
async def add_timing(request: Request, call_next):
    rid = f"{int(time.time()*1000)}-{os.getpid()}"
    log.info(f"[REQ] id={rid} {request.method} {request.url.path} | {mem_info()}")
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        dt = (time.perf_counter() - t0) * 1000.0
        log.info(f"[RES] id={rid} status={response.status_code} time_ms={dt:.1f} | {mem_info()}")
        return response
    except Exception:
        dt = (time.perf_counter() - t0) * 1000.0
        log.exception(f"[RES] id={rid} EXC after {dt:.1f} ms")
        raise

# ---------------------------
# Model loading
# ---------------------------
def load_model(lang: str):
    global _model, _processor, _current_lang
    if _model is not None and _current_lang == lang:
        log.info(f"Model already loaded for lang={lang} | {mem_info()}")
        return

    with StepTimer(f"load_model(lang={lang})"):
        log.info(f"Loading processor: {MODEL_ID} (tokenizer_lang={lang})")
        _processor = AutoProcessor.from_pretrained(MODEL_ID, tokenizer_lang=lang)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" and os.getenv("FP16", "0") == "1" else torch.float32

        log.info(f"Loading model....: {MODEL_ID} | device={device} | dtype={dtype}")
        log.info(f"[debug] HF_HOME = {os.getenv('HF_HOME')}")
        log.info(f"[debug] MODEL_ID = {MODEL_ID}")
        _model = AutoModelForCTC.from_pretrained(MODEL_ID, torch_dtype=dtype)
        _model.eval()
        _model.to(device)
        _current_lang = lang

        # Log mémoire après chargement
        log.info(f"Model loaded for lang={lang} | {mem_info()}")

# ---------------------------
# Correction avec le LLM
# ---------------------------
# https://platform.openai.com/docs/models/gpt-oss-20b
# https://huggingface.co/openai/gpt-oss-20b
# https://cookbook.openai.com/articles/openai-harmony

def correct_with_llm(
    text: str,
    *,
    api_key: str = GPT_OSS_API_KEY,
    endpoint: str = LLM_MODEL_ENDPOINT,
    model: str = "gpt-oss-20b",
    temperature: float = 0.0,
    max_tokens: int = 350,
    lang: str = "fra",
    max_retries: int = 3,
    timeout: float = 60.0,
) -> str:
    """
    Corrige un texte via ton endpoint GPT-OSS (compatible OpenAI API).
    """
    if api_key is None:
        api_key = os.getenv("GPT_OSS_API_KEY")
    if not api_key:
        raise ValueError("Aucune clé API fournie (paramètre api_key ou env GPT_OSS_API_KEY).")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "Tu es un correcteur professionnel. "
        "Corrige uniquement l’orthographe, la grammaire et la ponctuation du texte suivant, "
        f"sans changer le style ni la signification. Langue cible : {lang}. "
        "Ne donne aucune explication, ne reformule pas — renvoie seulement le texte corrigé."
        "Reasoning: low\n# Valid channels: final. Channel must be included for every message."
    )

    user_prompt = f"Texte à corriger :\n{text}"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # gestion de retries basique
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            else:
                last_error = f"{response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)

        if attempt < max_retries:
            time.sleep(3 * attempt)

    raise KeyError(f"Erreur après {max_retries} tentatives: {last_error}. Le démarrage du service LLM peut prendre du temps (entre 5 et 10 mins).")

# ---------------------------
# Audio I/O
# ---------------------------
def read_audio(file_bytes: bytes, target_sr: int = 16000) -> np.ndarray:
    with StepTimer("read_audio/decode"):
        with io.BytesIO(file_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)

        # Log brut
        n_samples = audio.shape[0] if audio.ndim == 1 else audio.shape[0]
        dur_s = n_samples / float(sr)
        log.info(f"Decoded audio: sr={sr}Hz, channels={'mono' if audio.ndim==1 else audio.shape[1]}, "
                 f"samples={n_samples}, duration={dur_s:.2f}s | {mem_info()}")

        # mono
        if audio.ndim == 2:
            with StepTimer("stereo->mono"):
                audio = np.mean(audio, axis=1)

        waveform = torch.from_numpy(audio)

        # resample
        if sr != target_sr:
            with StepTimer(f"resample {sr}->{target_sr}"):
                waveform = torchaudio.functional.resample(waveform, sr, target_sr)

        # final info
        log.info(f"Audio ready: sr={target_sr}Hz, samples={waveform.numel()}, "
                 f"duration={waveform.numel()/target_sr:.2f}s")
        return waveform.numpy()

# ---------------------------
# Endpoints
# ---------------------------
@app.on_event("startup")
def _startup():
    log.info(f"[startup] HF_HOME={HF_HOME} MODEL_LOCAL_DIR={MODEL_LOCAL_DIR}")
    if ENABLE_LLM_CORRECTION:
        log.info(f"[startup] LLM correction is ENABLED using model {LLM_MODEL_ID}")

@app.post("/asr")
@app.post("/asr/")
async def asr(
    file: UploadFile = File(..., description="Audio file (wav/mp3/flac/ogg, etc.)"),
    lang: str = Form(..., description="Language code for tokenizer, e.g. 'fra', 'eng', 'ukr'"),
):
    start_time = time.perf_counter()

    try:
        with StepTimer("ensure_model"):
            load_model(lang)
    except Exception as e:
        log.exception("Model load failed")
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    # Lire le fichier
    with StepTimer("read_file_bytes"):
        file_bytes = await file.read()
        log.info(f"Uploaded file: name={file.filename} size_bytes={len(file_bytes)} content_type={file.content_type}")

    # Décodage / resample
    try:
        audio_16k = read_audio(file_bytes, 16000)
    except Exception as e:
        log.exception("Audio decode failed")
        raise HTTPException(status_code=400, detail=f"Audio decode failed: {e}")

    # Inference
    corrected_text = ""
    try:
        with StepTimer("preprocess"):
            inputs = _processor(audio=audio_16k, sampling_rate=16000, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with StepTimer("forward"):
            with torch.no_grad():
                logits = _model(**inputs).logits
        with StepTimer("ctc_decode"):
            pred_ids = torch.argmax(logits, dim=-1)
            text = _processor.batch_decode(pred_ids, skip_special_tokens=True)[0]

            # (facultatif) remplacements métier
            if lang.startswith(("fr", "fra", "fre")):
                REPLACEMENTS_FR = {
                    "mtiers": "métiers",
                    "persone": "personne",
                    "ofres": "offres",
                    "ofre": "offre",
                    "travaille": "travail",
                    "sesion": "session",
                    "dtection": "détection",
                    "bulant": "bilan",
                    "complmentaires": "complémentaires",
                    "metre": "mettre",
                    "judi": "jeudi",
                    "mtier": "métier",
                    "france travaile": "France Travail",
                    "genral de gol": "Général de Gaulle",
                    "batiment": "bâtiment",
                    "tage": "étage",
                    "arondisement": "arrondissement",
                    "dtp": "DTP",
                }
                for a, b in REPLACEMENTS_FR.items():
                    text = text.replace(a, b)
            
            asr_duration_s = (time.perf_counter() - start_time) * 1.0
            start_llm_time = time.perf_counter()
            
            # Correction par LLM
            if ENABLE_LLM_CORRECTION:
                corrected_text = correct_with_llm(text, lang=lang)
            else:
                corrected_text = text
            
            llm_duration_s = (time.perf_counter() - start_llm_time) * 1.0
            total_duration_s = asr_duration_s + llm_duration_s

        result = {
            "lang": lang,
            "text": text.strip(),
            "text_corrected": corrected_text.strip(),
            "num_samples": len(audio_16k),
            "sampling_rate": 16000,
            "model": MODEL_ID,
            "llm_model": LLM_MODEL_ID if ENABLE_LLM_CORRECTION else None,
            "audio_duration_s": round(len(audio_16k) / 16000.0, 1),
            "asr_duration_s": round(asr_duration_s, 1),  # à compléter si besoin
            "llm_duration_s": round(llm_duration_s, 1),  # à compléter si besoin
            "total_duration_s": round(total_duration_s, 1),  # à compléter si besoin
        }
        log.info(f"ASR done: chars={len(result['text'])} | {mem_info()}")
        return result
    except Exception as e:
        log.exception("ASR failed")
        raise HTTPException(status_code=500, detail=f"ASR failed: {e}")

@app.get("/healthz")
@app.get("/healthz/")
def healthz():
    info = {
        "status": "ok",
        "model": MODEL_ID,
        "lang_loaded": _current_lang or "",
        "llm_model": LLM_MODEL_ID if ENABLE_LLM_CORRECTION else None,
    }
    log.info(f"/healthz -> {info} | {mem_info()}")
    return info