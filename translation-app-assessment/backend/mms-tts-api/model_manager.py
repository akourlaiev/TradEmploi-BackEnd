import os
import tarfile
import tempfile
import shutil
import requests
from pathlib import Path

from ttsmms import TTS

# Emplacement local des modèles (persisté entre déploiements de révision sur Cloud Run si tu actives le volume)
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/models"))

# URLs officielles Meta (S3) + fallback Hugging Face
META_BASE = "https://dl.fbaipublicfiles.com/mms/tts"
HF_BASE = "https://huggingface.co/facebook"

def _model_dir(lang: str) -> Path:
    return MODELS_DIR / lang

def _meta_url(lang: str) -> str:
    return f"{META_BASE}/{lang}.tar.gz"

def _hf_url(lang: str) -> str:
    # Sur HF, chaque langue a un repo ex: facebook/mms-tts-bod => on récupère l’archive .tar.gz via raw
    # On va plutôt cloner l’archive complète .tar.gz de Meta si dispo; sinon, on télécharge sur HF le snapshot tar.
    return f"{HF_BASE}/mms-tts-{lang}/resolve/main/{lang}.tar.gz"

def _download_and_extract(url: str, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_tar = Path(tmpd) / "model.tar.gz"
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_tar, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        with tarfile.open(tmp_tar, "r:gz") as tar:
            tar.extractall(dest_dir)

def _ensure_model(lang: str) -> Path:
    print(f"Ensuring model for lang={lang} is downloaded...")
    d = _model_dir(lang)
    print(f"Model directory: {d}")

    if d.exists() and ((d / "config.json").exists() or any(d.iterdir())):
        print(f"Model for lang={lang} already exists locally.")
        return d
    # Essai Meta
    try:
        print(f"Downloading model for lang={lang} from Meta...")
        _download_and_extract(_meta_url(lang), d)
    except Exception:
        # Fallback Hugging Face
        print(f"Meta download failed for lang={lang}, trying Hugging Face...")
        _download_and_extract(_hf_url(lang), d)
    return d

_tts_cache = {}

def get_tts_for_lang(lang: str) -> TTS:
    lang = lang.strip().lower()
    if lang not in _tts_cache:
        print(f"Model for lang={lang} not in cache, ensuring download...")
        model_dir = _ensure_model(lang)
        model_dir = str(model_dir)+"/"+lang
        print(f"Loading TTS model for lang={lang} from {model_dir}")
        _tts_cache[lang] = TTS(str(model_dir))
    return _tts_cache[lang]
