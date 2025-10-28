#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Utils ----------

def find_wavs(root: Path, pattern: str) -> list[Path]:
    if pattern:
        return sorted(root.rglob(pattern))
    return sorted(p for p in root.rglob("*.wav") if p.is_file())

def load_existing(csv_path: Path) -> set[str]:
    """Retourne l'ensemble des chemins déjà traités (pour reprise)."""
    done = set()
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("filepath"):
                    done.add(row["filepath"])
    return done

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def json_lenient(text: str) -> Dict[str, Any]:
    """
    Tente de parser du JSON même s'il y a des caractères parasites à la fin (ex: '%').
    Cherche la dernière '}' et coupe.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        idx = text.rfind("}")
        if idx != -1:
            try:
                return json.loads(text[:idx+1])
            except json.JSONDecodeError:
                pass
        raise

# ---------- API Call ----------

def call_asr(api_url: str, wav_path: Path, lang: str, timeout: float, max_retries: int = 3) -> Dict[str, Any]:
    """
    Envoie le fichier WAV à l'API ASR avec retries exponentiels.
    Retourne le dict JSON de la réponse.
    """
    files = {"file": (wav_path.name, wav_path.open("rb"), "audio/wav")}
    data = {"lang": lang}

    last_err: Optional[str] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(api_url, files=files, data=data, timeout=timeout)
            # Certains serveurs gardent le stream ouvert: forçons lecture complète
            content = resp.text
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {content[:200]}")
            return json_lenient(content)
        except Exception as e:
            last_err = str(e)
            # petit backoff
            time.sleep(min(2 ** attempt, 10))
        finally:
            # Rewind le fichier pour un potentiel retry
            try:
                files["file"][1].seek(0)
            except Exception:
                pass

    raise RuntimeError(f"Echec après {max_retries} tentatives: {last_err}")

# ---------- CSV ----------

CSV_FIELDS = [
    "filepath",
    "filename",
    "lang",
    "text",
    "text_corrected",
    "num_samples",
    "sampling_rate",
    "model",
    "llm_model",
    "audio_duration_s",
    "asr_duration_s",
    "llm_duration_s",
    "total_duration_s",
    "status",
    "error",
]

def write_csv_header_if_needed(csv_path: Path):
    if not csv_path.exists():
        ensure_parent(csv_path)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()

def append_row(csv_path: Path, row: Dict[str, Any]):
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)

# ---------- Worker ----------

def process_one(wav_path: Path, api_url: str, lang: str, timeout: float) -> Dict[str, Any]:
    try:
        res = call_asr(api_url, wav_path, lang, timeout=timeout)
        row = {
            "filepath": str(wav_path.resolve()),
            "filename": wav_path.name,
            "lang": res.get("lang", lang),
            "text": res.get("text", ""),
            "text_corrected": res.get("text_corrected", ""),
            "num_samples": res.get("num_samples", ""),
            "sampling_rate": res.get("sampling_rate", ""),
            "model": res.get("model", ""),
            "llm_model": res.get("llm_model", ""),
            "audio_duration_s": res.get("audio_duration_s", ""),
            "asr_duration_s": res.get("asr_duration_s", ""),
            "llm_duration_s": res.get("llm_duration_s", ""),
            "total_duration_s": res.get("total_duration_s", ""),
            "status": "ok",
            "error": "",
        }
        print(f"Processed {wav_path}: {row['text'][:30]}...")
        return row
    except Exception as e:
        return {
            "filepath": str(wav_path.resolve()),
            "filename": wav_path.name,
            "lang": lang,
            "text": "",
            "text_corrected": "",
            "num_samples": "",
            "sampling_rate": "",
            "model": "",
            "llm_model": "",
            "audio_duration_s": "",
            "asr_duration_s": "",
            "llm_duration_s": "",
            "total_duration_s": "",
            "status": "error",
            "error": str(e),
        }

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Batch ASR: parcourt un dossier, appelle l'API ASR, exporte un CSV."
    )
    parser.add_argument("--dir", required=True, help="Répertoire racine contenant les .wav")
    parser.add_argument("--api-url", default="http://localhost:8080/asr", help="URL de l'endpoint ASR")
    parser.add_argument("--lang", default="fra", help="Code langue à poster (ex: 'fra')")
    parser.add_argument("--out", default="asr_results.csv", help="Chemin du CSV de sortie")
    parser.add_argument("--pattern", default="", help="Pattern glob relatif (ex: '**/*.wav' ou '*.wav')")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="Nombre de threads (par défaut: min(8, CPU))")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout requête (sec)")
    args = parser.parse_args()

    root = Path(args.dir).expanduser().resolve()
    csv_path = Path(args.out).expanduser().resolve()

    if not root.exists():
        print(f"Erreur: le répertoire {root} n'existe pas.", file=sys.stderr)
        sys.exit(1)

    wavs = find_wavs(root, args.pattern)
    if not wavs:
        print("Aucun fichier .wav trouvé.")
        sys.exit(0)

    write_csv_header_if_needed(csv_path)
    already = load_existing(csv_path)
    to_process = [p for p in wavs if str(p.resolve()) not in already]

    total = len(wavs)
    todo = len(to_process)
    skipped = total - todo
    print(f"Trouvés: {total} wav | À traiter: {todo} | Ignorés (déjà en CSV): {skipped}")

    # Parallélisation
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(process_one, p, args.api_url, args.lang, args.timeout): p for p in to_process}
        done_count = 0
        err_count = 0
        try:
            for fut in as_completed(futures):
                row = fut.result()
                append_row(csv_path, row)
                done_count += 1
                if row["status"] != "ok":
                    err_count += 1
                if done_count % 10 == 0 or done_count == todo:
                    print(f"[{done_count}/{todo}] terminés (erreurs: {err_count}) -> {csv_path.name}")
        except KeyboardInterrupt:
            print("\nInterrompu par l'utilisateur. Les résultats déjà écrits restent dans le CSV.")
            sys.exit(130)

    print(f"Terminé. Résultats dans: {csv_path}")
    if skipped:
        print(f"({skipped} entrées déjà présentes ont été ignorées)")

if __name__ == "__main__":
    main()
