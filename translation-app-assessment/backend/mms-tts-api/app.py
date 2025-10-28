import io
import os
import traceback
from fastapi import FastAPI, HTTPException, Body, Response, Query
from pydantic import BaseModel, Field
from model_manager import get_tts_for_lang

app = FastAPI(title="Meta MMS TTS API", version="1.0.0")

class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à synthétiser")
    lang: str = Field(..., description="Code langue MMS (ISO 639-3, ex: 'bod' tibétain central, 'fra' français)")
    # Placeholders si tu veux plus tard gérer vitesse/voice, non utilisés par MMS:
    # speed: float | None = Field(None, gt=0)

@app.get("/healthz")
@app.get("/healthz/")
def healthz():
    return {"status": "ok"}

@app.post("/synthesize")
@app.post("/synthesize/")
def synthesize(req: SynthesisRequest):
    try:
        print(f"Received synthesis request: lang={req.lang}, text length={len(req.text)}")
        tts = get_tts_for_lang(req.lang)
        print(f"Synthesizing text for lang={req.lang}")
        wav = tts.synthesis(req.text)
        print("Synthesis complete")
        data = wav["x"]  # numpy array float32
        print("Extracted audio data")
        sr = wav["sampling_rate"]
        print(f"Sample rate: {sr}")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS error: {e}")

    # Encode en WAV 16-bit PCM
    #try:
        #import soundfile as sf
        #buf = io.BytesIO()
        #sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        #audio_bytes = buf.getvalue()
    # Encode en WAV 16-bit PCM (sans soundfile, 100% stdlib)
    try:
        import numpy as np
        import io, wave

        x = wav["x"]
        sr = wav["sampling_rate"]

        # x: float32 [-1.0, 1.0] (mono). Normalise + convertit en int16 LE
        x = np.asarray(x, dtype=np.float32)
        x = np.clip(x, -1.0, 1.0)
        pcm16 = (x * 32767.0).astype("<i2")  # int16 little-endian

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)           # MMS TTS est mono
            wf.setsampwidth(2)           # 16-bit PCM
            wf.setframerate(int(sr))
            wf.writeframes(pcm16.tobytes())

        audio_bytes = buf.getvalue()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio encoding error: {e}")

    headers = {
        "Content-Type": "audio/wav",
        "X-Sample-Rate": str(sr),
        "Content-Disposition": 'inline; filename="speech.wav"',
        "Cache-Control": "no-store",
    }
    return Response(content=audio_bytes, media_type="audio/wav", headers=headers)

@app.get("/supported")
def supported():
    # Liste ultra-simplifiée ; les codes MMS suivent majoritairement ISO 639-3 (ex: bod, fra).
    # Pour la liste complète, consulte l’index des dépôts HF facebook/mms-tts-<lang>. 
    return {
        "hint": "Codes MMS ~ ISO 639-3 (ex: bod, fra). Voir Hugging Face facebook/mms-tts-*.",
        "examples": ["bod", "fra", "eng", "deu", "spa"]
    }
