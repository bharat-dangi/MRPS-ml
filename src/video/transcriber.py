import logging
import os
from functools import lru_cache
from typing import Any

import torch
import whisper

logger = logging.getLogger(__name__)

# Use large-v3 on GPU; fall back to medium on CPU to keep latency acceptable
_GPU_MODEL = "large-v3"
_CPU_MODEL = os.getenv("WHISPER_MODEL", "medium")


@lru_cache(maxsize=1)
def _load_model() -> whisper.Whisper:
    if torch.cuda.is_available():
        logger.info("Loading Whisper %s on GPU", _GPU_MODEL)
        return whisper.load_model(_GPU_MODEL, device="cuda")
    logger.info("CUDA not available — loading Whisper %s on CPU", _CPU_MODEL)
    return whisper.load_model(_CPU_MODEL, device="cpu")


def transcribe(audio_path: str) -> dict[str, Any]:
    """
    Transcribe audio with Whisper.
    Returns: {transcript, confidence, language, word_timestamps}
    """
    model = _load_model()
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        verbose=False,
    )

    # Estimate confidence as mean of per-segment no-speech probability inverted
    segments = result.get("segments", [])
    if segments:
        no_speech_probs = [s.get("no_speech_prob", 0.0) for s in segments]
        confidence = 1.0 - (sum(no_speech_probs) / len(no_speech_probs))
    else:
        confidence = 0.0

    return {
        "transcript": result["text"].strip(),
        "confidence": round(confidence, 4),
        "language": result.get("language", "en"),
        "word_timestamps": [
            {"word": w["word"], "start": w["start"], "end": w["end"]}
            for seg in segments
            for w in seg.get("words", [])
        ],
    }
