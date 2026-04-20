import re

import librosa
import numpy as np

_FILLERS = ["um", "uh", "like", "you know", "basically"]


def extract_audio_features(audio_path: str) -> dict:
    """
    Extract communication-quality features using librosa.
    Returns: mfcc_mean (13 coeffs), pitch_mean, pitch_std, speaking_rate,
             pause_ratio, snr_db
    """
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    # 13 MFCC coefficients — voice quality fingerprint
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1).tolist()

    # Fundamental frequency (pitch) using YIN algorithm
    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    voiced_f0 = f0[f0 > 0]
    pitch_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

    # Speaking rate: number of voiced frames / total duration
    rms = librosa.feature.rms(y=y)[0]
    silence_threshold = 0.02
    voiced_frames = (rms > silence_threshold).sum()
    total_frames = len(rms)
    speaking_rate = float(voiced_frames / total_frames) if total_frames > 0 else 0.0
    pause_ratio = 1.0 - speaking_rate

    # Signal-to-Noise Ratio: ratio of voiced to silent RMS in dB
    voiced_rms = rms[rms > silence_threshold]
    silent_rms = rms[rms <= silence_threshold]
    if len(voiced_rms) > 0 and len(silent_rms) > 0:
        snr_db = float(20 * np.log10(np.mean(voiced_rms) / (np.mean(silent_rms) + 1e-10)))
    else:
        snr_db = 0.0

    return {
        "mfcc_mean": mfcc_mean,
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "speaking_rate": speaking_rate,
        "pause_ratio": pause_ratio,
        "snr_db": snr_db,
    }


def detect_filler_words(transcript: str, duration_seconds: float) -> dict:
    """Count filler words in transcript and compute rate per minute."""
    text = transcript.lower()
    counts: dict[str, int] = {}
    for filler in _FILLERS:
        pattern = r"\b" + re.escape(filler) + r"\b"
        counts[filler] = len(re.findall(pattern, text))

    total = sum(counts.values())
    duration_minutes = max(duration_seconds / 60.0, 1e-6)
    return {
        "filler_count": total,
        "filler_rate": round(total / duration_minutes, 3),
        "filler_words": counts,
    }
