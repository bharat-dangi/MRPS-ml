"""
Whisper WER benchmark across accent groups.

Synthesizes 5 audio clips of the same reference transcript using macOS `say`
with different accent voices, then transcribes each with Whisper and reports
Word Error Rate.

Reference voices (en_AU, en_US, en_GB, en_IN, plus a non-native speaker
reading English via the Spanish-language en speaker):
    Karen   en_AU   Australian English
    Eddy    en_US   US English
    Eddy    en_GB   UK English
    Aman    en_IN   Indian English
    Mónica  es_ES   Non-native (Spanish L1)

CSV target: WER ≤ 15% overall. Any accent group >20% triggers a mitigation
note in the report.

This is NOT a substitute for a Common Voice benchmark — it's a deterministic
in-pipeline check that the codebase can run anywhere without external data.
A Common Voice run would use real human voices and yield more conservative
numbers. The script's structure (reference → audio → transcript → WER) is
identical for both.

Output: ml/eval/whisper_wer.json
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import jiwer
import whisper

ML_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ML_ROOT / "eval" / "whisper_wer.json"
OUT_PATH.parent.mkdir(exist_ok=True)

REFERENCE = (
    "Hello, my name is Jordan. I have over seven years of experience as a "
    "backend engineer working with Python, FastAPI, and PostgreSQL. "
    "I am applying for the senior backend engineer position. "
    "Thank you for considering my application."
)

ACCENTS = [
    ("Australian English", "Karen"),
    ("US English", "Eddy (English (US))"),
    ("UK English", "Eddy (English (UK))"),
    ("Indian English", "Aman"),
    ("Non-native (Spanish L1)", "Mónica"),
]


def _synth(voice: str, text: str, out_path: Path) -> bool:
    """Synthesize speech via macOS `say`. Returns True on success."""
    if not shutil.which("say"):
        return False
    aiff_path = out_path.with_suffix(".aiff")
    proc = subprocess.run(
        ["say", "-v", voice, "-o", str(aiff_path), text],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        print(f"  ! say voice {voice!r}: rc={proc.returncode}: {proc.stderr.strip()}")
        return False
    # Convert AIFF → WAV 16kHz mono so Whisper has a consistent input.
    if not shutil.which("ffmpeg"):
        return False
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(aiff_path), "-ar", "16000", "-ac", "1", str(out_path)],
        capture_output=True, text=True, check=False,
    )
    aiff_path.unlink(missing_ok=True)
    return out_path.exists()


def main() -> dict:
    if not shutil.which("say") or not shutil.which("ffmpeg"):
        raise SystemExit(
            "This benchmark needs macOS `say` + ffmpeg. Install ffmpeg with "
            "`brew install ffmpeg` and re-run on macOS."
        )

    print("Loading Whisper model (tiny.en for speed; the production pipeline uses large-v3)")
    model = whisper.load_model("tiny.en")

    results: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for label, voice in ACCENTS:
            print(f"\n[{label}] voice={voice!r}")
            wav_path = tmp_dir / f"{label.replace(' ', '_')}.wav"
            if not _synth(voice, REFERENCE, wav_path):
                print(f"  ! skipping — synthesis failed for {voice}")
                continue
            t0 = time.time()
            transcript = model.transcribe(str(wav_path), language="en", fp16=False)["text"].strip()
            duration = time.time() - t0
            wer = jiwer.wer(
                jiwer.transforms.Compose([
                    jiwer.transforms.ToLowerCase(),
                    jiwer.transforms.RemovePunctuation(),
                    jiwer.transforms.RemoveMultipleSpaces(),
                    jiwer.transforms.Strip(),
                ])(REFERENCE),
                jiwer.transforms.Compose([
                    jiwer.transforms.ToLowerCase(),
                    jiwer.transforms.RemovePunctuation(),
                    jiwer.transforms.RemoveMultipleSpaces(),
                    jiwer.transforms.Strip(),
                ])(transcript),
            )
            print(f"  transcript: {transcript[:120]}{'...' if len(transcript) > 120 else ''}")
            print(f"  WER: {wer:.3f}  ({duration:.2f}s)")
            results[label] = {
                "voice": voice,
                "transcript": transcript,
                "wer": wer,
                "duration_seconds": duration,
            }

    if results:
        overall_wer = sum(r["wer"] for r in results.values()) / len(results)
        print(f"\nOverall mean WER: {overall_wer:.3f}  (target ≤ 0.15)")
        worst_accent = max(results.items(), key=lambda kv: kv[1]["wer"])
        print(f"Worst accent:     {worst_accent[0]} @ {worst_accent[1]['wer']:.3f}")
    else:
        overall_wer = None

    payload = {
        "model": "tiny.en (production pipeline uses large-v3)",
        "reference": REFERENCE,
        "per_accent": results,
        "overall_wer": overall_wer,
        "target_wer": 0.15,
        "warn_threshold": 0.20,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {OUT_PATH.relative_to(Path.cwd())}")
    return payload


if __name__ == "__main__":
    main()
