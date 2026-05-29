"""
End-to-end smoke test for the video pipeline.

Synthesizes a short MP4 *and* a WebM clip (TTS audio + still-frame video) and
runs each through the production pipeline:

    extract_audio (ffmpeg)
    └─> transcribe (Whisper)
    └─> extract_audio_features (librosa)
    └─> compute_video_score (semantic + clarity + coverage)

If either format fails at any step, the script exits non-zero with a clear
error. Lives under `ml/scripts/` so it can be wired into CI as a regression
gate alongside `run_eval.py` and friends.

Usage:
    cd ml && source venv/bin/activate
    WHISPER_MODEL=base.en PYTHONPATH=. python scripts/smoke_test_video_pipeline.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ROOT))

from src.video.audio_features import extract_audio_features  # noqa: E402
from src.video.preprocessor import extract_audio  # noqa: E402
from src.video.transcriber import transcribe  # noqa: E402
from src.video.video_scorer import compute_video_score  # noqa: E402

REFERENCE_TRANSCRIPT = (
    "Hi, my name is Casey. I have over five years of experience as a backend "
    "engineer working with Python, FastAPI, and PostgreSQL. Thank you for "
    "your time."
)
JD_TEXT = (
    "Senior backend engineer. Required: Python, FastAPI, PostgreSQL, Docker. "
    "Minimum five years of experience."
)
JD_SKILLS = ["Python", "FastAPI", "PostgreSQL", "Docker"]


def _check_tools() -> None:
    missing = [t for t in ("say", "ffmpeg") if not shutil.which(t)]
    if missing:
        raise SystemExit(
            f"Missing tools: {missing}. This smoke test needs macOS `say` + ffmpeg."
        )


def _synth_clip(target_path: Path, encoder_args: list[str]) -> None:
    """Render a short clip with TTS audio + a 720x720 still frame in the
    requested container. `encoder_args` selects the codec set."""
    tmp_dir = target_path.parent
    aiff_path = tmp_dir / "tts.aiff"

    # 1) TTS audio
    subprocess.run(
        ["say", "-v", "Daniel", "-o", str(aiff_path), REFERENCE_TRANSCRIPT],
        check=True, capture_output=True,
    )

    # 2) Convert to the target container with a synthetic blue background
    ff_args = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x4338ca:s=720x720:r=24",  # still frame
        "-i", str(aiff_path),
        "-shortest",
        *encoder_args,
        str(target_path),
    ]
    proc = subprocess.run(ff_args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed for {target_path}:\n{proc.stderr[-800:]}")
    aiff_path.unlink(missing_ok=True)


def _run_pipeline(video_path: Path, label: str) -> dict:
    """Run the production video pipeline against a single file. Returns the
    score breakdown if everything works; raises otherwise."""
    print(f"\n[{label}] {video_path.name}  ({video_path.stat().st_size / 1024:.1f} KB)")

    print("  → extract_audio (ffmpeg) ...", end=" ", flush=True)
    audio_path = extract_audio(str(video_path))
    print("ok")

    try:
        print("  → transcribe (Whisper) ...", end=" ", flush=True)
        tr = transcribe(audio_path)
        print(f"ok  ({len(tr['transcript'])} chars, conf={tr['confidence']:.2f})")

        print("  → extract_audio_features (librosa) ...", end=" ", flush=True)
        feats = extract_audio_features(audio_path)
        print(f"ok  ({len(feats)} features)")

        print("  → compute_video_score ...", end=" ", flush=True)
        score = compute_video_score(
            transcript=tr["transcript"],
            jd_text=JD_TEXT,
            audio_features=feats,
            jd_skills=JD_SKILLS,
            word_timestamps=tr.get("word_timestamps"),
        )
        print(f"ok  video_score={score['video_score']:.3f}")
        print(f"     transcript_sim={score['transcript_sim']:.3f}  "
              f"comm_clarity={score['comm_clarity']:.3f}  "
              f"content_coverage={score['content_coverage']:.3f}")
        return score
    finally:
        Path(audio_path).unlink(missing_ok=True)


def main() -> None:
    _check_tools()

    with tempfile.TemporaryDirectory(prefix="video-smoke-") as tmp:
        tmp_dir = Path(tmp)
        cases = [
            (
                tmp_dir / "sample.mp4",
                # H.264 video + AAC audio in an MP4 container — the common case
                ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"],
            ),
            (
                tmp_dir / "sample.webm",
                # VP9 video + Opus audio in a WebM container
                ["-c:v", "libvpx-vp9", "-b:v", "300k", "-c:a", "libopus", "-b:a", "96k"],
            ),
        ]

        results: dict[str, dict] = {}
        for path, encoder_args in cases:
            label = path.suffix.lstrip(".").upper()
            _synth_clip(path, encoder_args)
            results[label] = _run_pipeline(path, label)

    print("\n─────────────────────────────────────────────────")
    print("  All formats processed successfully.")
    print("─────────────────────────────────────────────────")
    for label, r in results.items():
        print(f"  {label:5s} video_score={r['video_score']:.3f}  "
              f"sim={r['transcript_sim']:.3f}  clarity={r['comm_clarity']:.3f}")


if __name__ == "__main__":
    main()
