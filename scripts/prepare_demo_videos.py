"""
One-shot script: download 6 CC-BY YouTube video resumes, upload to S3, and
run them through the real video pipeline (ffmpeg → Whisper → librosa →
video_scorer). Result is cached as JSON for `fast_seed.py` to consume.

All source videos are licensed CC-BY (Creative Commons Attribution) per
YouTube's metadata at the time of download — verified by yt-dlp's
%(license)s field. Attribution is written to data/demo_videos/CREDITS.md.

Each video is paired with one of the 6 demo job postings — the JD's text
and required-skill list are fed to the scorer so transcript_sim and
content_coverage are computed against the real JD.

Usage:
    cd ml && source venv/bin/activate
    WHISPER_MODEL=base.en PYTHONPATH=. python scripts/prepare_demo_videos.py

Notes:
    * Set WHISPER_MODEL=base.en (74M) for a ~5–10 min total run on CPU.
      Default is `medium` (769M) which is ~30–60 min for 6 short clips.
    * Re-running the script is idempotent — already-processed videos are
      skipped unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import load_dotenv

ML_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ROOT))
load_dotenv(ML_ROOT / ".env")

from src.video.audio_features import extract_audio_features  # noqa: E402
from src.video.preprocessor import extract_audio  # noqa: E402
from src.video.transcriber import transcribe  # noqa: E402
from src.video.video_scorer import compute_video_score  # noqa: E402

CACHE_DIR = ML_ROOT / "data" / "demo_videos"
CACHE_PATH = CACHE_DIR / "seed_videos.json"
CREDITS_PATH = CACHE_DIR / "CREDITS.md"
S3_PREFIX = "videos/demo-seed"


@dataclass
class DemoVideo:
    youtube_id: str
    creator: str
    job_title: str  # the JD this video is paired with in the seed
    jd_text: str
    jd_required_skills: list[str]
    duration_s: int


# Source URLs are all confirmed CC-BY per yt-dlp's %(license)s output on
# 2026-05-23. License: Creative Commons Attribution license (reuse allowed).
SOURCES: list[DemoVideo] = [
    DemoVideo(
        youtube_id="MiulJCYLGI0",
        creator="Eduardo Morales",
        job_title="Senior Backend Engineer",
        jd_text=(
            "We need an experienced backend engineer to build scalable services. "
            "Required skills: Python, FastAPI, PostgreSQL, Docker, REST API. "
            "Preferred: Kubernetes, Redis, AWS. Minimum 5 years experience."
        ),
        jd_required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "REST API"],
        duration_s=153,
    ),
    DemoVideo(
        youtube_id="teMWt6BYg9E",
        creator="Tomislav Pajtak",
        job_title="Machine Learning Engineer",
        jd_text=(
            "Looking for an ML engineer to build production ML systems. "
            "Required: Python, PyTorch, machine learning, MLOps, AWS. "
            "Preferred: Kubernetes, MLflow, Spark. Minimum 4 years experience."
        ),
        jd_required_skills=["Python", "PyTorch", "machine learning", "MLOps", "AWS"],
        duration_s=85,
    ),
    DemoVideo(
        youtube_id="lTxfdl6HpT4",
        creator="Ali",
        job_title="Financial Analyst",
        jd_text=(
            "Support financial planning and reporting. Required: financial "
            "analysis, financial modeling, budgeting, variance analysis, "
            "Microsoft Excel. Preferred: Power BI, SQL, financial forecasting."
        ),
        jd_required_skills=[
            "financial analysis", "financial modeling", "budgeting",
            "variance analysis", "Microsoft Excel",
        ],
        duration_s=132,
    ),
    DemoVideo(
        youtube_id="trfJlgMCRiw",
        creator="Stephanie Plumeri",
        job_title="Digital Marketing Manager",
        jd_text=(
            "Lead digital marketing strategy and campaigns. Required: SEO, "
            "Google Analytics, content marketing, social media marketing, "
            "campaign management. Preferred: HubSpot, A/B testing, copywriting."
        ),
        jd_required_skills=[
            "SEO", "Google Analytics", "content marketing",
            "social media marketing", "campaign management",
        ],
        duration_s=114,
    ),
    DemoVideo(
        youtube_id="2NWH0566dsI",
        creator="Nur Hidayah Sazali",
        job_title="Registered Nurse",
        jd_text=(
            "Provide direct patient care in a hospital setting. Required: "
            "patient care, clinical assessment, IV therapy, electronic health "
            "records, medication administration. Preferred: ACLS, telemetry."
        ),
        jd_required_skills=[
            "patient care", "clinical assessment", "IV therapy",
            "electronic health records", "medication administration",
        ],
        duration_s=93,
    ),
    DemoVideo(
        youtube_id="7h6WRTqEo20",
        creator="Nurul Shakirah Norhan",
        job_title="HR Business Partner",
        jd_text=(
            "Partner with business leaders on people strategy. Required: "
            "employee relations, performance management, talent management, "
            "HR policy, workforce planning. Preferred: HRIS, coaching."
        ),
        jd_required_skills=[
            "employee relations", "performance management",
            "talent management", "HR policy", "workforce planning",
        ],
        duration_s=76,
    ),
]


def _s3_client():
    region = os.getenv("AWS_REGION", "ap-southeast-2")
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
    )


def _s3_bucket() -> str:
    return os.getenv("S3_BUCKET", "resume-screener-upload")


def _yt_download(youtube_id: str, dest_dir: Path) -> Path:
    """Download a YouTube video to {dest_dir}/{youtube_id}.mp4 using yt-dlp.
    Always merges to MP4 + H.264 + AAC so the downstream pipeline gets a
    consistent input format."""
    import subprocess
    out_path = dest_dir / f"{youtube_id}.mp4"
    if out_path.exists():
        print(f"  already downloaded: {out_path.name}")
        return out_path
    # mp4 single-file format (bestaudio[ext=m4a]+bestvideo[ext=mp4]) → merge to mp4
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path.with_suffix(".%(ext)s")),
        f"https://www.youtube.com/watch?v={youtube_id}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        print(proc.stderr[-1000:])
        raise SystemExit(f"yt-dlp failed for {youtube_id}")
    if not out_path.exists():
        # yt-dlp may have chosen a different extension; rename if needed
        candidates = list(dest_dir.glob(f"{youtube_id}.*"))
        if not candidates:
            raise SystemExit(f"yt-dlp produced no output for {youtube_id}")
        candidates[0].rename(out_path)
    return out_path


def _s3_upload(local_path: Path, s3_key: str) -> str:
    s3 = _s3_client()
    bucket = _s3_bucket()
    try:
        s3.head_object(Bucket=bucket, Key=s3_key)
        print(f"  already in S3: {s3_key}")
        return s3_key
    except s3.exceptions.ClientError:
        pass
    s3.upload_file(str(local_path), bucket, s3_key,
                   ExtraArgs={"ContentType": "video/mp4"})
    print(f"  uploaded → s3://{bucket}/{s3_key}")
    return s3_key


def _process_one(video_path: Path, source: DemoVideo) -> dict:
    print(f"  extracting audio ...", end=" ", flush=True)
    t0 = time.time()
    audio_path = extract_audio(str(video_path))
    print(f"{time.time()-t0:.1f}s")
    try:
        print(f"  transcribing (Whisper) ...", end=" ", flush=True)
        t0 = time.time()
        transcript_result = transcribe(audio_path)
        print(f"{time.time()-t0:.1f}s | conf={transcript_result['confidence']:.2f}")

        print(f"  audio features (librosa) ...", end=" ", flush=True)
        t0 = time.time()
        audio_features = extract_audio_features(audio_path)
        print(f"{time.time()-t0:.1f}s")

        print(f"  scoring against JD ...", end=" ", flush=True)
        word_timestamps = transcript_result.get("word_timestamps")
        score_result = compute_video_score(
            transcript=transcript_result["transcript"],
            jd_text=source.jd_text,
            audio_features=audio_features,
            jd_skills=source.jd_required_skills,
            word_timestamps=word_timestamps,
        )
        print(f"video_score={score_result['video_score']:.3f}")

        full_features = {**audio_features, **{
            k: score_result[k]
            for k in ("transcript_sim", "comm_clarity", "content_coverage",
                      "pace_score", "fluency_score", "articulation_score",
                      "filler_rate", "wpm")
            if k in score_result
        }}
        if word_timestamps:
            full_features["word_timestamps"] = word_timestamps[:200]  # cap to keep JSON small

        return {
            "transcript": transcript_result["transcript"],
            "transcript_confidence": transcript_result["confidence"],
            "audio_features": full_features,
            "video_score": score_result["video_score"],
            "skill_sources_from_video": score_result.get("skill_sources", {}),
        }
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def write_credits(processed: list[dict]) -> None:
    """CC-BY requires attribution. Drop a CREDITS.md next to the data."""
    lines = [
        "# Demo video sources (CC-BY attribution)",
        "",
        "The 6 video resumes used in the demo seed are sourced from YouTube and",
        "are licensed under **Creative Commons Attribution license (reuse allowed)** —",
        "verified via `yt-dlp --print %(license)s` at the time of download.",
        "",
        "Per CC-BY, the original creators are credited below.",
        "",
        "| # | Creator | YouTube URL | Paired job | Duration |",
        "|---|---------|------------|------------|----------|",
    ]
    for i, p in enumerate(processed, start=1):
        url = f"https://www.youtube.com/watch?v={p['youtube_id']}"
        lines.append(
            f"| {i} | {p['creator']} | [{p['youtube_id']}]({url}) "
            f"| {p['job_title']} | {p['duration_s']}s |"
        )
    lines.append("")
    lines.append(
        "These videos are used for academic capstone-demo purposes only. "
        "If you are one of the creators and would like your work removed from "
        "this demo, please file an issue on the project repository."
    )
    CREDITS_PATH.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Re-process all videos even if already cached")
    args = parser.parse_args()

    if CACHE_PATH.exists() and not args.force:
        cached = json.loads(CACHE_PATH.read_text())
        cached_ids = {row["youtube_id"] for row in cached}
        print(f"Existing cache covers {len(cached_ids)} videos.")
    else:
        cached = []
        cached_ids = set()

    bucket = _s3_bucket()
    print(f"Target bucket: s3://{bucket}/{S3_PREFIX}\n")

    processed: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="demo-videos-") as tmp:
        tmp_dir = Path(tmp)
        for i, src in enumerate(SOURCES, start=1):
            print(f"[{i}/{len(SOURCES)}] {src.youtube_id}  ({src.creator}, {src.duration_s}s)")
            existing = next((r for r in cached if r["youtube_id"] == src.youtube_id), None)
            if existing and not args.force:
                print(f"  → cached (video_score={existing['video_score']:.3f})")
                processed.append(existing)
                continue

            video_path = _yt_download(src.youtube_id, tmp_dir)
            s3_key = f"{S3_PREFIX}/{src.youtube_id}.mp4"
            _s3_upload(video_path, s3_key)

            result = _process_one(video_path, src)
            processed.append({
                "youtube_id": src.youtube_id,
                "creator": src.creator,
                "job_title": src.job_title,
                "duration_s": src.duration_s,
                "s3_key": s3_key,
                **result,
            })
            print()

    CACHE_PATH.write_text(json.dumps(processed, indent=2))
    write_credits(processed)
    print(f"\nWrote {len(processed)} videos → {CACHE_PATH.relative_to(Path.cwd())}")
    print(f"Wrote attribution → {CREDITS_PATH.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
