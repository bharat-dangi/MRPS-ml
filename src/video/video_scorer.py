import numpy as np

from src.video.audio_features import detect_filler_words

VIDEO_WEIGHTS: dict[str, float] = {
    "transcript_sim": 0.50,
    "comm_clarity": 0.30,
    "content_coverage": 0.20,
}

_PACE_MIN_WPM = 120
_PACE_MAX_WPM = 160
_PACE_FLOOR_WPM = 60
_PACE_CEIL_WPM = 220


def _pace_score(wpm: float) -> float:
    if _PACE_MIN_WPM <= wpm <= _PACE_MAX_WPM:
        return 1.0
    if wpm < _PACE_MIN_WPM:
        return max(0.0, (wpm - _PACE_FLOOR_WPM) / (_PACE_MIN_WPM - _PACE_FLOOR_WPM))
    return max(0.0, (_PACE_CEIL_WPM - wpm) / (_PACE_CEIL_WPM - _PACE_MAX_WPM))


def _comm_clarity_breakdown(audio_features: dict, transcript: str, word_timestamps: list[dict] | None) -> dict:
    """
    Compute comm_clarity = 0.40×pace_score + 0.40×fluency_score + 0.20×articulation_score.
    Returns a dict with all sub-scores plus filler_rate.
    """
    duration = audio_features.get("duration_seconds", 60.0)
    word_count = len(transcript.split()) if transcript.strip() else 0
    wpm = (word_count / max(duration / 60.0, 1e-6))

    pace = _pace_score(wpm)

    filler_info = detect_filler_words(transcript, duration)
    filler_rate = filler_info["filler_rate"]
    fluency = 1.0 - min(filler_rate / 5.0, 1.0)

    if word_timestamps:
        confidences = [w.get("probability", 1.0) for w in word_timestamps if "probability" in w]
        articulation = float(np.mean(confidences)) if confidences else 0.5
    else:
        articulation = 0.5

    clarity = float(np.clip(0.40 * pace + 0.40 * fluency + 0.20 * articulation, 0.0, 1.0))
    return {
        "comm_clarity": clarity,
        "pace_score": round(pace, 4),
        "fluency_score": round(fluency, 4),
        "articulation_score": round(articulation, 4),
        "filler_rate": round(filler_rate, 3),
        "filler_words": filler_info["filler_words"],
        "wpm": round(wpm, 1),
    }


def compute_video_score(
    transcript: str,
    jd_text: str,
    audio_features: dict,
    jd_skills: list[str],
    word_timestamps: list[dict] | None = None,
) -> dict:
    """
    Compute the video score from transcript + audio features.
    Returns full score breakdown dict including comm_clarity sub-scores.
    """
    from src.matching.embedder import ResumeEmbedder  # deferred — heavy model load

    embedder = ResumeEmbedder()

    if transcript.strip():
        t_emb = embedder.encode(transcript)
        jd_emb = embedder.encode(jd_text)
        transcript_sim = float(max(0.0, np.dot(t_emb, jd_emb)))
    else:
        transcript_sim = 0.0

    lower_transcript = transcript.lower()
    if jd_skills:
        mentioned = sum(1 for skill in jd_skills if skill.lower() in lower_transcript)
        content_coverage = mentioned / len(jd_skills)
    else:
        content_coverage = transcript_sim

    clarity_info = _comm_clarity_breakdown(audio_features, transcript, word_timestamps)
    comm_clarity = clarity_info["comm_clarity"]

    video_score = (
        VIDEO_WEIGHTS["transcript_sim"] * transcript_sim
        + VIDEO_WEIGHTS["comm_clarity"] * comm_clarity
        + VIDEO_WEIGHTS["content_coverage"] * content_coverage
    )

    # Collect skills mentioned only in the transcript (source = "video")
    video_skill_sources: dict[str, str] = {
        skill: "video"
        for skill in jd_skills
        if skill.lower() in lower_transcript
    }

    return {
        "video_score": float(np.clip(video_score, 0.0, 1.0)),
        "transcript_sim": transcript_sim,
        "comm_clarity": comm_clarity,
        "content_coverage": content_coverage,
        **clarity_info,
        "skill_sources": video_skill_sources,
    }
