"""Locks the key-name contract between compute_video_score's return dict and
the keys read by the /video/process FastAPI response builder.

A previous bug had the route reading `transcript_similarity` while the scorer
returned `transcript_sim`, causing every video task to 500.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.video.video_scorer import compute_video_score


def _stub_embedder() -> MagicMock:
    """ResumeEmbedder stub — every .encode() returns the same unit vector,
    which makes cosine similarity deterministic (1.0 when both inputs match)."""
    embedder = MagicMock()
    embedder.encode.return_value = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return embedder


def test_compute_video_score_returns_keys_consumed_by_api() -> None:
    """The keys named below are read directly by ml/src/api/routes/video.py.
    If you rename any of them in the scorer, also update the route."""
    audio_features = {
        "wpm": 130.0,
        "filler_rate_per_min": 2.0,
        "long_pause_count": 1,
        "voiced_ratio": 0.85,
    }
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()):
        result = compute_video_score(
            transcript="hello world",
            jd_text="hello world",
            audio_features=audio_features,
            jd_skills=["python"],
        )

    # These exact keys are consumed by VideoResponse / audio_features blob.
    required_keys = {
        "video_score",
        "transcript_sim",
        "comm_clarity",
        "content_coverage",
        "skill_sources",
    }
    missing = required_keys - set(result)
    assert not missing, f"compute_video_score is missing keys: {missing}"

    # Values must be in the expected unit interval where applicable.
    assert 0.0 <= result["video_score"] <= 1.0
    assert 0.0 <= result["transcript_sim"] <= 1.0
    assert 0.0 <= result["comm_clarity"] <= 1.0
    assert 0.0 <= result["content_coverage"] <= 1.0


def _bare_candidate(**overrides):
    """SimpleNamespace stand-in for a Candidate ORM row. Used so we don't
    have to spin up a DB session just to exercise _backfill_*."""
    from types import SimpleNamespace
    defaults = {
        "full_name": "Test",
        "email": None,
        "phone": None,
        "raw_text": None,
        "embedding": None,
        "skills": [],
        "skill_sources": None,
        "years_experience": 0,
        "education_level": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _stub_profile(**overrides):
    """Minimal CandidateProfile stand-in."""
    from types import SimpleNamespace
    defaults = {
        "full_name": "Unknown",
        "email": None,
        "phone": None,
        "skills": [],
        "skill_sources": {},
        "years_experience": 0.0,
        "education_level": None,
        "raw_text": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_backfill_candidate_from_video_populates_empty_raw_text() -> None:
    """Video-only candidates start with raw_text=None — process_video should
    hydrate it from the transcript so the ranker has text to score against."""
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate()
    profile = _stub_profile(
        full_name="Henry Davis",
        skills=["python", "fastapi"],
    )
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(
            candidate,
            transcript_text="hello world",
            jd_skill_sources={"python": "video"},
        )
    assert candidate.raw_text == "hello world"
    assert candidate.embedding is not None
    assert set(candidate.skills) == {"python", "fastapi"}
    assert candidate.skill_sources == {"python": "video", "fastapi": "video"}


def test_backfill_candidate_from_video_preserves_existing_raw_text() -> None:
    """If the candidate already had a parsed resume, the video shouldn't
    overwrite their raw_text — both modalities coexist."""
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate(
        full_name="Alice Real",
        raw_text="parsed from PDF — Senior Engineer at Acme",
        embedding=np.array([0.5, 0.5, 0.0], dtype=np.float32),
        skills=["python"],
        skill_sources={"python": "text"},
    )
    profile = _stub_profile(full_name="Hello From Video", skills=["fastapi", "python"])
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(
            candidate, transcript_text="hello from video",
            jd_skill_sources={"fastapi": "video"},
        )
    # raw_text and embedding unchanged.
    assert candidate.raw_text == "parsed from PDF — Senior Engineer at Acme"
    assert candidate.embedding[0] == pytest.approx(0.5)
    # Real-looking existing full_name preserved.
    assert candidate.full_name == "Alice Real"
    # Skills merged (python deduped, fastapi added).
    assert "fastapi" in candidate.skills
    assert candidate.skills.count("python") == 1


def test_backfill_overwrites_placeholder_name_with_ner_person() -> None:
    """If the candidate row was created with a placeholder name ("Test"),
    the PERSON entity from the transcript wins."""
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate(full_name="Test")
    profile = _stub_profile(full_name="Henry Davis")
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(candidate, "transcript text", {})
    assert candidate.full_name == "Henry Davis"


def test_backfill_does_not_overwrite_real_name_with_ner() -> None:
    """If the candidate name was already real-looking, NER doesn't clobber it."""
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate(full_name="Alice Realname")
    profile = _stub_profile(full_name="Random Misread")
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(candidate, "transcript text", {})
    assert candidate.full_name == "Alice Realname"


def test_backfill_keeps_existing_email_when_present() -> None:
    """Existing email (from upload form / resume) is preserved over transcript NER."""
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate(email="alice@real.com")
    profile = _stub_profile(email="something.else@example.com")
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(candidate, "txt", {})
    assert candidate.email == "alice@real.com"


def test_backfill_fills_email_from_transcript_when_missing() -> None:
    from src.api.routes.video import _backfill_candidate_from_video

    candidate = _bare_candidate(email=None)
    profile = _stub_profile(email="henry@example.com")
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()), \
         patch("src.ner.extractor.extract_candidate_profile", return_value=profile):
        _backfill_candidate_from_video(candidate, "transcript", {})
    assert candidate.email == "henry@example.com"


def test_is_placeholder_name_recognises_common_placeholders() -> None:
    from src.api.routes.video import _is_placeholder_name
    for placeholder in ("", "  ", "Test", "test", "TEST", "Unknown",
                        "demo", "Candidate", "n/a", "-", None):
        assert _is_placeholder_name(placeholder), f"{placeholder!r} should be placeholder"


def test_is_placeholder_name_accepts_real_names() -> None:
    from src.api.routes.video import _is_placeholder_name
    for real in ("Henry Davis", "Alice Smith", "Bharat Dangi", "Liam Okafor"):
        assert not _is_placeholder_name(real), f"{real!r} shouldn't be placeholder"


def test_compute_video_score_does_not_crash_with_empty_jd_text() -> None:
    """Empty JD text shouldn't raise — used when a video has no associated job."""
    with patch("src.matching.embedder.ResumeEmbedder", return_value=_stub_embedder()):
        result = compute_video_score(
            transcript="hello",
            jd_text="",
            audio_features={"wpm": 130, "filler_rate_per_min": 1.0,
                            "long_pause_count": 0, "voiced_ratio": 0.9},
            jd_skills=[],
        )
    # The function must still return all required keys.
    assert {"video_score", "transcript_sim", "comm_clarity", "content_coverage"} <= set(result)
