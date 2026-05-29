"""
Minimal ORM models for the ML service.
Only the tables that ML needs to read from or write to.
These must stay in sync with the backend's migration definitions.
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(255))
    description: Mapped[str] = mapped_column(Text)
    required_skills: Mapped[list[str]] = mapped_column(ARRAY(Text))
    preferred_skills: Mapped[list[str]] = mapped_column(ARRAY(Text))
    min_experience_years: Mapped[int] = mapped_column(Integer)
    education_requirement: Mapped[str | None] = mapped_column(String(100))


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    full_name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    phone: Mapped[str | None] = mapped_column(String(50))
    skills: Mapped[list[str]] = mapped_column(ARRAY(Text))
    years_experience: Mapped[float] = mapped_column(Float)
    education_level: Mapped[str | None] = mapped_column(String(100))
    raw_text: Mapped[str | None] = mapped_column(Text)
    skill_sources: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(768))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Resume(Base):
    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("candidates.id"))
    job_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("jobs.id"))
    s3_key: Mapped[str] = mapped_column(String(512))
    file_type: Mapped[str] = mapped_column(String(20))
    parse_status: Mapped[str] = mapped_column(String(50))


class VideoResume(Base):
    __tablename__ = "video_resumes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("candidates.id"))
    job_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("jobs.id"))
    s3_key: Mapped[str] = mapped_column(String(512))
    transcript: Mapped[str | None] = mapped_column(Text)
    transcript_confidence: Mapped[float | None] = mapped_column(Float)
    audio_features: Mapped[dict | None] = mapped_column(JSON)
    video_score: Mapped[float | None] = mapped_column(Float)
    process_status: Mapped[str] = mapped_column(String(50))

    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(Integer, ForeignKey("jobs.id"))
    candidate_id: Mapped[int] = mapped_column(Integer, ForeignKey("candidates.id"))
    composite_score: Mapped[float] = mapped_column(Float)
    semantic_score: Mapped[float | None] = mapped_column(Float)
    skill_overlap_score: Mapped[float | None] = mapped_column(Float)
    experience_score: Mapped[float | None] = mapped_column(Float)
    education_score: Mapped[float | None] = mapped_column(Float)
    video_score: Mapped[float] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("job_id", "candidate_id", name="uq_screening_job_candidate"),)


class ExplainabilityData(Base):
    __tablename__ = "explainability_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    screening_result_id: Mapped[int] = mapped_column(Integer, ForeignKey("screening_results.id"), unique=True)
    shap_values: Mapped[dict] = mapped_column(JSON)
    baseline_score: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
