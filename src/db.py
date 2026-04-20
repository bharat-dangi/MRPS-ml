"""
ML service's own database connection.
Shares the same PostgreSQL as the backend — connected via DATABASE_URL env var.
ML only reads what it needs and writes results (embeddings, scores, rankings).
"""
import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://resume_user:resume_pass@localhost:5432/resume_db",
)

# Sync engine — all ML processing is CPU-bound so async adds no value here
engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session
