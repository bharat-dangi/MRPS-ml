import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.db import get_db
from src.matching.embedder import ResumeEmbedder
from src.models import Candidate, Resume
from src.ner.extractor import extract_candidate_profile
from src.parsers.docx_parser import extract_text_from_docx
from src.parsers.pdf_parser import extract_text_from_pdf
from src.storage import download_to_temp

router = APIRouter(prefix="/parse", tags=["parse"])


class ParseRequest(BaseModel):
    resume_id: int  # DB primary key — ML looks up s3_key from this


class ParseResponse(BaseModel):
    status: str
    candidate_id: int
    skills_found: int
    years_experience: float


@router.post("/resume", response_model=ParseResponse)
def parse_resume(body: ParseRequest, db: Session = Depends(get_db)) -> ParseResponse:
    """
    1. Look up resume record by ID → get s3_key + candidate_id + file_type
    2. Download file from S3 (or Cloudinary)
    3. Parse text → run NER → generate SBERT embedding
    4. Update candidate row + mark resume as done
    """
    resume = db.scalar(select(Resume).where(Resume.id == body.resume_id))
    if not resume:
        raise HTTPException(status_code=404, detail=f"Resume {body.resume_id} not found")

    ext = f".{resume.file_type}"
    tmp_path = download_to_temp(resume.s3_key, suffix=ext)

    try:
        raw_text = (
            extract_text_from_pdf(tmp_path)
            if resume.file_type == "pdf"
            else extract_text_from_docx(tmp_path)
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    profile = extract_candidate_profile(raw_text)
    embedder = ResumeEmbedder()
    embedding = embedder.encode(raw_text).tolist()

    # Write extracted data back to the shared DB
    candidate = db.scalar(select(Candidate).where(Candidate.id == resume.candidate_id))
    if candidate:
        # Only use the NER-extracted name if the user didn't provide one
        if candidate.full_name in ("Unknown", "", None):
            candidate.full_name = profile.full_name or candidate.full_name
        candidate.email = profile.email or candidate.email
        candidate.phone = profile.phone or candidate.phone
        candidate.skills = profile.skills
        candidate.years_experience = profile.years_experience
        candidate.education_level = profile.education_level
        candidate.raw_text = raw_text
        candidate.embedding = embedding

    resume.parse_status = "done"
    db.commit()

    return ParseResponse(
        status="done",
        candidate_id=resume.candidate_id,
        skills_found=len(profile.skills),
        years_experience=profile.years_experience,
    )
