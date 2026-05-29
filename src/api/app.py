from dotenv import load_dotenv

load_dotenv()  # must be before any src.* import that reads os.getenv at module level

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from src.api.routes import parse, rank, video  # noqa: E402

app = FastAPI(
    title="Multimodal Resume Persona Screening with Explainable AI — ML Service",
    description=(
        "Stateless-ish ML compute service. "
        "Receives a DB record ID from the backend, downloads the file from S3/Cloudinary, "
        "processes it, and writes results back to the shared database."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

app.include_router(parse.router)
app.include_router(rank.router)
app.include_router(video.router)


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "ok", "service": "ml"}
