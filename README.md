# ML Service

This is where the machine-learning work happens. It reads resumes, pulls out
skills and experience, turns text into number vectors for matching, ranks
candidates against a job, and processes video resumes (transcribes the speech
and scores how the person communicates).

The backend calls this service in the background — you don't talk to it
directly. It runs as its own small API on port 9000.

Built with FastAPI, sentence-transformers, spaCy, Whisper, and librosa.

## What you need first

- Python 3.11
- `ffmpeg` (needed for video resumes) — install with `brew install ffmpeg`
- The same PostgreSQL database the backend uses (this service reads and writes
  to it)

## First-time setup

1. Create a virtual environment and install the packages. The first install
   downloads some large model files, so it can take a while:

   ```bash
   cd ml
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Download the spaCy language model (only needed once):

   ```bash
   python -m spacy download en_core_web_sm
   ```

3. Create your settings file:

   ```bash
   cp .env.example .env
   ```

   Make sure `DATABASE_URL` points to the same database the backend uses.

## Running it

```bash
cd ml
source venv/bin/activate
WHISPER_MODEL=base.en uvicorn src.api.app:app --host 0.0.0.0 --port 9000 --reload
```

The service is now at http://localhost:9000.

> **Heads up:** the first request is slow. The first time you parse a resume or
> rank candidates, it loads the language and speech models into memory, which
> takes a moment. After that it's fast.

## A note on the speech model

The `WHISPER_MODEL` setting controls how accurate (and how slow) the speech
transcription is:

- `base.en` or `medium` — good for a laptop without a graphics card
- `large-v3` — most accurate, but really only practical on a machine with a GPU

For local development on a Mac, `base.en` is the sensible choice.

## Running the tests

```bash
cd ml
source venv/bin/activate
pytest
```

## Settings

All settings live in the `.env` file. The important ones:

- `DATABASE_URL` — must point to the same database as the backend
- `SBERT_MODEL` — the text-matching model (default is fine)
- `WHISPER_MODEL` — the speech model (see note above)
- AWS or Cloudinary keys — so the service can download the uploaded files

See `.env.example` for the full list with comments.
