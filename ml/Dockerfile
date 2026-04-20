FROM python:3.11-slim

# ffmpeg for audio extraction, libsndfile for librosa
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 9000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "9000"]
