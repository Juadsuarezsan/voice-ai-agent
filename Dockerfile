# syntax=docker/dockerfile:1.7
#
# Two targets:
#   base  -> API with stub/hosted backends (Claude, Whisper API, ElevenLabs). ~300 MB.
#   audio -> base + openai-whisper, silero-vad and coqui XTTS (CPU torch). Several GB.
#
#   docker build --target base  -t voice-agent:base  .
#   docker build --target audio -t voice-agent:audio .

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY eval ./eval
RUN pip install --upgrade pip && pip install .

COPY data ./data
COPY demo ./demo

RUN useradd --create-home --uid 1000 voice && chown -R voice:voice /app
USER voice

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
FROM base AS audio
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
# CPU-only torch keeps the image far smaller than the CUDA wheels.
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install ".[audio]"
USER voice
ENV WHISPER_BACKEND=local WHISPER_MODEL=base VAD_BACKEND=silero
