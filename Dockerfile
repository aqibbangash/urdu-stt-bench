# syntax=docker/dockerfile:1.7
#
# Urdu STT Benchmark — Linux x86_64 CPU-only, slim image.
# Single engine: faster-whisper (CTranslate2). No torch / transformers /
# librosa — faster-whisper ships its own audio decode + VAD.
#
# Build:   docker build -t urdu-stt-bench .
# Run:     docker run --rm -p 8501:8501 --cpus=4 --memory=8g \
#              -v urdu_stt_hf:/data/hf -e HF_TOKEN=hf_xxx urdu-stt-bench

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# System deps: ffmpeg (audio extraction) + curl/ca (HF over TLS, healthcheck).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-cpu.txt ./
RUN pip install --upgrade pip && pip install -r requirements-cpu.txt

# Copy source last so code-only edits don't bust the dependency layer.
COPY . .

# /data/hf is the HF cache mount point; chmod for non-root bind mounts.
RUN mkdir -p /data/hf && chmod 777 /data/hf

EXPOSE 8501

# CPU thread caps (override at runtime to match the cgroup CPU cap).
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    CT2_NUM_THREADS=4 \
    URDU_STT_CPU_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
