# syntax=docker/dockerfile:1.7
#
# Urdu STT Benchmark — Linux x86_64 CPU-only image.
#
# Build:   docker build -t urdu-stt-bench .
# Run:     docker run --rm -p 8501:8501 --cpus=4 --memory=8g \
#              -v urdu_stt_hf:/data/hf -e HF_TOKEN=hf_xxx urdu-stt-bench
# Compose: docker compose up -d

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# System deps:
#   ffmpeg     — audio extraction
#   curl/ca    — model downloads over TLS
#   build-essential + cargo — fallback for sdists without wheels (e.g. tiktoken
#                              if a transitive resolver picks an old version)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        curl \
        ca-certificates \
        build-essential \
        cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (the default torch wheel ships with CUDA and
# is multi-GB; the CPU index gives a ~200 MB wheel).
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.* torchaudio==2.4.*

# Install the rest of the deps with the public PyPI index (extra-index makes
# torchaudio's pin resolve cleanly against torch we already installed).
COPY requirements-cpu.txt ./
RUN pip install -r requirements-cpu.txt \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source last so code-only edits don't bust the dependency layer.
COPY . .

# /data/hf is the HF cache mount point; create with permissive perms so a
# host-uid bind mount doesn't trip on root-owned dirs.
RUN mkdir -p /data/hf && chmod 777 /data/hf

EXPOSE 8501

# CPU thread caps. These can be overridden at runtime — docker-compose.yml
# wires them to URDU_STT_CPUS by default so they match --cpus.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
