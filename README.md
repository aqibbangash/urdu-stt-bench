# Urdu STT Benchmark

A standalone micro-app for benchmarking offline Urdu speech-to-text on **pure CPU**, using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Drop a video or audio file in, pick any Hugging Face model from the sidebar, and see load time, inference time, RAM, RTF, and the transcript — all from a Streamlit UI.

Designed as a research / decision-support tool — separate from any production pipeline.

---

## Two ways to run

### A. Docker (recommended)

The published image is `ghcr.io/aqibbangash/urdu-stt-bench:latest` (linux/amd64).

```bash
docker run --rm -p 8501:8501 \
  --cpus=4 --memory=8g \
  -v urdu_stt_hf:/data/hf \
  -e HF_TOKEN=hf_xxx \
  ghcr.io/aqibbangash/urdu-stt-bench:latest
```

Open `http://localhost:8501`. Models download to the `urdu_stt_hf` volume on first use and persist across restarts.

### B. docker compose (build locally)

```bash
git clone https://github.com/aqibbangash/urdu-stt-bench
cd urdu-stt-bench
echo "HF_TOKEN=hf_xxx" > .env
docker compose up -d
```

Tighten or relax CPU / memory caps:
```bash
URDU_STT_CPUS=2 URDU_STT_MEM=4g docker compose up -d
URDU_STT_CPUS=8 URDU_STT_MEM=16g docker compose up -d
```

### C. Portainer (Docker Swarm)

Use the included `portainer-stack.yml`. In Portainer → Stacks → Add stack → Web editor, paste the file, set `HF_TOKEN` in the environment variables section, and deploy.

| Variable | Required | Default | Effect |
|---|---|---|---|
| `HF_TOKEN` | yes | — | Lifts Hugging Face download rate limits |
| `URDU_STT_CPUS` | no | `4` | cgroup CPU cap + per-library thread caps |
| `URDU_STT_MEM` | no | `8g` | Container memory limit |
| `URDU_STT_PORT` | no | `8501` | Published host port |

Setting only `URDU_STT_CPUS` automatically fills in `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `CT2_NUM_THREADS`, and `URDU_STT_CPU_THREADS` via compose interpolation.

### D. Local Python (without Docker)

```bash
git clone https://github.com/aqibbangash/urdu-stt-bench
cd urdu-stt-bench
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "HF_TOKEN=hf_xxx" > .env       # optional
streamlit run app.py
```

Requires Python ≥ 3.10, `ffmpeg` on PATH, and Linux/macOS. Windows works in theory but isn't tested.

---

## Using the UI

1. **Upload** a video or audio file (mp4 / mov / mkv / ts / wav / m4a / mp3 / flac / ogg).
2. **Pick a model** in the sidebar — either from the quick-pick dropdown, or type any HF repo id (e.g. `large-v3`, `tiny`, `deepdml/faster-whisper-large-v3-turbo-ct2`, `Systran/faster-distil-whisper-large-v3`).
3. Choose a **compute type** (`int8` is the default; smallest + fastest on CPU).
4. Click **Load model** — weights pull from HF, you see live elapsed time, current RAM, and the library log lines.
5. Click **Run inference** — transcription streams in segment-by-segment, with live RSS / CPU / log.
6. The result card shows load time, inference time, RTF (real-time factor), peak RAM, average CPU, transcript download, and per-segment timing.
7. Every run accumulates in the **Run history** table at the bottom for cross-model comparison.

The model picker is fully runtime — there's no hardcoded list. Anything that resolves on Hugging Face as a CTranslate2 Whisper model works.

### Suggested models for Urdu

Start here; move up the ladder if accuracy needs improvement:

| Model | RAM | Speed | Quality |
|---|---|---|---|
| `tiny` | ~75 MB | very fast | smoke-test only |
| `base` | ~140 MB | fast | rough |
| `small` ✅ | ~480 MB | fast | good — sensible default |
| `medium` | ~1.5 GB | moderate | better |
| `large-v3` | ~3 GB | slow | best vanilla |
| `deepdml/faster-whisper-large-v3-turbo-ct2` ⭐ | ~1.6 GB | ~4× faster than large-v3 | quality very close to large-v3 — strongest CPU pick |

---

## Metrics reported

| Metric | Meaning |
|---|---|
| **Load (s)** | Wall time from `WhisperModel(...)` instantiation to weights loaded |
| **Inference (s)** | End-to-end transcribe time |
| **RTF** | `inference_seconds / audio_seconds`. `< 1.0×` = faster than real-time |
| **Peak RSS** | Maximum resident-set-size during transcription |
| **Avg CPU %** | Average CPU utilization (psutil) |
| **Words** | Word count of the transcript |
| **Detected lang** | Language returned by Whisper (should match what you set) |

First run on a cold model includes weight download + ctranslate2 kernel init in the wall time. Run twice and trust the second number.

---

## Persistent model cache

Inside the container, `HF_HOME=/data/hf`. Mount a named volume there (the included compose files do this automatically) and downloaded weights survive container restarts / image updates / stack redeploys.

To force a fresh download (e.g. switching to a different ctranslate2 variant of the same model), remove the volume:
```bash
docker volume rm urdu_stt_hf_cache
```

---

## CPU control

Both layers are wired up:
1. **cgroup limit** — `--cpus=N` / `deploy.resources.limits.cpus: N`
2. **Per-library thread caps** — `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `CT2_NUM_THREADS`, `URDU_STT_CPU_THREADS` all default to `URDU_STT_CPUS`. faster-whisper's `WhisperModel` reads `URDU_STT_CPU_THREADS` and passes it as `cpu_threads`, so ctranslate2 spawns exactly that many worker threads.

Without the second layer, BLAS / OpenMP would spawn one thread per logical core and get throttled at the cgroup boundary — slower than honoring the cap from the start.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ffmpeg: command not found` (local Python path) | Install ffmpeg — `apt install ffmpeg` or `brew install ffmpeg` |
| Streamlit "address in use" | `streamlit run app.py --server.port 8502` |
| Container OOM-killed during load | Increase `URDU_STT_MEM` (large-v3 needs ~6g, medium ~4g, small ~2g) |
| Model download hangs / 401 | Set `HF_TOKEN` in `.env` or Portainer env vars |
| Transcript is empty | Confirm audio extraction worked — try a `.wav` directly |
| RTF much worse than expected | First run includes load; run a second time. Also check `URDU_STT_CPUS` matches the cgroup cap |

---

## Architecture

```
urdu-stt-bench/
├── README.md
├── Dockerfile               # linux/amd64 CPU build
├── requirements-cpu.txt     # the deps installed in the image
├── requirements.txt         # same deps for local Python
├── docker-compose.yml       # local build + run
├── portainer-stack.yml      # for Portainer swarm deployment
├── .github/workflows/
│   └── docker.yml           # builds + pushes ghcr.io/<user>/urdu-stt-bench
├── app.py                   # Streamlit entry — model picker + live UI
└── stt/
    ├── base.py              # STTEngine ABC + Segment / Transcription dataclasses
    ├── registry.py          # engine resolver (only faster-whisper now)
    ├── audio.py             # ffmpeg wrapper
    ├── metrics.py           # ResourceTracker (wall / peak RSS / CPU)
    └── engines/
        └── faster_whisper_engine.py
```

Adding more engines later (e.g. a CUDA path) means dropping a new adapter under `stt/engines/`, registering it in `stt/registry.py`, and adding the matching imports — no UI changes required.
