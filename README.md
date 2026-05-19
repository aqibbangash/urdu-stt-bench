# Urdu STT Benchmark

A standalone micro-app for finding the best Urdu speech-to-text model for offline use on Apple Silicon Macs. Drag a video in, tick which models to run, see transcripts side-by-side with latency, RAM, and real-time-factor metrics.

Models are **swappable** — adding a new one is a YAML entry plus (optionally) one Python adapter file.

This tool is intentionally separate from the broadcast-monitoring app. It is a research / decision-support utility.

---

## Quick start

### 1. Prerequisites

- Apple Silicon Mac (M1 / M2 / M3 / M4)
- Python 3.10 or 3.11 (3.12 also works for most engines)
- `ffmpeg` on PATH — `brew install ffmpeg`
- ~20 GB free disk if you want to compare large models

### 2. Install

```bash
cd tools/urdu-stt-bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not need all engines, comment out their dependencies in `requirements.txt` before installing. Each engine fails gracefully at runtime if its deps are missing — the model just shows as "unavailable" in the UI.

`uv` is much faster than pip for these heavy ML deps if you have it:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 3. Run

```bash
streamlit run app.py
```

A browser tab opens. Upload your video (mp4 / mov / mkv / .ts / wav / m4a / mp3), tick the models you want to compare on the left, click **Run benchmark**. Results stream in as each model finishes.

First run downloads the model weights from Hugging Face — large-v3 is ~3 GB, large-v3-turbo is ~1.6 GB, medium is ~1.5 GB. They land in `~/.cache/huggingface/`. Subsequent runs are instant.

---

## Running in Docker (Linux x86_64, pure CPU)

This is the path for the dev-server — no Apple Silicon optimizations, deterministic CPU cap.

### One-time setup on the server

```bash
git clone <repo> && cd report-tool/tools/urdu-stt-bench
echo "HF_TOKEN=hf_xxx" > .env          # optional, raises HF rate limits
docker compose build
```

### Run with curated CPU/memory

```bash
# Defaults: 4 CPUs, 8 GiB RAM, port 8501.
docker compose up -d

# Pin tighter:
URDU_STT_CPUS=2 URDU_STT_MEM=4g URDU_STT_PORT=8501 docker compose up -d

# Wider:
URDU_STT_CPUS=8 URDU_STT_MEM=16g docker compose up -d
```

The compose file:
- Enforces the CPU cap at the cgroup level (`deploy.resources.limits.cpus`).
- Propagates the same number into `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `CT2_NUM_THREADS`, and `URDU_STT_CPU_THREADS` so every threading layer (faster-whisper / ctranslate2 / torch / numpy) respects the cap — not just cgroup throttling.
- Persists Hugging Face weights in the `urdu_stt_hf_cache` named volume so restarts don't re-download multi-GB models.

### Useful commands

```bash
docker compose logs -f urdu-stt           # tail logs
docker compose ps                          # status + healthcheck
docker compose down                        # stop
docker compose down -v                     # stop AND wipe model cache
docker stats urdu-stt                      # live CPU / RAM utilization
docker exec -it urdu-stt /bin/bash         # shell into container
```

Open `http://<dev-server>:8501` in a browser (or `ssh -L 8501:localhost:8501 dev-server` and hit localhost).

### What's different from the Mac path

The Docker image installs `requirements-cpu.txt`, which drops `lightning-whisper-mlx` and `mlx` (Apple Silicon only) and pulls a CPU-only PyTorch wheel. Whisper variants run via faster-whisper / ctranslate2 on CPU. MLX entries in `models.yaml` will show as "unavailable" inside the container — expected.

---

## Adding a new model

There are two paths:

### Path A — same engine, new weights

If the new model uses an engine the app already supports (Whisper variants, generic Hugging Face ASR pipelines, MMS adapters, whisper.cpp `.bin` files), just add a YAML entry. No Python required.

Open `models.yaml` and copy an existing entry:

```yaml
- name: my-new-whisper
  description: My fine-tuned whisper for Pakistani Urdu
  engine: faster-whisper
  params:
    model_size: my-org/my-finetuned-whisper   # any HF repo id
    compute_type: int8
```

Reload Streamlit (press `R`) — the new model appears.

### Path B — new engine

If the model needs a different runtime (a new framework, a custom decoder), add an adapter under `stt/engines/`:

1. Subclass `STTEngine` (see `stt/base.py`).
2. Implement `load()` and `transcribe()`.
3. Register the class in `stt/registry.py`'s `ENGINE_TYPES` dict.
4. Reference it from `models.yaml` with the new `engine:` value.

The adapter file should be small — 30–60 lines. Look at `stt/engines/faster_whisper_engine.py` as a template.

### Downloading from GitHub releases (e.g. whisper.cpp ggml files)

The `whisper-cpp` engine takes a local file path. Download the `.bin` manually:

```bash
mkdir -p models/whisper-cpp
curl -L -o models/whisper-cpp/ggml-large-v3.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin
```

Then point `models.yaml` at it:

```yaml
- name: whisper.cpp large-v3 (Core ML)
  engine: whisper-cpp
  params:
    model_file: models/whisper-cpp/ggml-large-v3.bin
```

---

## What gets reported per model

| Metric | Meaning |
|---|---|
| **RTF** | Real-time factor — `wall_seconds / audio_seconds`. 0.5x = twice as fast as real-time. Below 1.0x means it can run live. |
| **Wall (s)** | End-to-end transcription time. Includes model load on first run, excludes audio decode. |
| **Peak RAM (MB)** | Maximum resident set size of the Python process during transcription. |
| **Avg CPU %** | Average CPU utilization. On Apple Silicon, MLX-accelerated models will look "low CPU" because work is on the Neural Engine / GPU. |
| **Detected lang** | What the model thinks the audio language is. Should say `ur`. |
| **Words** | Word count of the transcript. Useful as a sanity check — a model returning 5 words for a 3-min clip likely failed. |
| **Transcript** | Full text output, scrollable, downloadable. |

The metrics table sorts naturally so you can spot Pareto-dominant models (lowest RTF AND lowest RAM AND most words).

---

## Default model list

The `models.yaml` shipped ready-to-go includes:

1. **Whisper Large-v3 (MLX)** — best quality reference. Apple Silicon-accelerated.
2. **Whisper Large-v3-Turbo (MLX, 4-bit)** — fastest accurate model. Strong recommendation for production.
3. **Whisper Medium (MLX)** — half the RAM of large. Good for testing pipelines.
4. **Whisper Large-v3 (faster-whisper, int8)** — CT2 quantized fallback. Runs without MLX.
5. **Meta MMS-1B-all (Urdu adapter)** — Urdu-specific 1B-param model. Different architecture from Whisper.
6. **kingabzpro/wav2vec2-large-xls-r-300m-Urdu** — community fine-tune. Small and quick.

Disable any of these by unchecking in the sidebar; add more as described above.

---

## Tips for fair comparison

- Use the **same audio clip** across runs (the app guarantees this — audio is extracted once).
- For first-pass evaluation, pick a clip with mixed content: anchor speech (clean), guest speech (noisier), and maybe an ad break.
- Whisper models do auto language-detect. If a model returns English when the audio is Urdu, it failed.
- "Best results" is subjective for Urdu — script choice (Nastaliq vs Naskh ligatures), proper-noun handling, and code-switching (Urdu/English mid-sentence) all matter. The side-by-side view makes these differences obvious.
- Run twice and ignore the first run's wall time — the first transcribe includes model load + Metal kernel JIT.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'lightning_whisper_mlx'` | `pip install lightning-whisper-mlx` |
| `ffmpeg: command not found` | `brew install ffmpeg` |
| Streamlit "address in use" | `streamlit run app.py --server.port 8502` |
| Out-of-memory on Whisper Large | Switch to Large-v3-Turbo or Medium, or use `quant: 4bit` |
| Engine downloads timeout | Pre-download with `huggingface-cli download <repo_id>` |
| Transcript is empty | Check the audio extraction step — try uploading a `.wav` directly to bypass video decode |

---

## Layout

```
tools/urdu-stt-bench/
├── README.md
├── requirements.txt
├── models.yaml              # the swappable registry
├── app.py                   # Streamlit entry
└── stt/
    ├── base.py              # STTEngine ABC + dataclasses
    ├── registry.py          # YAML → engine instantiation
    ├── audio.py             # ffmpeg wrapper
    ├── metrics.py           # resource tracker
    └── engines/
        ├── faster_whisper_engine.py
        ├── mlx_whisper_engine.py
        ├── transformers_engine.py
        ├── mms_engine.py
        └── whisper_cpp_engine.py
```
