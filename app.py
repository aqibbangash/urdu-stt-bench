"""Streamlit entry point for the Urdu STT Benchmark — single-model workflow.

Flow:
  1. Upload audio/video.
  2. Pick one model.
  3. Click "Load model" — weights pull into RAM. Live stats + log pane show
     elapsed time, current RSS, CPU, and library log lines.
  4. Click "Run inference" — transcribe runs in a background thread. Live
     stats + log pane update. For faster-whisper, segments stream in as they
     decode.
  5. Final card shows load time, inference time, RTF, peak RAM during
     inference, full transcript, and per-segment timing.

Pure CPU is the design target (see models.yaml). The transformers / MMS
engines accept `device: cpu` in their YAML params.
"""
from __future__ import annotations

import gc
import logging
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import psutil
import streamlit as st
from dotenv import load_dotenv

# Load .env (HF_TOKEN, etc.) before any huggingface_hub / transformers import path runs.
load_dotenv(Path(__file__).parent / ".env")

from stt.audio import extract_audio, probe_duration
from stt.base import Segment, Transcription
from stt.metrics import ResourceTracker
from stt.registry import availability_status, instantiate, load_registry

# Quieter HF hub progress bars in stderr — we surface logger events instead.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

st.set_page_config(
    page_title="Urdu STT Benchmark",
    page_icon="🗣",
    layout="wide",
    initial_sidebar_state="expanded",
)

REGISTRY_PATH = Path(__file__).parent / "models.yaml"


# ── Logging plumbing ────────────────────────────────────────────────────
class QueueLogHandler(logging.Handler):
    """Push every log record into a thread-safe queue for the UI to drain."""

    def __init__(self, q: "queue.Queue[str]") -> None:
        super().__init__()
        self.q = q
        self.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001
            pass


def _install_log_handler(q: "queue.Queue[str]") -> QueueLogHandler:
    handler = QueueLogHandler(q)
    handler.setLevel(logging.INFO)
    # Capture from libs that actually log meaningful events.
    for name in ("huggingface_hub", "faster_whisper", "transformers", "ctranslate2"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    return handler


def _remove_log_handler(handler: QueueLogHandler) -> None:
    for name in ("huggingface_hub", "faster_whisper", "transformers", "ctranslate2"):
        logging.getLogger(name).removeHandler(handler)


# ── Session state ───────────────────────────────────────────────────────
@dataclass
class LoadResult:
    seconds: float
    rss_before_mb: float
    rss_after_mb: float
    peak_rss_during_mb: float

    @property
    def model_rss_mb(self) -> float:
        return max(0.0, self.rss_after_mb - self.rss_before_mb)


@dataclass
class InferResult:
    seconds: float
    audio_seconds: float
    peak_rss_mb: float
    avg_cpu: float
    transcription: Optional[Transcription] = None
    error: Optional[str] = None
    segments_stream: list[Segment] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.seconds / self.audio_seconds if self.audio_seconds else 0.0


def _ss(key: str, default: Any = None) -> Any:
    return st.session_state.setdefault(key, default)


_ss("engine", None)
_ss("engine_label", None)
_ss("load_result", None)
_ss("infer_result", None)
_ss("audio_path", None)
_ss("audio_seconds", None)
_ss("media_basename", None)
_ss("history", [])


# ── Sidebar: model registry ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _cached_registry(mtime: float) -> dict[str, Any]:
    return load_registry(REGISTRY_PATH)


registry = _cached_registry(REGISTRY_PATH.stat().st_mtime)
models = registry.get("models", [])

st.sidebar.header("Model")
st.sidebar.caption(f"{len(models)} registered · edit models.yaml to add more")

available_models = []
for m in models:
    ok, hint = availability_status(m.get("engine", "?"))
    available_models.append((m, ok, hint))

selectable = [(m, ok, hint) for (m, ok, hint) in available_models if ok]
if not selectable:
    st.sidebar.error("No engines available. See requirements.txt.")
    st.stop()

labels = [m["name"] for (m, _, _) in selectable]
current_label = _ss("engine_label") or labels[0]
if current_label not in labels:
    current_label = labels[0]
chosen_label = st.sidebar.radio(
    "Pick one model",
    labels,
    index=labels.index(current_label),
    key="model_radio",
)
chosen = next(m for (m, _, _) in selectable if m["name"] == chosen_label)
st.sidebar.caption(chosen.get("description", ""))

# Surface unavailable models for visibility.
unavailable = [(m, hint) for (m, ok, hint) in available_models if not ok]
if unavailable:
    with st.sidebar.expander(f"Unavailable ({len(unavailable)})"):
        for m, hint in unavailable:
            st.markdown(f"**{m['name']}** — {hint}")

# If the user switches model, drop any loaded engine so RAM doesn't pile up.
if st.session_state["engine_label"] != chosen_label and st.session_state["engine"] is not None:
    try:
        st.session_state["engine"].unload()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["engine"] = None
    st.session_state["load_result"] = None
    st.session_state["infer_result"] = None
    gc.collect()
st.session_state["engine_label"] = chosen_label


# ── Header ─────────────────────────────────────────────────────────────
st.title("Urdu STT Benchmark")
st.caption(
    "Pure-CPU offline Urdu speech-to-text. Load a model, run inference, "
    "compare load time, inference time, RAM, and RTF across runs."
)

# Always-visible process stats strip.
strip = st.columns(4)
proc = psutil.Process(os.getpid())
strip[0].metric("Process RSS (MB)", f"{proc.memory_info().rss / 1024**2:.0f}")
strip[1].metric("System CPU %", f"{psutil.cpu_percent(interval=None):.0f}")
strip[2].metric("System RAM used %", f"{psutil.virtual_memory().percent:.0f}")
strip[3].metric("Cores", f"{psutil.cpu_count(logical=False)}P / {psutil.cpu_count(logical=True)}L")


# ── Upload ─────────────────────────────────────────────────────────────
st.subheader("1. Audio")
upload = st.file_uploader(
    "Upload a video or audio file",
    type=["mp4", "mov", "mkv", "ts", "wav", "m4a", "mp3", "flac", "ogg"],
    help="Audio is extracted to 16 kHz mono WAV once and reused across runs.",
)

if upload is not None and upload.name != st.session_state.get("media_basename"):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(upload.name).suffix
    ) as fh:
        fh.write(upload.read())
        media_path = Path(fh.name)
    with st.spinner("Extracting audio with ffmpeg…"):
        try:
            audio_path = extract_audio(media_path)
            audio_seconds = probe_duration(audio_path)
            st.session_state["audio_path"] = str(audio_path)
            st.session_state["audio_seconds"] = float(audio_seconds)
            st.session_state["media_basename"] = upload.name
            # Reset inference result whenever audio changes.
            st.session_state["infer_result"] = None
        except Exception as exc:  # noqa: BLE001
            st.error(f"Audio extraction failed: {exc}")

if st.session_state["audio_path"]:
    ac = st.columns(3)
    ac[0].metric("Duration", f"{st.session_state['audio_seconds']:.1f} s")
    ac[1].metric("Format", "16 kHz mono WAV")
    ac[2].metric("Source", st.session_state.get("media_basename", "—"))
    st.audio(st.session_state["audio_path"])
else:
    st.info("Upload an audio or video file to begin.")


# ── Background runner ──────────────────────────────────────────────────
def _run_threaded_with_live_ui(
    target_fn,
    *,
    phase_label: str,
    log_q: "queue.Queue[str]",
    tracker: ResourceTracker,
    stats_placeholder,
    log_placeholder,
    poll_seconds: float = 0.5,
) -> Any:
    """Run target_fn() on a worker thread; meanwhile update placeholders with
    live elapsed/RSS/CPU/log lines. Returns whatever target_fn returns, or
    re-raises its exception."""

    result_box: dict[str, Any] = {}

    def _worker():
        try:
            result_box["value"] = target_fn()
        except BaseException as exc:  # noqa: BLE001
            result_box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    log_lines: list[str] = []
    start = time.time()
    while t.is_alive():
        elapsed = time.time() - start
        with stats_placeholder.container():
            cols = st.columns(4)
            cols[0].metric("Phase", phase_label)
            cols[1].metric("Elapsed (s)", f"{elapsed:.1f}")
            cols[2].metric("Current RSS (MB)", f"{tracker.current_rss_mb:.0f}")
            cols[3].metric("Last CPU %", f"{tracker.last_cpu:.0f}")
        # Drain queue.
        drained = False
        while True:
            try:
                line = log_q.get_nowait()
            except queue.Empty:
                break
            log_lines.append(line)
            drained = True
        if drained or elapsed < 1.0:
            log_placeholder.code(
                "\n".join(log_lines[-200:]) if log_lines else f"[{phase_label}] working…",
                language="text",
            )
        time.sleep(poll_seconds)

    # Final drain.
    while True:
        try:
            log_lines.append(log_q.get_nowait())
        except queue.Empty:
            break
    elapsed = time.time() - start
    with stats_placeholder.container():
        cols = st.columns(4)
        cols[0].metric("Phase", f"{phase_label} ✓")
        cols[1].metric("Elapsed (s)", f"{elapsed:.1f}")
        cols[2].metric("Current RSS (MB)", f"{tracker.current_rss_mb:.0f}")
        cols[3].metric("Last CPU %", f"{tracker.last_cpu:.0f}")
    log_placeholder.code("\n".join(log_lines[-400:]) or f"[{phase_label}] done", language="text")

    if "error" in result_box:
        raise result_box["error"]
    return result_box.get("value")


# ── Load / Run controls ────────────────────────────────────────────────
st.subheader("2. Load model")
load_col1, load_col2, load_col3 = st.columns([1, 1, 1])
load_btn = load_col1.button(
    "Load model",
    type="primary",
    disabled=(st.session_state["engine"] is not None),
    use_container_width=True,
)
unload_btn = load_col2.button(
    "Unload",
    disabled=(st.session_state["engine"] is None),
    use_container_width=True,
)
clear_results_btn = load_col3.button(
    "Clear history",
    disabled=not st.session_state["history"],
    use_container_width=True,
)
if clear_results_btn:
    st.session_state["history"] = []

if unload_btn and st.session_state["engine"] is not None:
    try:
        st.session_state["engine"].unload()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["engine"] = None
    st.session_state["load_result"] = None
    st.session_state["infer_result"] = None
    gc.collect()
    st.success("Model unloaded.")

load_stats = st.empty()
load_log = st.empty()

if load_btn:
    log_q: "queue.Queue[str]" = queue.Queue()
    handler = _install_log_handler(log_q)
    tracker = ResourceTracker()
    tracker.start()
    rss_before = tracker.current_rss_mb
    log_q.put(f"[ui] Instantiating engine '{chosen_label}'…")

    def _do_load():
        engine = instantiate(chosen)
        log_q.put(f"[ui] Calling load() on {type(engine).__name__}")
        engine.load()
        log_q.put("[ui] load() returned")
        return engine

    try:
        t0 = time.time()
        engine = _run_threaded_with_live_ui(
            _do_load,
            phase_label="Loading",
            log_q=log_q,
            tracker=tracker,
            stats_placeholder=load_stats,
            log_placeholder=load_log,
        )
        load_seconds = time.time() - t0
        tracker.stop()
        st.session_state["engine"] = engine
        st.session_state["load_result"] = LoadResult(
            seconds=load_seconds,
            rss_before_mb=rss_before,
            rss_after_mb=tracker.current_rss_mb,
            peak_rss_during_mb=tracker.peak_rss_mb,
        )
        st.session_state["infer_result"] = None
        st.success(f"Loaded in {load_seconds:.1f}s.")
    except Exception as exc:  # noqa: BLE001
        tracker.stop()
        st.error(f"Load failed: {exc}")
    finally:
        _remove_log_handler(handler)

# Load-result card.
if st.session_state["load_result"] is not None:
    lr: LoadResult = st.session_state["load_result"]
    with st.container(border=True):
        st.markdown(f"**Loaded:** `{chosen_label}`")
        c = st.columns(4)
        c[0].metric("Load time (s)", f"{lr.seconds:.2f}")
        c[1].metric("Model RAM (MB)", f"{lr.model_rss_mb:.0f}")
        c[2].metric("Peak RSS during load (MB)", f"{lr.peak_rss_during_mb:.0f}")
        c[3].metric("Process RSS now (MB)", f"{proc.memory_info().rss / 1024**2:.0f}")


# ── Inference ──────────────────────────────────────────────────────────
st.subheader("3. Run inference")
can_infer = (
    st.session_state["engine"] is not None
    and st.session_state["audio_path"] is not None
)
infer_btn = st.button(
    "Run inference",
    type="primary",
    disabled=not can_infer,
    use_container_width=True,
    help=("Load a model and upload audio first." if not can_infer else ""),
)

infer_stats = st.empty()
infer_log = st.empty()

if infer_btn and can_infer:
    log_q2: "queue.Queue[str]" = queue.Queue()
    handler2 = _install_log_handler(log_q2)
    tracker = ResourceTracker()
    tracker.start()
    audio_path = st.session_state["audio_path"]
    audio_seconds = st.session_state["audio_seconds"]

    streamed_segments: list[Segment] = []
    streamed_text_parts: list[str] = []
    text_lock = threading.Lock()

    def _on_segment(seg: Segment) -> None:
        with text_lock:
            streamed_segments.append(seg)
            streamed_text_parts.append(seg.text)
            log_q2.put(
                f"[seg {len(streamed_segments):>3}] "
                f"[{seg.start:6.2f}–{seg.end:6.2f}] {seg.text[:80]}"
            )

    # Capture the engine in the main thread — st.session_state is not safe
    # to access from the background worker thread.
    engine_for_infer = st.session_state["engine"]

    def _do_infer():
        if hasattr(engine_for_infer, "transcribe_stream"):
            log_q2.put("[ui] transcribe_stream(): streaming segments as they decode")
            return engine_for_infer.transcribe_stream(
                audio_path, language="ur", on_segment=_on_segment
            )
        log_q2.put("[ui] transcribe(): no streaming path on this engine")
        return engine_for_infer.transcribe(audio_path, language="ur")

    try:
        t0 = time.time()
        trans = _run_threaded_with_live_ui(
            _do_infer,
            phase_label="Transcribing",
            log_q=log_q2,
            tracker=tracker,
            stats_placeholder=infer_stats,
            log_placeholder=infer_log,
        )
        infer_seconds = time.time() - t0
        tracker.stop()
        res = InferResult(
            seconds=infer_seconds,
            audio_seconds=audio_seconds,
            peak_rss_mb=tracker.peak_rss_mb,
            avg_cpu=tracker.avg_cpu,
            transcription=trans,
            segments_stream=list(streamed_segments),
        )
        st.session_state["infer_result"] = res
        # Append to history for cross-run comparison.
        st.session_state["history"].append(
            {
                "model": chosen_label,
                "engine": chosen.get("engine"),
                "load_s": st.session_state["load_result"].seconds
                if st.session_state["load_result"] else None,
                "infer_s": round(infer_seconds, 2),
                "rtf": round(res.rtf, 2),
                "peak_rss_mb": int(res.peak_rss_mb),
                "avg_cpu": int(res.avg_cpu),
                "words": len(trans.text.split()) if trans else 0,
                "lang": (trans.language if trans else "—"),
            }
        )
        st.success(f"Done in {infer_seconds:.1f}s (RTF {res.rtf:.2f}×).")
    except Exception as exc:  # noqa: BLE001
        tracker.stop()
        st.session_state["infer_result"] = InferResult(
            seconds=time.time() - t0,
            audio_seconds=audio_seconds,
            peak_rss_mb=tracker.peak_rss_mb,
            avg_cpu=tracker.avg_cpu,
            error=str(exc),
        )
        st.error(f"Inference failed: {exc}")
    finally:
        _remove_log_handler(handler2)


# ── Results card ───────────────────────────────────────────────────────
ir: Optional[InferResult] = st.session_state.get("infer_result")
if ir is not None and ir.transcription is not None:
    st.subheader("4. Result")
    with st.container(border=True):
        st.markdown(f"**Model:** `{chosen_label}`")
        m = st.columns(5)
        m[0].metric("Load (s)",
                    f"{st.session_state['load_result'].seconds:.2f}"
                    if st.session_state['load_result'] else "—")
        m[1].metric("Inference (s)", f"{ir.seconds:.2f}")
        m[2].metric("RTF", f"{ir.rtf:.2f}×")
        m[3].metric("Peak RSS (MB)", f"{ir.peak_rss_mb:.0f}")
        m[4].metric("Avg CPU %", f"{ir.avg_cpu:.0f}")

        st.text_area(
            "Transcript",
            ir.transcription.text,
            height=300,
        )

        slug = chosen_label.replace(" ", "_").replace("/", "_")
        st.download_button(
            "Download .txt",
            ir.transcription.text,
            file_name=f"{slug}.txt",
            mime="text/plain",
        )

        if ir.transcription.segments:
            with st.expander(f"Per-segment timing ({len(ir.transcription.segments)} segments)"):
                seg_df = pd.DataFrame(
                    [
                        {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
                        for s in ir.transcription.segments
                    ]
                )
                st.dataframe(seg_df, use_container_width=True, hide_index=True)


# ── History ─────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.subheader("Run history")
    st.dataframe(
        pd.DataFrame(st.session_state["history"]),
        use_container_width=True,
        hide_index=True,
    )
