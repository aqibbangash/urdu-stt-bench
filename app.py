"""Streamlit entry point for the Urdu STT Benchmark — single engine, pure CPU.

Pick any faster-whisper / CTranslate2 model from the sidebar:
  - Standard size shortcuts: tiny / base / small / medium / large-v3
  - Any Hugging Face repo id, e.g. deepdml/faster-whisper-large-v3-turbo-ct2

Flow:
  1. Upload audio/video.
  2. Configure model in the sidebar (size or HF repo id, compute type, …).
  3. Click "Load model" — weights pull into RAM. Live stats + log pane.
  4. Click "Run inference" — transcribe streams segments as they decode.
  5. Result card shows load time, inference time, RTF, peak RAM, transcript.

Pure CPU is the target — faster-whisper / ctranslate2 on Linux x86_64.
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

# Load .env (HF_TOKEN, etc.) before huggingface_hub / faster_whisper imports.
load_dotenv(Path(__file__).parent / ".env")

from stt.base import Segment, Transcription
from stt.metrics import ResourceTracker
from stt.engines.faster_whisper_engine import FasterWhisperEngine
from stt.audio import extract_audio, probe_duration

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

st.set_page_config(
    page_title="Urdu STT Benchmark",
    page_icon="🗣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Logging plumbing ────────────────────────────────────────────────────
class QueueLogHandler(logging.Handler):
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
    for name in ("huggingface_hub", "faster_whisper", "ctranslate2"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
    return handler


def _remove_log_handler(handler: QueueLogHandler) -> None:
    for name in ("huggingface_hub", "faster_whisper", "ctranslate2"):
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
_ss("loaded_config", None)
_ss("load_result", None)
_ss("infer_result", None)
_ss("audio_path", None)
_ss("audio_seconds", None)
_ss("media_basename", None)
_ss("history", [])


# ── Sidebar: pick any faster-whisper model on the fly ───────────────────
st.sidebar.header("Model")

QUICK_PICKS = [
    "(custom)",
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "Systran/faster-distil-whisper-large-v3",
]
pick = st.sidebar.selectbox(
    "Quick pick",
    QUICK_PICKS,
    index=3,  # small
    help="Pick a preset or choose (custom) and type any HF repo id below.",
)
default_model = "" if pick == "(custom)" else pick
model_size = st.sidebar.text_input(
    "Model size or HF repo id",
    value=default_model,
    placeholder="tiny / small / large-v3 / org/repo-ct2",
    help=(
        "Size shortcuts: tiny, base, small, medium, large-v2, large-v3. "
        "Or any Hugging Face repo id hosting a CTranslate2-converted model."
    ),
)
if not model_size.strip():
    model_size = "small"

compute_type = st.sidebar.selectbox(
    "Compute type",
    ["int8", "int8_float16", "float16", "float32"],
    index=0,
    help="int8 is smallest+fastest on CPU. float32 is highest fidelity, multi-GB.",
)

with st.sidebar.expander("Advanced"):
    beam_size = st.number_input("Beam size", 1, 10, 5, step=1)
    vad_filter = st.checkbox(
        "VAD filter (silero)", value=True,
        help="Skip silence via silero VAD. Usually a quality win, slightly slower.",
    )
    language = st.text_input(
        "Language code", value="ur",
        help="ISO 639-1 (e.g. ur, en, ar). Faster-whisper uses this to skip lang-detect.",
    )

active_config = {
    "model_size": model_size.strip(),
    "compute_type": compute_type,
    "beam_size": int(beam_size),
    "vad_filter": bool(vad_filter),
}
chosen_label = f"faster-whisper {active_config['model_size']} ({active_config['compute_type']}, CPU)"

st.sidebar.divider()
st.sidebar.caption(
    "Engine: faster-whisper (CTranslate2) on CPU. Models pulled from Hugging Face "
    "on first load and cached at `$HF_HOME` (persistent across container restarts)."
)

# Drop engine if config changes so RAM isn't pinned to the wrong model.
if (
    st.session_state["loaded_config"] is not None
    and st.session_state["loaded_config"] != active_config
    and st.session_state["engine"] is not None
):
    try:
        st.session_state["engine"].unload()
    except Exception:  # noqa: BLE001
        pass
    st.session_state["engine"] = None
    st.session_state["load_result"] = None
    st.session_state["infer_result"] = None
    st.session_state["loaded_config"] = None
    gc.collect()


# ── Header + process stats ──────────────────────────────────────────────
st.title("Urdu STT Benchmark")
st.caption(
    "Pure-CPU offline speech-to-text via faster-whisper. Pick any HF model from the sidebar; "
    "compare load time, inference time, RAM, and RTF across runs."
)

proc = psutil.Process(os.getpid())
strip = st.columns(4)
strip[0].metric("Process RSS (MB)", f"{proc.memory_info().rss / 1024**2:.0f}")
strip[1].metric("System CPU %", f"{psutil.cpu_percent(interval=None):.0f}")
strip[2].metric("System RAM used %", f"{psutil.virtual_memory().percent:.0f}")
strip[3].metric("Cores", f"{psutil.cpu_count(logical=False)}P / {psutil.cpu_count(logical=True)}L")


# ── Upload ──────────────────────────────────────────────────────────────
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


# ── Background runner with live UI ──────────────────────────────────────
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


# ── Load / Run controls ─────────────────────────────────────────────────
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
    st.session_state["loaded_config"] = None
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
    log_q.put(f"[ui] Instantiating FasterWhisperEngine for '{active_config['model_size']}'")

    cfg_for_load = dict(active_config)

    def _do_load():
        engine = FasterWhisperEngine(
            name=chosen_label,
            model_size=cfg_for_load["model_size"],
            compute_type=cfg_for_load["compute_type"],
            device="cpu",
            beam_size=cfg_for_load["beam_size"],
            vad_filter=cfg_for_load["vad_filter"],
        )
        log_q.put(f"[ui] Calling load() — cpu_threads={engine.cpu_threads}")
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
        st.session_state["loaded_config"] = active_config
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

if st.session_state["load_result"] is not None:
    lr: LoadResult = st.session_state["load_result"]
    with st.container(border=True):
        st.markdown(f"**Loaded:** `{chosen_label}`")
        c = st.columns(4)
        c[0].metric("Load time (s)", f"{lr.seconds:.2f}")
        c[1].metric("Model RAM (MB)", f"{lr.model_rss_mb:.0f}")
        c[2].metric("Peak RSS during load (MB)", f"{lr.peak_rss_during_mb:.0f}")
        c[3].metric("Process RSS now (MB)", f"{proc.memory_info().rss / 1024**2:.0f}")


# ── Inference ───────────────────────────────────────────────────────────
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
    text_lock = threading.Lock()
    lang_for_infer = language.strip() or "ur"

    def _on_segment(seg: Segment) -> None:
        with text_lock:
            streamed_segments.append(seg)
            log_q2.put(
                f"[seg {len(streamed_segments):>3}] "
                f"[{seg.start:6.2f}–{seg.end:6.2f}] {seg.text[:80]}"
            )

    engine_for_infer = st.session_state["engine"]

    def _do_infer():
        log_q2.put(f"[ui] transcribe_stream(): language={lang_for_infer}")
        return engine_for_infer.transcribe_stream(
            audio_path, language=lang_for_infer, on_segment=_on_segment
        )

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
        st.session_state["history"].append(
            {
                "model": active_config["model_size"],
                "compute": active_config["compute_type"],
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


# ── Result card ─────────────────────────────────────────────────────────
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

        st.text_area("Transcript", ir.transcription.text, height=300)

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
