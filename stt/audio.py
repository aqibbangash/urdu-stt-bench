"""ffmpeg-based audio extraction + duration probe.

We shell out to ffmpeg / ffprobe directly (no Python ffmpeg wrapper needed).
The output is always mono 16 kHz signed-16 PCM — the lingua franca for every
ASR engine we support.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path


def extract_audio(
    media_path: Path | str,
    sample_rate: int = 16000,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Extract mono PCM_S16LE audio at `sample_rate` Hz. Returns the WAV path.

    Raises FileNotFoundError if `ffmpeg` is not on PATH, and
    subprocess.CalledProcessError if extraction fails.
    """
    media_path = Path(media_path)
    if not media_path.exists():
        raise FileNotFoundError(media_path)

    out_dir = out_dir or Path(tempfile.gettempdir())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"urdu_stt_{os.getpid()}_{time.time_ns()}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install with `brew install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg failed: {stderr}") from exc

    return out_path


def probe_duration(audio_path: Path | str) -> float:
    """Return audio duration in seconds via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(audio_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "ffprobe not found on PATH. Install with `brew install ffmpeg`."
        ) from exc
    out = result.stdout.strip()
    if not out:
        return 0.0
    try:
        return float(out)
    except ValueError:
        return 0.0
