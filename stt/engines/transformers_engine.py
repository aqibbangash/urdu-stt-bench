"""Generic Hugging Face ASR pipeline adapter.

Good for Wav2Vec2 / XLS-R fine-tunes (e.g. kingabzpro/wav2vec2-large-xls-r-300m-Urdu)
and for any model exposed via `pipeline("automatic-speech-recognition")`.

For Meta MMS specifically, use the dedicated `mms_engine.py` instead — it sets
the language adapter that pipeline() doesn't expose.

Install:  pip install transformers torch torchaudio soundfile
"""
from __future__ import annotations

import warnings

from ..base import Segment, STTEngine, Transcription


def _pick_device() -> str:
    """Prefer MPS on Apple Silicon; fall back to CPU."""
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class TransformersASREngine(STTEngine):
    def __init__(
        self,
        name: str,
        repo_id: str,
        device: str | None = None,
        chunk_length_s: float = 30.0,
        stride_length_s: float = 5.0,
        return_timestamps: bool = True,
    ) -> None:
        """
        :param repo_id: Hugging Face repo id (e.g. "kingabzpro/wav2vec2-...-Urdu").
        :param device: "mps" | "cpu" | None (auto-detect on Apple Silicon).
        :param chunk_length_s: Long-audio chunking; pipeline handles stitching.
        :param return_timestamps: If True, surfaces per-chunk timestamps.
        """
        self.name = name
        self.repo_id = repo_id
        self.device = device or _pick_device()
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.return_timestamps = return_timestamps
        self._pipe = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, f"pip install transformers torch ({exc})"
        return True, ""

    def load(self) -> None:
        from transformers import pipeline

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._pipe = pipeline(
                task="automatic-speech-recognition",
                model=self.repo_id,
                device=self.device,
                chunk_length_s=self.chunk_length_s,
                stride_length_s=self.stride_length_s,
            )

    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:
        if self._pipe is None:
            self.load()
        # Wav2Vec2 fine-tunes don't use a language token, so `language` is
        # informational here (we still pass it to engines that honor it).
        result = self._pipe(
            audio_path,
            return_timestamps=self.return_timestamps,
        )
        text = (result.get("text") or "").strip()
        segments = []
        chunks = result.get("chunks") or []
        for c in chunks:
            ts = c.get("timestamp") or (None, None)
            start = float(ts[0]) if ts[0] is not None else 0.0
            end = float(ts[1]) if ts[1] is not None else start
            segments.append(
                Segment(start=start, end=end, text=(c.get("text") or "").strip())
            )
        return Transcription(
            text=text,
            segments=segments,
            language=language,
            metadata={
                "engine": "transformers",
                "model": self.repo_id,
                "device": self.device,
            },
        )

    def unload(self) -> None:
        # Release model + processor refs; torch will GC.
        self._pipe = None
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
