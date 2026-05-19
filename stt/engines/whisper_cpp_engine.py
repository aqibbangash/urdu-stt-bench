"""whisper.cpp adapter via pywhispercpp.

The lowest-RAM Whisper option. Useful as a baseline and for laptops where
running large-v3 in MLX is too memory-heavy.

The .bin model file is NOT downloaded automatically — download it yourself:

    mkdir -p models/whisper-cpp
    curl -L -o models/whisper-cpp/ggml-large-v3.bin \\
      https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin

Then reference it from models.yaml with `model_file: models/whisper-cpp/ggml-large-v3.bin`.

Install:  pip install pywhispercpp
"""
from __future__ import annotations

from pathlib import Path

from ..base import Segment, STTEngine, Transcription


class WhisperCppEngine(STTEngine):
    def __init__(
        self,
        name: str,
        model_file: str,
        n_threads: int = 8,
    ) -> None:
        self.name = name
        self.model_file = str(Path(model_file))
        self.n_threads = n_threads
        self._model = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import pywhispercpp  # noqa: F401
        except ImportError:
            return False, "pip install pywhispercpp (optional)"
        return True, ""

    def load(self) -> None:
        if not Path(self.model_file).exists():
            raise FileNotFoundError(
                f"whisper.cpp model not found at {self.model_file}. "
                "Download a ggml-*.bin from "
                "https://huggingface.co/ggerganov/whisper.cpp first."
            )
        from pywhispercpp.model import Model

        self._model = Model(self.model_file, n_threads=self.n_threads)

    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:
        if self._model is None:
            self.load()
        # whisper.cpp segments expose t0 / t1 in centiseconds.
        segments = self._model.transcribe(audio_path, language=language)
        out = []
        text_parts = []
        for s in segments:
            t = (s.text or "").strip()
            out.append(
                Segment(start=s.t0 / 100.0, end=s.t1 / 100.0, text=t)
            )
            text_parts.append(t)
        return Transcription(
            text=" ".join(text_parts).strip(),
            segments=out,
            language=language,
            metadata={
                "engine": "whisper.cpp",
                "model_file": self.model_file,
                "n_threads": self.n_threads,
            },
        )

    def unload(self) -> None:
        self._model = None
