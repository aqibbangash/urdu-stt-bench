"""lightning-whisper-mlx adapter — Apple Silicon accelerated.

This is the fastest Whisper path on M-series Macs because work runs on the
Apple Neural Engine / Metal via the MLX framework.

Install:  pip install lightning-whisper-mlx mlx
Requires: Apple Silicon Mac (M1 / M2 / M3 / M4)
"""
from __future__ import annotations

from typing import Optional

from ..base import Segment, STTEngine, Transcription


class MLXWhisperEngine(STTEngine):
    def __init__(
        self,
        name: str,
        model_size: str = "large-v3",
        quant: Optional[str] = None,
        batch_size: int = 12,
    ) -> None:
        """
        :param model_size: One of "tiny", "small", "medium", "large-v2",
            "large-v3", "large-v3-turbo", "distil-large-v3", etc.
        :param quant: None | "4bit" | "8bit". Quantization saves RAM and
            modestly speeds up inference; quality drop is small at 4-bit
            on these models.
        :param batch_size: Internal batch size; raise to use more memory
            for faster throughput on long audio.
        """
        self.name = name
        self.model_size = model_size
        self.quant = quant
        self.batch_size = batch_size
        self._model = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import lightning_whisper_mlx  # noqa: F401
        except ImportError:
            return False, "pip install lightning-whisper-mlx mlx (Apple Silicon only)"
        return True, ""

    def load(self) -> None:
        from lightning_whisper_mlx import LightningWhisperMLX

        self._model = LightningWhisperMLX(
            model=self.model_size,
            batch_size=self.batch_size,
            quant=self.quant,
        )

    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:
        if self._model is None:
            self.load()
        result = self._model.transcribe(audio_path=audio_path, language=language)
        text = (result.get("text") or "").strip()
        raw_segments = result.get("segments") or []
        # lightning-whisper-mlx returns segments as [start_frame, end_frame, text]
        # where frame counts use HOP_LENGTH=160 at SAMPLE_RATE=16000 → seconds = frame * 0.01.
        from lightning_whisper_mlx.audio import HOP_LENGTH, SAMPLE_RATE
        frame_to_sec = HOP_LENGTH / SAMPLE_RATE
        segments: list[Segment] = []
        for s in raw_segments:
            if isinstance(s, dict):
                start = float(s.get("start", 0.0))
                end = float(s.get("end", 0.0))
                seg_text = (s.get("text") or "").strip()
            else:
                start = float(s[0]) * frame_to_sec
                end = float(s[1]) * frame_to_sec
                seg_text = str(s[2]).strip() if len(s) > 2 else ""
            segments.append(Segment(start=start, end=end, text=seg_text))
        return Transcription(
            text=text,
            segments=segments,
            language=result.get("language") or language,
            metadata={
                "engine": "lightning-whisper-mlx",
                "model": self.model_size,
                "quant": self.quant,
                "batch_size": self.batch_size,
            },
        )

    def unload(self) -> None:
        self._model = None
