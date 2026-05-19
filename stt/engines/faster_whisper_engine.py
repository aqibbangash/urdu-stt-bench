"""faster-whisper adapter — CTranslate2 quantized Whisper.

Apple Silicon Macs don't have CT2 GPU support yet, so this runs on CPU.
Still useful as an MLX-independent baseline.

Install:  pip install faster-whisper
"""
from __future__ import annotations

import os
from typing import Callable, Iterator, Optional

from ..base import Segment, STTEngine, Transcription


class FasterWhisperEngine(STTEngine):
    def __init__(
        self,
        name: str,
        model_size: str = "large-v3",
        compute_type: str = "int8",
        device: str = "cpu",
        beam_size: int = 5,
        vad_filter: bool = True,
        cpu_threads: int = 0,
        num_workers: int = 1,
    ) -> None:
        """
        :param model_size: HF repo id or one of "tiny"/"small"/"medium"/"large-v3".
        :param compute_type: "int8" (smallest), "int8_float16", "float16", "float32".
        :param device: "cpu" (Apple Silicon — CT2 Metal isn't ready) or "cuda".
        :param vad_filter: silero VAD to skip silence — usually a quality win.
        :param cpu_threads: 0 = library default. Overridden by URDU_STT_CPU_THREADS
            env var when set, so docker compose can pin threads to the CPU cap.
        :param num_workers: ctranslate2 parallel workers. Usually 1 for single-stream.
        """
        self.name = name
        self.model_size = model_size
        self.compute_type = compute_type
        self.device = device
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        # Env var wins if set — keeps Dockerfile/compose in charge of the cap.
        env_threads = os.environ.get("URDU_STT_CPU_THREADS")
        self.cpu_threads = int(env_threads) if env_threads else cpu_threads
        self.num_workers = num_workers
        self._model = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "pip install faster-whisper"
        return True, ""

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
        )

    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:
        if self._model is None:
            self.load()
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        segments = []
        text_parts = []
        for s in segments_iter:
            txt = (s.text or "").strip()
            segments.append(
                Segment(
                    start=float(s.start),
                    end=float(s.end),
                    text=txt,
                    confidence=float(s.avg_logprob) if s.avg_logprob is not None else None,
                )
            )
            text_parts.append(txt)
        return Transcription(
            text=" ".join(text_parts).strip(),
            segments=segments,
            language=info.language,
            metadata={
                "engine": "faster-whisper",
                "model": self.model_size,
                "compute_type": self.compute_type,
                "device": self.device,
                "cpu_threads": self.cpu_threads,
                "language_probability": float(info.language_probability),
            },
        )

    def transcribe_stream(
        self,
        audio_path: str,
        language: str = "ur",
        on_segment: Optional[Callable[[Segment], None]] = None,
    ) -> Transcription:
        """Stream segments as they decode. on_segment is called for each Segment
        before the function returns the full Transcription. Useful for live UIs.
        """
        if self._model is None:
            self.load()
        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        segments = []
        text_parts = []
        for s in segments_iter:
            txt = (s.text or "").strip()
            seg = Segment(
                start=float(s.start),
                end=float(s.end),
                text=txt,
                confidence=float(s.avg_logprob) if s.avg_logprob is not None else None,
            )
            segments.append(seg)
            text_parts.append(txt)
            if on_segment is not None:
                try:
                    on_segment(seg)
                except Exception:  # noqa: BLE001
                    pass
        return Transcription(
            text=" ".join(text_parts).strip(),
            segments=segments,
            language=info.language,
            metadata={
                "engine": "faster-whisper",
                "model": self.model_size,
                "compute_type": self.compute_type,
                "device": self.device,
                "cpu_threads": self.cpu_threads,
                "language_probability": float(info.language_probability),
            },
        )

    def unload(self) -> None:
        self._model = None
