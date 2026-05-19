"""Abstract base for STT engines + shared dataclasses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Segment:
    """One timestamped chunk of transcript."""

    start: float  # seconds from the start of the audio
    end: float
    text: str
    confidence: Optional[float] = None  # engine-specific; not all populate this


@dataclass
class Transcription:
    """Standard output shape every engine returns."""

    text: str
    segments: list[Segment] = field(default_factory=list)
    language: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class STTEngine(ABC):
    """Subclass this to add a new model adapter.

    Implement `load()` and `transcribe()`. Override `unload()` if you can free
    memory between runs (called between models in the benchmark).

    Engines are instantiated by `stt.registry.instantiate(model_config)`.
    """

    #: Human-readable name. Set from the YAML config in __init__.
    name: str = "base"

    @abstractmethod
    def load(self) -> None:  # pragma: no cover
        """Pull weights into memory. Called lazily on first transcribe()."""

    @abstractmethod
    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:  # pragma: no cover
        """Transcribe a 16 kHz mono WAV. Return a Transcription."""

    def unload(self) -> None:
        """Optional. Free model memory between benchmark runs."""
        return None
