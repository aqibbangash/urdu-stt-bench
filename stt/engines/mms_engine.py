"""Meta MMS adapter — facebook/mms-1b-all + Urdu language adapter.

MMS uses Wav2Vec2ForCTC under the hood but needs a per-language adapter
loaded explicitly. The transformers `pipeline()` doesn't expose
`load_adapter`, so this engine drives the model directly.

Install:  pip install transformers torch torchaudio soundfile librosa
"""
from __future__ import annotations

import warnings

from ..base import STTEngine, Transcription


def _pick_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class MMSEngine(STTEngine):
    def __init__(
        self,
        name: str,
        repo_id: str = "facebook/mms-1b-all",
        target_lang: str = "urd",  # ISO-639-3 — MMS uses 3-letter codes
        device: str | None = None,
        chunk_length_s: float = 20.0,
    ) -> None:
        self.name = name
        self.repo_id = repo_id
        self.target_lang = target_lang
        self.device = device or _pick_device()
        self.chunk_length_s = chunk_length_s
        self._model = None
        self._processor = None

    @classmethod
    def is_available(cls) -> tuple[bool, str]:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
        except ImportError as exc:
            return False, f"pip install transformers torch torchaudio ({exc})"
        return True, ""

    def load(self) -> None:
        from transformers import AutoModelForCTC, AutoProcessor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._processor = AutoProcessor.from_pretrained(self.repo_id)
            self._model = AutoModelForCTC.from_pretrained(self.repo_id).to(self.device)
            self._processor.tokenizer.set_target_lang(self.target_lang)
            self._model.load_adapter(self.target_lang)

    def transcribe(self, audio_path: str, language: str = "ur") -> Transcription:
        if self._model is None or self._processor is None:
            self.load()
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:  # mix down if stereo
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        chunk_samples = int(self.chunk_length_s * sr)
        n = waveform.shape[1]
        chunks = []
        for i in range(0, n, chunk_samples):
            chunk = waveform[:, i : i + chunk_samples].squeeze(0)
            inputs = self._processor(
                chunk.numpy(), sampling_rate=sr, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = self._model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(ids)[0]
            chunks.append(text.strip())

        return Transcription(
            text=" ".join(c for c in chunks if c).strip(),
            segments=[],  # MMS direct path doesn't surface timestamps cheaply
            language=self.target_lang,
            metadata={
                "engine": "mms",
                "model": self.repo_id,
                "target_lang": self.target_lang,
                "device": self.device,
            },
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass
