"""Model registry — resolve an engine type string to its STTEngine class.

Single-engine build: only faster-whisper is shipped. To add another engine
(e.g. a CUDA path, or whisper.cpp), drop the adapter under
`stt/engines/<engine>_engine.py` and add an entry to `ENGINE_TYPES`.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml

from .base import STTEngine

# YAML `engine:` value  →  (module name relative to stt.engines, class name)
ENGINE_TYPES: dict[str, tuple[str, str]] = {
    "faster-whisper": ("faster_whisper_engine", "FasterWhisperEngine"),
}


def load_registry(path: Path) -> dict[str, Any]:
    """Parse models.yaml. Returns the raw dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "models" not in data or not isinstance(data["models"], list):
        raise ValueError(f"{path} is missing a top-level `models:` list")
    return data


def _resolve_engine_class(engine_type: str) -> type[STTEngine]:
    if engine_type not in ENGINE_TYPES:
        raise ValueError(
            f"Unknown engine type '{engine_type}'. "
            f"Known types: {sorted(ENGINE_TYPES)}"
        )
    mod_name, cls_name = ENGINE_TYPES[engine_type]
    module = importlib.import_module(f"stt.engines.{mod_name}")
    cls = getattr(module, cls_name)
    return cls


def instantiate(model_config: dict[str, Any]) -> STTEngine:
    """Build an engine instance from one entry in models.yaml."""
    engine_type = model_config["engine"]
    name = model_config["name"]
    params = model_config.get("params") or {}
    cls = _resolve_engine_class(engine_type)
    return cls(name=name, **params)


def availability_status(engine_type: str) -> tuple[bool, str]:
    """Best-effort check that the engine's optional deps import.

    Returns (True, "") if importable, else (False, hint).
    Used by the Streamlit sidebar to grey out unavailable engines.
    """
    if engine_type not in ENGINE_TYPES:
        return False, f"Unknown engine type: {engine_type}"
    mod_name, _ = ENGINE_TYPES[engine_type]
    try:
        importlib.import_module(f"stt.engines.{mod_name}")
    except ImportError as exc:
        return False, f"Missing dependency: {exc}. See requirements.txt."
    # Then check the engine's own availability hook if defined.
    try:
        cls = _resolve_engine_class(engine_type)
        check = getattr(cls, "is_available", None)
        if callable(check):
            return check()
    except ImportError as exc:
        return False, f"Missing dependency: {exc}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Init failed: {exc}"
    return True, ""
