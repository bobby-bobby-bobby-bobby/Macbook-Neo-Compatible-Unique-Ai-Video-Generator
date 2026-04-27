"""Configuration loader with preset support and environment adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from myvideo.config.schema import PipelineConfig
from myvideo.utils.hardware import detect_hardware



def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result



def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config at {path} must contain a dictionary at top level")
    return data



def load_config(mode: str, config_dir: str | Path = "configs") -> PipelineConfig:
    """Load config by mode and adapt it to detected hardware.

    Args:
        mode: preview, final, or low_memory
        config_dir: directory containing base.yaml and mode-specific YAML files
    """
    config_dir = Path(config_dir)
    base = load_yaml(config_dir / "base.yaml")
    mode_file = config_dir / f"{mode}.yaml"
    if not mode_file.exists():
        raise FileNotFoundError(f"Unknown mode '{mode}'. Missing file: {mode_file}")

    merged = _deep_update(base, load_yaml(mode_file))

    profile = detect_hardware()
    if merged.get("runtime", {}).get("device_preference", "auto") == "auto":
        merged.setdefault("runtime", {})["device_preference"] = profile.device

    if profile.environment in {"mac_apple_silicon_mps", "mac_apple_silicon_cpu"}:
        merged.setdefault("memory", {})["max_frames_in_memory"] = min(
            int(merged.get("memory", {}).get("max_frames_in_memory", 8)),
            6,
        )

    return PipelineConfig.model_validate(merged)
