"""Path helpers used across the application."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def output_root() -> Path:
    """Return the default output directory."""
    return ensure_dir(project_root() / "output")


def timestamped_output_dir(base_dir: str | Path, prefix: str) -> Path:
    """Create a timestamped child directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(base_dir) / f"{prefix}_{timestamp}")
