"""Small logging helpers for the GUI."""

from __future__ import annotations

from datetime import datetime


def format_log_message(message: str) -> str:
    """Format a single log line with a local timestamp."""
    return f"[{datetime.now().strftime('%H:%M:%S')}] {message}"


def exception_to_text(exc: Exception) -> str:
    """Render an exception in a compact, user-friendly form."""
    return f"{type(exc).__name__}: {exc}"
