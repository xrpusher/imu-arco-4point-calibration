"""Simple webcam management helpers."""

from __future__ import annotations

import sys
import threading
from contextlib import suppress

import cv2
import numpy as np

COMMON_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 480),
    (1280, 720),
    (1600, 900),
    (1920, 1080),
]

OPEN_TIMEOUT_SECONDS = 2.5


def resolution_label(resolution: tuple[int, int]) -> str:
    """Return a human-readable resolution label."""
    return f"{resolution[0]}x{resolution[1]}"


def parse_resolution_label(label: str) -> tuple[int, int]:
    """Parse a resolution label produced by resolution_label."""
    width_text, height_text = label.lower().split("x", maxsplit=1)
    return int(width_text), int(height_text)


class CameraManager:
    """Thin wrapper around cv2.VideoCapture with a timeout on open."""

    def __init__(self) -> None:
        self.capture: cv2.VideoCapture | None = None
        self.index: int | None = None

    @staticmethod
    def _backend_preferences() -> list[tuple[int, str]]:
        if sys.platform.startswith("win"):
            return [
                (cv2.CAP_MSMF, "MSMF"),
                (cv2.CAP_DSHOW, "DirectShow"),
            ]
        return [(cv2.CAP_ANY, "Default")]

    @staticmethod
    def _suppress_opencv_logs() -> tuple[object | None, object | None]:
        previous_level = None
        log_api = None
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "getLogLevel"):
            with suppress(Exception):
                previous_level = cv2.getLogLevel()
                cv2.setLogLevel(0)
                log_api = "cv2"
        return previous_level, log_api

    @staticmethod
    def _restore_opencv_logs(previous_level: object | None, log_api: object | None) -> None:
        if log_api == "cv2" and previous_level is not None:
            with suppress(Exception):
                cv2.setLogLevel(previous_level)

    def _try_open_backend(
        self,
        index: int,
        resolution: tuple[int, int],
        backend: int,
        backend_name: str,
        timeout_seconds: float = OPEN_TIMEOUT_SECONDS,
    ) -> tuple[cv2.VideoCapture | None, tuple[int, int] | None, str | None]:
        result: dict[str, object] = {}
        done = threading.Event()
        cancelled = threading.Event()

        def worker() -> None:
            capture: cv2.VideoCapture | None = None
            try:
                capture = cv2.VideoCapture(index, backend)
                if not capture.isOpened():
                    result["error"] = f"{backend_name}: устройство не открылось"
                    return

                capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                success, _ = capture.read()
                if not success:
                    result["error"] = f"{backend_name}: не удалось получить первый кадр"
                    return

                if cancelled.is_set():
                    return

                result["capture"] = capture
                result["resolution"] = (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
                capture = None
            except Exception as exc:  # pragma: no cover - defensive branch
                result["error"] = f"{backend_name}: {exc}"
            finally:
                if capture is not None:
                    with suppress(Exception):
                        capture.release()
                done.set()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        if not done.wait(timeout_seconds):
            cancelled.set()
            return None, None, f"{backend_name}: таймаут открытия камеры"

        capture = result.get("capture")
        actual_resolution = result.get("resolution")
        error = result.get("error")
        return (
            capture if isinstance(capture, cv2.VideoCapture) else None,
            actual_resolution if isinstance(actual_resolution, tuple) else None,
            str(error) if error else None,
        )

    def open(self, index: int, resolution: tuple[int, int]) -> tuple[int, int]:
        """Open a camera and request a resolution."""
        self.stop()

        previous_level, log_api = self._suppress_opencv_logs()
        backend_errors: list[str] = []
        try:
            for backend, backend_name in self._backend_preferences():
                capture, actual_resolution, error = self._try_open_backend(index, resolution, backend, backend_name)
                if capture is not None and actual_resolution is not None:
                    self.capture = capture
                    self.index = index
                    return actual_resolution
                if error:
                    backend_errors.append(error)
        finally:
            self._restore_opencv_logs(previous_level, log_api)

        details = "; ".join(backend_errors[:3])
        raise RuntimeError(
            f"Не удалось открыть камеру с индексом {index}. "
            "Попробуйте другой индекс, закройте Zoom/Teams/OBS и проверьте шторку камеры. "
            f"Подробности: {details}"
        )

    def read_frame(self) -> np.ndarray:
        """Read a single frame."""
        if self.capture is None or not self.capture.isOpened():
            raise RuntimeError("Камера не запущена.")
        success, frame = self.capture.read()
        if not success or frame is None:
            raise RuntimeError("Не удалось прочитать кадр с камеры.")
        return frame

    def actual_resolution(self) -> tuple[int, int]:
        """Return the current frame resolution."""
        if self.capture is None:
            return 0, 0
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def stop(self) -> None:
        """Release the camera."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            self.index = None

    def is_open(self) -> bool:
        """Return True if the camera is active."""
        return self.capture is not None and self.capture.isOpened()


def enumerate_camera_indices(max_index: int = 6) -> list[int]:
    """Return a short list of likely camera indices without slow probing."""
    return list(range(max_index))
