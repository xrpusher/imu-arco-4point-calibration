"""Frame acceptance logic for calibration sessions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from app.core.models import FrameObservation, MarkerPoseObservation, RejectedFrame


@dataclass(slots=True)
class SessionRecorderConfig:
    """Thresholds for recording good calibration frames."""

    required_ids: list[int]
    min_tag_size_px: float = 18.0
    blur_threshold: float = 25.0
    max_reprojection_error_px: float = 8.0
    max_angle_deg: float = 82.0
    min_mean_displacement_mm: float = 5.0
    target_frames: int = 8


class SessionRecorder:
    """Collect accepted and rejected frames with basic quality rules."""

    def __init__(self, config: SessionRecorderConfig) -> None:
        self.config = config
        self.accepted_frames: list[FrameObservation] = []
        self.rejected_frames: list[RejectedFrame] = []
        self.last_accepted_positions: dict[int, np.ndarray] | None = None
        self.active = False

    def start(self) -> None:
        """Start a new recording session."""
        self.accepted_frames.clear()
        self.rejected_frames.clear()
        self.last_accepted_positions = None
        self.active = True

    def stop(self) -> None:
        """Stop the current recording session."""
        self.active = False

    def process_frame(
        self,
        frame_index: int,
        detections: dict[int, MarkerPoseObservation],
        blur_score: float,
    ) -> tuple[bool, str]:
        """Evaluate one frame and either accept it or store a reject reason."""
        if not self.active:
            return False, "Сессия не активна."

        timestamp = datetime.now().isoformat(timespec="seconds")
        required = set(self.config.required_ids)
        visible = set(detections)
        missing = sorted(required - visible)
        if missing:
            return self._reject(frame_index, timestamp, f"Не хватает нужных ID: {missing}")

        selected = {tag_id: detections[tag_id] for tag_id in self.config.required_ids}
        reprojection_errors = [item.reprojection_error_px for item in selected.values()]
        mean_reprojection_error = float(np.mean(reprojection_errors))
        if mean_reprojection_error > self.config.max_reprojection_error_px:
            return self._reject(
                frame_index,
                timestamp,
                f"Слишком большая reprojection error: {mean_reprojection_error:.2f}px",
            )

        mean_tag_size = float(np.mean([item.tag_size_px for item in selected.values()]))
        if mean_tag_size < self.config.min_tag_size_px:
            return self._reject(
                frame_index,
                timestamp,
                f"Теги слишком мелкие: {mean_tag_size:.1f}px",
            )

        max_angle = float(np.max([item.angle_deg for item in selected.values()]))
        if max_angle > self.config.max_angle_deg:
            return self._reject(
                frame_index,
                timestamp,
                f"Слишком острый угол обзора: {max_angle:.1f} град",
            )

        if blur_score < self.config.blur_threshold:
            return self._reject(
                frame_index,
                timestamp,
                f"Кадр слишком смазан: {blur_score:.1f}",
            )

        if self._is_duplicate(selected):
            return self._reject(
                frame_index,
                timestamp,
                "Кадр слишком похож на предыдущий принятый кадр.",
            )

        quality_score = self._quality_score(mean_tag_size, blur_score, mean_reprojection_error, max_angle)
        accepted = FrameObservation(
            frame_index=frame_index,
            timestamp=timestamp,
            markers=selected,
            quality_score=quality_score,
            blur_score=blur_score,
            mean_tag_size_px=mean_tag_size,
            notes=[
                f"Средняя reprojection error: {mean_reprojection_error:.2f}px",
                f"Максимальный угол тега: {max_angle:.1f} град",
            ],
        )
        self.accepted_frames.append(accepted)
        self.last_accepted_positions = {
            tag_id: np.asarray(observation.position_mm, dtype=np.float64)
            for tag_id, observation in selected.items()
        }
        return True, "Принят"

    def _reject(self, frame_index: int, timestamp: str, reason: str) -> tuple[bool, str]:
        self.rejected_frames.append(
            RejectedFrame(frame_index=frame_index, timestamp=timestamp, reason=reason)
        )
        return False, reason

    def _is_duplicate(self, detections: dict[int, MarkerPoseObservation]) -> bool:
        if self.last_accepted_positions is None:
            return False

        displacements: list[float] = []
        for tag_id, observation in detections.items():
            previous = self.last_accepted_positions.get(tag_id)
            if previous is None:
                continue
            current = np.asarray(observation.position_mm, dtype=np.float64)
            displacements.append(float(np.linalg.norm(current - previous)))

        if not displacements:
            return False
        return float(np.mean(displacements)) < self.config.min_mean_displacement_mm

    def _quality_score(
        self,
        mean_tag_size: float,
        blur_score: float,
        mean_reprojection_error: float,
        max_angle: float,
    ) -> float:
        size_score = np.clip(mean_tag_size / (self.config.min_tag_size_px * 2.2), 0.0, 1.0)
        blur_norm = np.clip(
            (blur_score - self.config.blur_threshold) / max(self.config.blur_threshold * 2.0, 1.0),
            0.0,
            1.0,
        )
        reproj_score = np.clip(
            1.0 - (mean_reprojection_error / max(self.config.max_reprojection_error_px, 1e-6)),
            0.0,
            1.0,
        )
        angle_score = np.clip(
            1.0 - (max_angle / max(self.config.max_angle_deg, 1e-6)),
            0.0,
            1.0,
        )
        return float(
            100.0 * (0.35 * size_score + 0.25 * blur_norm + 0.25 * reproj_score + 0.15 * angle_score)
        )
