"""Shared data models used by GUI and core layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class GeneratedTagInfo:
    """Metadata for one saved ArUco marker."""

    tag_id: int
    png_path: str
    jpg_path: str


@dataclass(slots=True)
class TagGenerationResult:
    """Result of generating a printable tag set."""

    output_dir: str
    preview_path: str
    preview_jpg_path: str
    tag_ids: list[int]
    dictionary_name: str
    tag_size_mm: float
    canvas_width_px: int
    canvas_height_px: int
    layout_name: str
    rendered_tag_size_mm: float | None = None
    files: list[GeneratedTagInfo] = field(default_factory=list)


@dataclass(slots=True)
class CameraCalibrationData:
    """Camera intrinsics and distortion data."""

    mode: str
    image_width: int
    image_height: int
    camera_matrix: list[list[float]]
    dist_coeffs: list[float]
    reprojection_error: float | None = None
    chessboard_size: tuple[int, int] | None = None
    square_size_mm: float | None = None
    notes: list[str] = field(default_factory=list)

    def camera_matrix_np(self) -> np.ndarray:
        """Return the intrinsic matrix as a NumPy array."""
        return np.asarray(self.camera_matrix, dtype=np.float64)

    def dist_coeffs_np(self) -> np.ndarray:
        """Return distortion coefficients as a NumPy array."""
        return np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1, 1)

    def to_dict(self) -> dict[str, Any]:
        """Convert the calibration payload to JSON-serializable data."""
        return {
            "mode": self.mode,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": self.dist_coeffs,
            "reprojection_error": self.reprojection_error,
            "chessboard_size": list(self.chessboard_size) if self.chessboard_size else None,
            "square_size_mm": self.square_size_mm,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CameraCalibrationData":
        """Build a calibration instance from JSON data."""
        chessboard_size_raw = payload.get("chessboard_size")
        chessboard_size = tuple(chessboard_size_raw) if chessboard_size_raw else None
        return cls(
            mode=str(payload["mode"]),
            image_width=int(payload["image_width"]),
            image_height=int(payload["image_height"]),
            camera_matrix=[[float(value) for value in row] for row in payload["camera_matrix"]],
            dist_coeffs=[float(value) for value in payload["dist_coeffs"]],
            reprojection_error=(
                float(payload["reprojection_error"])
                if payload.get("reprojection_error") is not None
                else None
            ),
            chessboard_size=chessboard_size,
            square_size_mm=(
                float(payload["square_size_mm"])
                if payload.get("square_size_mm") is not None
                else None
            ),
            notes=[str(note) for note in payload.get("notes", [])],
        )


@dataclass(slots=True)
class MarkerPoseObservation:
    """Pose and quality information for a detected marker."""

    tag_id: int
    rvec: list[float]
    tvec: list[float]
    position_mm: list[float]
    reprojection_error_px: float
    tag_size_px: float
    angle_deg: float


@dataclass(slots=True)
class FrameObservation:
    """Accepted frame with all required marker observations."""

    frame_index: int
    timestamp: str
    markers: dict[int, MarkerPoseObservation]
    quality_score: float
    blur_score: float
    mean_tag_size_px: float
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RejectedFrame:
    """Rejected frame metadata."""

    frame_index: int
    timestamp: str
    reason: str


@dataclass(slots=True)
class PairwiseStats:
    """Robust statistics for a point-to-point distance."""

    mean: float
    median: float
    std: float
    mad: float
    samples: int

    def to_dict(self) -> dict[str, float | int]:
        """Convert statistics to a JSON-friendly mapping."""
        return {
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "mad": self.mad,
            "samples": self.samples,
        }


@dataclass(slots=True)
class CalibrationComputationResult:
    """Final exported calibration payload."""

    version: int
    timestamp: str
    solver_mode: str
    aruco_dictionary: str
    tag_size_mm: float
    tag_ids: list[int]
    camera_calibration_file: str
    used_frames: int
    rejected_frames: int
    points_local: dict[str, list[float]]
    pairwise_distances_mm: dict[str, float]
    pairwise_stats_mm: dict[str, PairwiseStats]
    point_variance_mm: dict[str, list[float]]
    quality_overall_score: float
    quality_notes: list[str]
    dropped_geometry_frames: list[int] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize the result into the requested JSON structure."""
        return {
            "version": self.version,
            "timestamp": self.timestamp,
            "solver_mode": self.solver_mode,
            "aruco_dictionary": self.aruco_dictionary,
            "tag_size_mm": self.tag_size_mm,
            "tag_ids": self.tag_ids,
            "camera_calibration_file": self.camera_calibration_file,
            "used_frames": self.used_frames,
            "rejected_frames": self.rejected_frames,
            "points_local": self.points_local,
            "pairwise_distances_mm": self.pairwise_distances_mm,
            "pairwise_stats_mm": {
                key: value.to_dict() for key, value in self.pairwise_stats_mm.items()
            },
            "point_variance_mm": self.point_variance_mm,
            "quality": {
                "overall_score": self.quality_overall_score,
                "notes": self.quality_notes,
            },
            "dropped_geometry_frames": self.dropped_geometry_frames,
        }
