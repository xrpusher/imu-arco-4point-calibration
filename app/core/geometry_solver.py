"""Robust geometry estimation for the four IMU attachment points."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Iterable

import numpy as np

from app.core.models import CalibrationComputationResult, FrameObservation, PairwiseStats

try:
    from scipy.optimize import least_squares
except Exception:  # pragma: no cover - optional runtime dependency fallback
    least_squares = None


def _pair_key(tag_a: int, tag_b: int) -> str:
    left, right = sorted((tag_a, tag_b))
    return f"{left}-{right}"


def _mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)))


def _robust_mask(values: np.ndarray, threshold: float = 3.5) -> np.ndarray:
    if len(values) <= 2:
        return np.ones(len(values), dtype=bool)

    median = float(np.median(values))
    mad = _mad(values)
    if mad < 1e-6:
        std = float(np.std(values))
        if std < 1e-6:
            return np.ones(len(values), dtype=bool)
        return np.abs(values - median) <= max(3.0 * std, 1.0)

    robust_z = 0.6745 * (values - median) / mad
    return np.abs(robust_z) <= threshold


def _stats(values: Iterable[float]) -> PairwiseStats:
    array = np.asarray(list(values), dtype=np.float64)
    return PairwiseStats(
        mean=float(np.mean(array)),
        median=float(np.median(array)),
        std=float(np.std(array)),
        mad=_mad(array),
        samples=int(len(array)),
    )


def _frame_pairwise_distances(frame: FrameObservation, tag_ids: list[int]) -> dict[str, float]:
    output: dict[str, float] = {}
    for tag_a, tag_b in combinations(tag_ids, 2):
        point_a = np.asarray(frame.markers[tag_a].position_mm, dtype=np.float64)
        point_b = np.asarray(frame.markers[tag_b].position_mm, dtype=np.float64)
        output[_pair_key(tag_a, tag_b)] = float(np.linalg.norm(point_a - point_b))
    return output


def _robust_pairwise_distances(
    frames: list[FrameObservation],
    tag_ids: list[int],
) -> tuple[dict[str, float], dict[str, PairwiseStats]]:
    samples_by_pair: dict[str, list[float]] = defaultdict(list)
    for frame in frames:
        frame_distances = _frame_pairwise_distances(frame, tag_ids)
        for key, value in frame_distances.items():
            samples_by_pair[key].append(value)

    final_distances: dict[str, float] = {}
    final_stats: dict[str, PairwiseStats] = {}
    for key, raw_values in samples_by_pair.items():
        values = np.asarray(raw_values, dtype=np.float64)
        mask = _robust_mask(values)
        filtered = values[mask]
        final_distances[key] = float(np.median(filtered))
        final_stats[key] = _stats(filtered)
    return final_distances, final_stats


def _select_consistent_frames(
    frames: list[FrameObservation],
    tag_ids: list[int],
    target_distances: dict[str, float],
) -> tuple[list[FrameObservation], list[int]]:
    if len(frames) <= 3:
        return frames, []

    residuals: list[float] = []
    for frame in frames:
        frame_distances = _frame_pairwise_distances(frame, tag_ids)
        errors = [
            abs(frame_distances[key] - target_distances[key]) for key in target_distances.keys()
        ]
        residuals.append(float(np.median(errors)) if errors else 0.0)

    residual_array = np.asarray(residuals, dtype=np.float64)
    mask = _robust_mask(residual_array)
    kept_frames = [frame for frame, keep in zip(frames, mask) if keep]
    dropped = [frame.frame_index for frame, keep in zip(frames, mask) if not keep]
    return kept_frames, dropped


def _mds_embedding(tag_ids: list[int], pairwise_distances: dict[str, float]) -> np.ndarray:
    size = len(tag_ids)
    distance_matrix = np.zeros((size, size), dtype=np.float64)
    for index_a, tag_a in enumerate(tag_ids):
        for index_b, tag_b in enumerate(tag_ids):
            if index_a == index_b:
                continue
            distance_matrix[index_a, index_b] = pairwise_distances[_pair_key(tag_a, tag_b)]

    squared = distance_matrix**2
    identity = np.eye(size)
    ones = np.ones((size, size)) / size
    centered = -0.5 * (identity - ones) @ squared @ (identity - ones)
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order][:3], 0.0)
    eigenvectors = eigenvectors[:, order][:, :3]
    coordinates = eigenvectors * np.sqrt(eigenvalues)
    coordinates -= coordinates[0]
    return coordinates


def _simple_coordinates_from_distances(
    tag_ids: list[int],
    pairwise_distances: dict[str, float],
) -> dict[str, list[float]]:
    if len(tag_ids) == 4 and least_squares is not None:
        ordered = list(tag_ids)

        def build_points(parameters: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    [0.0, 0.0, 0.0],
                    [parameters[0], 0.0, 0.0],
                    [parameters[1], parameters[2], 0.0],
                    [parameters[3], parameters[4], parameters[5]],
                ],
                dtype=np.float64,
            )

        def residuals(parameters: np.ndarray) -> np.ndarray:
            points = build_points(parameters)
            errors: list[float] = []
            for index_a, tag_a in enumerate(ordered):
                for index_b, tag_b in enumerate(ordered[index_a + 1 :], start=index_a + 1):
                    target = pairwise_distances[_pair_key(tag_a, tag_b)]
                    estimate = np.linalg.norm(points[index_a] - points[index_b])
                    errors.append(estimate - target)
            return np.asarray(errors, dtype=np.float64)

        initial_coordinates = _mds_embedding(tag_ids, pairwise_distances)
        initial = np.array(
            [
                max(np.linalg.norm(initial_coordinates[1] - initial_coordinates[0]), 1.0),
                initial_coordinates[2, 0],
                max(abs(initial_coordinates[2, 1]), 1.0),
                initial_coordinates[3, 0],
                initial_coordinates[3, 1],
                initial_coordinates[3, 2],
            ],
            dtype=np.float64,
        )
        result = least_squares(residuals, initial, loss="soft_l1")
        coordinates = build_points(result.x)
    else:
        coordinates = _mds_embedding(tag_ids, pairwise_distances)

    return {
        str(tag_id): [float(value) for value in coordinates[index]]
        for index, tag_id in enumerate(tag_ids)
    }


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Вырожденный базис локальной системы координат.")
    return vector / norm


def _frame_local_coordinates(
    frame: FrameObservation,
    tag_ids: list[int],
    upper_id: int,
    lower_id: int,
    plane_id: int,
) -> dict[int, np.ndarray]:
    origin = np.asarray(frame.markers[upper_id].position_mm, dtype=np.float64)
    lower = np.asarray(frame.markers[lower_id].position_mm, dtype=np.float64)
    plane = np.asarray(frame.markers[plane_id].position_mm, dtype=np.float64)

    axis_y = _normalize(lower - origin)
    plane_vector = plane - origin
    axis_z = _normalize(np.cross(axis_y, plane_vector))
    axis_x = _normalize(np.cross(axis_z, axis_y))
    basis = np.vstack([axis_x, axis_y, axis_z])

    local: dict[int, np.ndarray] = {}
    for tag_id in tag_ids:
        point = np.asarray(frame.markers[tag_id].position_mm, dtype=np.float64)
        local[tag_id] = basis @ (point - origin)
    return local


def _aggregate_local_coordinates(
    frames: list[FrameObservation],
    tag_ids: list[int],
    upper_id: int,
    lower_id: int,
    plane_id: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]], list[int]]:
    per_frame_coordinates: list[tuple[int, dict[int, np.ndarray]]] = []
    dropped_frames: list[int] = []
    for frame in frames:
        try:
            per_frame_coordinates.append(
                (
                    frame.frame_index,
                    _frame_local_coordinates(frame, tag_ids, upper_id, lower_id, plane_id),
                )
            )
        except ValueError:
            dropped_frames.append(frame.frame_index)

    if not per_frame_coordinates:
        raise ValueError("После построения torso frame не осталось корректных кадров.")

    points_local: dict[str, list[float]] = {}
    point_variance: dict[str, list[float]] = {}
    for tag_id in tag_ids:
        coordinates = np.asarray(
            [frame_data[tag_id] for _, frame_data in per_frame_coordinates],
            dtype=np.float64,
        )
        medians: list[float] = []
        stds: list[float] = []
        for axis_index in range(3):
            axis_values = coordinates[:, axis_index]
            mask = _robust_mask(axis_values)
            filtered_axis = axis_values[mask]
            medians.append(float(np.median(filtered_axis)))
            stds.append(float(np.std(filtered_axis)))
        points_local[str(tag_id)] = medians
        point_variance[str(tag_id)] = stds
    return points_local, point_variance, dropped_frames


def solve_geometry(
    frames: list[FrameObservation],
    tag_ids: list[int],
    aruco_dictionary: str,
    tag_size_mm: float,
    camera_calibration_file: str,
    rejected_frames_count: int,
    solver_mode: str,
    reference_ids: dict[str, int] | None = None,
) -> CalibrationComputationResult:
    """Compute robust pairwise distances and local coordinates."""
    if len(frames) < 3:
        raise ValueError("Для оценки геометрии нужно минимум 3 хороших кадра.")

    pairwise_distances, pairwise_stats = _robust_pairwise_distances(frames, tag_ids)
    consistent_frames, outlier_frames = _select_consistent_frames(frames, tag_ids, pairwise_distances)
    if len(consistent_frames) < 3:
        consistent_frames = frames

    warnings: list[str] = []
    if outlier_frames:
        warnings.append(f"Отброшены попарные выбросы по кадрам: {outlier_frames}")

    if solver_mode == "torso":
        if reference_ids is None:
            reference_ids = {
                "upper": tag_ids[0],
                "lower": tag_ids[1],
                "plane": tag_ids[2],
            }
        points_local, point_variance, dropped_local_frames = _aggregate_local_coordinates(
            consistent_frames,
            tag_ids,
            reference_ids["upper"],
            reference_ids["lower"],
            reference_ids["plane"],
        )
        if dropped_local_frames:
            warnings.append(
                f"Отброшены кадры с вырожденным torso frame: {dropped_local_frames}"
            )
        dropped_frames = sorted(set(outlier_frames + dropped_local_frames))
        used_frames = len(consistent_frames) - len(dropped_local_frames)
        if used_frames < 3:
            raise ValueError("После фильтрации torso frame осталось слишком мало стабильных кадров.")
    else:
        points_local = _simple_coordinates_from_distances(tag_ids, pairwise_distances)
        point_variance = {
            str(tag_id): [
                float(np.std([frame.markers[tag_id].position_mm[axis] for frame in consistent_frames]))
                for axis in range(3)
            ]
            for tag_id in tag_ids
        }
        dropped_frames = outlier_frames
        used_frames = len(consistent_frames)
        warnings.append(
            "Простой режим восстанавливает форму только по попарным расстояниям; локальные оси условные."
        )

    mean_quality = float(np.mean([frame.quality_score for frame in consistent_frames]))
    mean_pairwise_mad = float(np.mean([value.mad for value in pairwise_stats.values()]))
    overall_score = float(
        np.clip(mean_quality - mean_pairwise_mad * 1.5 + min(20.0, used_frames * 1.5), 0.0, 100.0)
    )
    if used_frames < 5:
        warnings.append("В финальное решение вошло мало кадров; точность может быть ниже.")
    if mean_pairwise_mad > 8.0:
        warnings.append("Обнаружена высокая вариативность между позами; при необходимости переснимите сессию.")

    return CalibrationComputationResult(
        version=1,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        solver_mode=solver_mode,
        aruco_dictionary=aruco_dictionary,
        tag_size_mm=tag_size_mm,
        tag_ids=list(tag_ids),
        camera_calibration_file=camera_calibration_file,
        used_frames=used_frames,
        rejected_frames=rejected_frames_count,
        points_local=points_local,
        pairwise_distances_mm=pairwise_distances,
        pairwise_stats_mm=pairwise_stats,
        point_variance_mm=point_variance,
        quality_overall_score=overall_score,
        quality_notes=warnings,
        dropped_geometry_frames=dropped_frames,
    )
