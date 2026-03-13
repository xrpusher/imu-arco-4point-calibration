"""Export helpers for calibration results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.core.models import CalibrationComputationResult
from app.utils.paths import ensure_dir


def write_calibration_json(
    result: CalibrationComputationResult,
    file_path: str | Path,
) -> Path:
    """Write calibration JSON."""
    path = Path(file_path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(result.to_json_dict(), indent=2), encoding="utf-8")
    return path


def write_calibration_report_csv(
    result: CalibrationComputationResult,
    file_path: str | Path,
) -> Path:
    """Write pairwise distance statistics as CSV."""
    path = Path(file_path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["pair", "distance_mm", "mean_mm", "median_mm", "std_mm", "mad_mm", "samples"])
        for pair_key in sorted(result.pairwise_distances_mm.keys()):
            stats = result.pairwise_stats_mm[pair_key]
            writer.writerow(
                [
                    pair_key,
                    f"{result.pairwise_distances_mm[pair_key]:.3f}",
                    f"{stats.mean:.3f}",
                    f"{stats.median:.3f}",
                    f"{stats.std:.3f}",
                    f"{stats.mad:.3f}",
                    stats.samples,
                ]
            )
    return path


def write_calibration_report_txt(
    result: CalibrationComputationResult,
    file_path: str | Path,
) -> Path:
    """Write a human-readable calibration summary."""
    path = Path(file_path)
    ensure_dir(path.parent)
    lines: list[str] = [
        "Отчёт по калибровке IMU ArUco",
        "=" * 32,
        f"Время: {result.timestamp}",
        f"Режим решателя: {result.solver_mode}",
        f"Словарь ArUco: {result.aruco_dictionary}",
        f"Размер тега (мм): {result.tag_size_mm:.2f}",
        f"ID тегов: {', '.join(str(tag_id) for tag_id in result.tag_ids)}",
        f"Использовано кадров: {result.used_frames}",
        f"Отброшено кадров: {result.rejected_frames}",
        f"Итоговая оценка качества: {result.quality_overall_score:.2f}",
        "",
        "Локальные координаты (мм):",
    ]
    for tag_id in result.tag_ids:
        coordinates = result.points_local[str(tag_id)]
        variance = result.point_variance_mm.get(str(tag_id), [0.0, 0.0, 0.0])
        lines.append(
            f"  {tag_id}: x={coordinates[0]:.3f}, y={coordinates[1]:.3f}, z={coordinates[2]:.3f} "
            f"| std=({variance[0]:.3f}, {variance[1]:.3f}, {variance[2]:.3f})"
        )
    lines.append("")
    lines.append("Попарные расстояния (мм):")
    for pair_key in sorted(result.pairwise_distances_mm.keys()):
        stats = result.pairwise_stats_mm[pair_key]
        lines.append(
            f"  {pair_key}: {result.pairwise_distances_mm[pair_key]:.3f} "
            f"(mean={stats.mean:.3f}, median={stats.median:.3f}, std={stats.std:.3f}, "
            f"mad={stats.mad:.3f}, n={stats.samples})"
        )
    lines.append("")
    lines.append("Примечания по качеству:")
    if result.quality_notes:
        for note in result.quality_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("  - Предупреждений нет.")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def export_all_reports(
    result: CalibrationComputationResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write the full export bundle to the target directory."""
    directory = ensure_dir(output_dir)
    paths = {
        "json": write_calibration_json(result, directory / "calibration.json"),
        "csv": write_calibration_report_csv(result, directory / "calibration_report.csv"),
        "txt": write_calibration_report_txt(result, directory / "calibration_report.txt"),
    }
    return paths
