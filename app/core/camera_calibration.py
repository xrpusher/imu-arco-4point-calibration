"""Camera calibration helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.core.models import CameraCalibrationData
from app.utils.paths import ensure_dir


def approximate_calibration(
    image_size: tuple[int, int],
    horizontal_fov_deg: float = 69.0,
) -> CameraCalibrationData:
    """Build a quick approximate camera model from image size and assumed FOV."""
    width, height = image_size
    f_x = (width / 2.0) / np.tan(np.deg2rad(horizontal_fov_deg / 2.0))
    f_y = f_x
    c_x = width / 2.0
    c_y = height / 2.0

    return CameraCalibrationData(
        mode="quick_approximation",
        image_width=width,
        image_height=height,
        camera_matrix=[
            [float(f_x), 0.0, float(c_x)],
            [0.0, float(f_y), float(c_y)],
            [0.0, 0.0, 1.0],
        ],
        dist_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
        notes=[
            "Быстрое приближение по размеру изображения и предполагаемому горизонтальному FOV.",
            "Этот режим менее точен, чем калибровка по шахматной доске.",
        ],
    )


def rescale_calibration(
    calibration: CameraCalibrationData,
    image_size: tuple[int, int],
) -> CameraCalibrationData:
    """Scale intrinsics when the live frame resolution differs from the calibration file."""
    width, height = image_size
    if calibration.image_width == width and calibration.image_height == height:
        return calibration

    scale_x = width / calibration.image_width
    scale_y = height / calibration.image_height
    camera_matrix = np.asarray(calibration.camera_matrix, dtype=np.float64).copy()
    camera_matrix[0, 0] *= scale_x
    camera_matrix[1, 1] *= scale_y
    camera_matrix[0, 2] *= scale_x
    camera_matrix[1, 2] *= scale_y

    return CameraCalibrationData(
        mode=calibration.mode,
        image_width=width,
        image_height=height,
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=list(calibration.dist_coeffs),
        reprojection_error=calibration.reprojection_error,
        chessboard_size=calibration.chessboard_size,
        square_size_mm=calibration.square_size_mm,
        notes=list(calibration.notes)
        + [f"Rescaled from {calibration.image_width}x{calibration.image_height}."],
    )


def find_chessboard_corners(
    frame: np.ndarray,
    board_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    """Find and draw chessboard corners."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    if hasattr(cv2, "findChessboardCornersSB"):
        try:
            normalized = cv2.equalizeHist(gray)
            found, corners = cv2.findChessboardCornersSB(normalized, board_size, flags=0)
        except cv2.error:
            found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
            if found:
                term_criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    30,
                    0.001,
                )
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    term_criteria,
                )
    else:
        found, corners = cv2.findChessboardCorners(gray, board_size, flags=flags)
        if found:
            term_criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                term_criteria,
            )

    preview = frame.copy()
    if found and corners is not None:
        cv2.drawChessboardCorners(preview, board_size, corners, found)
    return found, corners, preview


def calibrate_from_chessboard_samples(
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
    board_size: tuple[int, int],
    square_size_mm: float,
) -> CameraCalibrationData:
    """Run camera calibration from captured chessboard frames."""
    if len(image_points) < 4:
        raise ValueError("Нужно минимум 4 корректных кадра шахматной доски.")

    obj_points_template = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    grid = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    obj_points_template[:, :2] = grid * square_size_mm

    object_points = [obj_points_template.copy() for _ in image_points]
    calibration_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )

    total_error = 0.0
    for object_points_frame, image_points_frame, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
    ):
        projected, _ = cv2.projectPoints(
            object_points_frame,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        error = cv2.norm(image_points_frame, projected, cv2.NORM_L2) / len(projected)
        total_error += float(error)

    mean_error = total_error / len(object_points)

    return CameraCalibrationData(
        mode="chessboard",
        image_width=image_size[0],
        image_height=image_size[1],
        camera_matrix=camera_matrix.tolist(),
        dist_coeffs=dist_coeffs.reshape(-1).tolist(),
        reprojection_error=float(mean_error if mean_error else calibration_error),
        chessboard_size=board_size,
        square_size_mm=square_size_mm,
        notes=[
            "Калибровка по шахматной доске.",
            "Рекомендуемый режим для более точной оценки позы ArUco.",
        ],
    )


def save_camera_calibration(
    calibration: CameraCalibrationData,
    file_path: str | Path,
) -> Path:
    """Persist camera calibration to JSON."""
    path = Path(file_path)
    ensure_dir(path.parent)
    payload = {
        "version": 1,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "calibration": calibration.to_dict(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_camera_calibration(file_path: str | Path) -> CameraCalibrationData:
    """Load camera calibration JSON from disk."""
    payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return CameraCalibrationData.from_dict(payload["calibration"])
