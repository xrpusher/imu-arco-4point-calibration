"""Utilities for generating and detecting ArUco markers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from app.core.models import GeneratedTagInfo, MarkerPoseObservation, TagGenerationResult
from app.utils.paths import ensure_dir

DEFAULT_TAG_IDS = [10, 20, 30, 40]
DEFAULT_DICTIONARY = "DICT_6X6_250"

LAYOUT_CUSTOM = "custom"
LAYOUT_B21S_50X30 = "b21s_50x30_203dpi"

B21S_LABEL_WIDTH_MM = 50.0
B21S_LABEL_HEIGHT_MM = 30.0
B21S_DPI = 203
B21S_CANVAS_WIDTH_PX = 400
B21S_CANVAS_HEIGHT_PX = 240

ARUCO_DICTIONARY_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_ARUCO_ORIGINAL",
]


def ensure_aruco_available() -> None:
    """Raise a helpful error if cv2.aruco is unavailable."""
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco недоступен. Установите пакет opencv-contrib-python.")


def available_dictionaries() -> list[str]:
    """Return supported dictionary names available in the local OpenCV build."""
    ensure_aruco_available()
    return [name for name in ARUCO_DICTIONARY_NAMES if hasattr(cv2.aruco, name)]


def get_dictionary(dictionary_name: str):
    """Return the OpenCV ArUco dictionary object."""
    ensure_aruco_available()
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Неподдерживаемый словарь ArUco: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    return cv2.aruco.getPredefinedDictionary(dictionary_id)


def parse_tag_ids(raw_value: str) -> list[int]:
    """Parse comma-separated tag IDs and validate the result."""
    parts = [chunk.strip() for chunk in raw_value.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("Укажите хотя бы один ID тега.")
    tag_ids = [int(part) for part in parts]
    if len(set(tag_ids)) != len(tag_ids):
        raise ValueError("ID тегов должны быть уникальными.")
    return tag_ids


def available_layouts() -> list[tuple[str, str]]:
    """Return supported printable layout presets."""
    return [
        (LAYOUT_CUSTOM, "Свободный макет"),
        (LAYOUT_B21S_50X30, "B21S 50x30 мм / 203 dpi"),
    ]


def layout_canvas_size(layout_name: str, fallback_square_px: int = 1000) -> tuple[int, int]:
    """Return the target canvas size for a named layout."""
    if layout_name == LAYOUT_B21S_50X30:
        return B21S_CANVAS_WIDTH_PX, B21S_CANVAS_HEIGHT_PX
    return fallback_square_px, fallback_square_px


def b21s_recommended_tag_size_mm() -> float:
    """Return the practical printed marker size for the B21S label preset."""
    marker_size_px = _b21s_marker_size_px(B21S_CANVAS_WIDTH_PX, B21S_CANVAS_HEIGHT_PX)
    return marker_size_px * 25.4 / B21S_DPI


def _render_marker_image(dictionary_name: str, marker_id: int, marker_size_px: int) -> np.ndarray:
    """Generate a grayscale ArUco marker image."""
    dictionary = get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size_px)
    marker = np.zeros((marker_size_px, marker_size_px), dtype=np.uint8)
    cv2.aruco.drawMarker(dictionary, marker_id, marker_size_px, marker, 1)
    return marker


def _b21s_marker_size_px(canvas_width_px: int, canvas_height_px: int) -> int:
    margin = max(10, canvas_height_px // 20)
    info_band = max(112, canvas_width_px // 3)
    return max(120, min(canvas_height_px - 2 * margin, canvas_width_px - info_band - 3 * margin))


def _generic_canvas(
    marker_image: np.ndarray,
    canvas_size_px: tuple[int, int],
    tag_id: int,
    tag_size_mm: float,
) -> tuple[np.ndarray, int]:
    """Place a marker on a white printable canvas with a caption below."""
    canvas_width_px, canvas_height_px = canvas_size_px
    canvas = np.full((canvas_height_px, canvas_width_px, 3), 255, dtype=np.uint8)
    label_band = max(34, canvas_height_px // 7)
    margin = max(18, min(canvas_width_px, canvas_height_px) // 16)
    inner_size = max(100, min(canvas_width_px - 2 * margin, canvas_height_px - label_band - 2 * margin))
    resized = cv2.resize(marker_image, (inner_size, inner_size), interpolation=cv2.INTER_NEAREST)
    marker_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    x_offset = (canvas_width_px - inner_size) // 2
    y_offset = margin
    canvas[y_offset : y_offset + inner_size, x_offset : x_offset + inner_size] = marker_bgr

    label = f"ID {tag_id} | {tag_size_mm:.1f} mm"
    cv2.putText(
        canvas,
        label,
        (margin, canvas_height_px - max(12, margin // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, min(canvas_width_px, canvas_height_px) / 1100.0),
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas, inner_size


def _b21s_canvas(
    marker_image: np.ndarray,
    canvas_size_px: tuple[int, int],
    tag_id: int,
) -> tuple[np.ndarray, int]:
    """Create a label-optimized layout for a 50x30 mm B21S thermal sticker."""
    canvas_width_px, canvas_height_px = canvas_size_px
    canvas = np.full((canvas_height_px, canvas_width_px, 3), 255, dtype=np.uint8)
    margin = max(10, canvas_height_px // 20)
    marker_size_px = _b21s_marker_size_px(canvas_width_px, canvas_height_px)
    x_offset = margin
    y_offset = (canvas_height_px - marker_size_px) // 2

    resized = cv2.resize(
        marker_image,
        (marker_size_px, marker_size_px),
        interpolation=cv2.INTER_NEAREST,
    )
    marker_bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas[y_offset : y_offset + marker_size_px, x_offset : x_offset + marker_size_px] = marker_bgr

    text_x = x_offset + marker_size_px + margin
    rendered_tag_size_mm = marker_size_px * 25.4 / B21S_DPI
    text_lines = [
        f"ID {tag_id}",
        f"{rendered_tag_size_mm:.1f} mm",
        "50x30",
        "203 dpi",
    ]
    for line_index, text in enumerate(text_lines):
        y = margin + 38 + line_index * 42
        cv2.putText(
            canvas,
            text,
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (0, 0), (canvas_width_px - 1, canvas_height_px - 1), (0, 0, 0), 1)
    return canvas, marker_size_px


def _tag_canvas(
    marker_image: np.ndarray,
    canvas_size_px: tuple[int, int],
    tag_id: int,
    tag_size_mm: float,
    layout_name: str,
) -> tuple[np.ndarray, int]:
    """Place a marker on a printable canvas and return the marker size in pixels."""
    if layout_name == LAYOUT_B21S_50X30:
        return _b21s_canvas(marker_image, canvas_size_px, tag_id)
    return _generic_canvas(marker_image, canvas_size_px, tag_id, tag_size_mm)


def _contact_sheet(
    canvases: list[np.ndarray],
    tag_ids: list[int],
    canvas_size_px: tuple[int, int],
) -> np.ndarray:
    """Build a 2x2 printable contact sheet."""
    canvas_width_px, canvas_height_px = canvas_size_px
    grid_cols = 2
    grid_rows = math.ceil(len(canvases) / grid_cols)
    gap = 28
    title_height = 80
    sheet_h = title_height + grid_rows * canvas_height_px + (grid_rows + 1) * gap
    sheet_w = grid_cols * canvas_width_px + (grid_cols + 1) * gap
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)

    title = f"ArUco sheet | ID: {', '.join(str(tag_id) for tag_id in tag_ids)}"
    cv2.putText(
        sheet,
        title,
        (gap, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )

    for index, canvas in enumerate(canvases):
        row = index // grid_cols
        col = index % grid_cols
        x = gap + col * (canvas_width_px + gap)
        y = title_height + gap + row * (canvas_height_px + gap)
        sheet[y : y + canvas_height_px, x : x + canvas_width_px] = canvas
    return sheet


def generate_tags(
    output_dir: str | Path,
    dictionary_name: str,
    tag_ids: list[int],
    tag_size_mm: float,
    canvas_size_px: tuple[int, int],
    layout_name: str = LAYOUT_CUSTOM,
) -> TagGenerationResult:
    """Generate PNG/JPG tags plus a preview sheet."""
    ensure_aruco_available()
    destination = ensure_dir(output_dir)
    files: list[GeneratedTagInfo] = []
    canvases: list[np.ndarray] = []
    canvas_width_px, canvas_height_px = canvas_size_px

    marker_core_size = max(220, min(canvas_width_px, canvas_height_px) * 2)
    rendered_tag_size_mm = tag_size_mm
    for tag_id in tag_ids:
        marker = _render_marker_image(dictionary_name, tag_id, marker_core_size)
        canvas, marker_size_px = _tag_canvas(
            marker,
            canvas_size_px,
            tag_id,
            tag_size_mm,
            layout_name,
        )
        if layout_name == LAYOUT_B21S_50X30:
            rendered_tag_size_mm = marker_size_px * 25.4 / B21S_DPI
        canvases.append(canvas)

        png_path = destination / f"tag_{tag_id}.png"
        jpg_path = destination / f"tag_{tag_id}.jpg"
        cv2.imwrite(str(png_path), canvas)
        cv2.imwrite(str(jpg_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        files.append(
            GeneratedTagInfo(
                tag_id=tag_id,
                png_path=str(png_path),
                jpg_path=str(jpg_path),
            )
        )

    preview = _contact_sheet(canvases, tag_ids, canvas_size_px)
    preview_path = destination / "contact_sheet.png"
    preview_jpg_path = destination / "contact_sheet.jpg"
    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(preview_jpg_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return TagGenerationResult(
        output_dir=str(destination),
        preview_path=str(preview_path),
        preview_jpg_path=str(preview_jpg_path),
        tag_ids=list(tag_ids),
        dictionary_name=dictionary_name,
        tag_size_mm=tag_size_mm,
        canvas_width_px=canvas_width_px,
        canvas_height_px=canvas_height_px,
        layout_name=layout_name,
        rendered_tag_size_mm=rendered_tag_size_mm,
        files=files,
    )


def create_detector(dictionary_name: str):
    """Create a detector compatible with multiple OpenCV versions."""
    dictionary = get_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_markers(frame: np.ndarray, dictionary_name: str):
    """Detect ArUco markers in a BGR frame."""
    ensure_aruco_available()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = create_detector(dictionary_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            dictionary,
            parameters=parameters,
        )
    ids_list = [] if ids is None else [int(value) for value in ids.flatten().tolist()]
    return corners, ids_list, rejected


def marker_object_points(tag_size_mm: float) -> np.ndarray:
    """Return 3D object points for a square marker centered at the origin."""
    half = tag_size_mm / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_observations(
    corners: list[np.ndarray],
    ids: list[int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    tag_size_mm: float,
) -> dict[int, MarkerPoseObservation]:
    """Estimate pose and simple quality metrics for each marker."""
    observations: dict[int, MarkerPoseObservation] = {}
    object_points = marker_object_points(tag_size_mm)
    solve_pnp_flag = (
        cv2.SOLVEPNP_IPPE_SQUARE
        if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
        else cv2.SOLVEPNP_ITERATIVE
    )

    for tag_id, marker_corners in zip(ids, corners):
        image_points = marker_corners.reshape(-1, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=solve_pnp_flag,
        )
        if not success:
            continue

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs,
        )
        projected_2d = projected.reshape(-1, 2)
        reprojection_error = float(
            np.sqrt(np.mean(np.sum((projected_2d - image_points) ** 2, axis=1)))
        )

        edge_lengths = [
            np.linalg.norm(image_points[i] - image_points[(i + 1) % 4]) for i in range(4)
        ]
        tag_size_px = float(np.mean(edge_lengths))

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        normal = rotation_matrix @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0))))

        observations[tag_id] = MarkerPoseObservation(
            tag_id=tag_id,
            rvec=[float(value) for value in rvec.reshape(-1)],
            tvec=[float(value) for value in tvec.reshape(-1)],
            position_mm=[float(value) for value in tvec.reshape(-1)],
            reprojection_error_px=reprojection_error,
            tag_size_px=tag_size_px,
            angle_deg=angle_deg,
        )

    return observations


def draw_marker_annotations(
    frame: np.ndarray,
    corners: list[np.ndarray],
    ids: list[int],
    observations: dict[int, MarkerPoseObservation] | None = None,
    camera_matrix: np.ndarray | None = None,
    dist_coeffs: np.ndarray | None = None,
    axis_length_mm: float = 30.0,
) -> np.ndarray:
    """Draw detected markers, IDs, and optional pose axes."""
    annotated = frame.copy()
    if ids:
        cv2.aruco.drawDetectedMarkers(
            annotated,
            corners,
            np.asarray(ids, dtype=np.int32).reshape(-1, 1),
        )

    observations = observations or {}
    for tag_id, marker_corners in zip(ids, corners):
        points = marker_corners.reshape(-1, 2)
        center = points.mean(axis=0).astype(int)
        label = f"ID {tag_id}"
        if tag_id in observations:
            observation = observations[tag_id]
            label = (
                f"ID {tag_id} | {observation.tag_size_px:.0f}px | "
                f"{observation.reprojection_error_px:.2f}px"
            )
        cv2.putText(
            annotated,
            label,
            (int(center[0]) - 40, int(center[1]) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 140, 20),
            2,
            cv2.LINE_AA,
        )

        if camera_matrix is not None and dist_coeffs is not None and tag_id in observations:
            observation = observations[tag_id]
            cv2.drawFrameAxes(
                annotated,
                camera_matrix,
                dist_coeffs,
                np.asarray(observation.rvec, dtype=np.float64),
                np.asarray(observation.tvec, dtype=np.float64),
                axis_length_mm,
                2,
            )

    return annotated


def blur_score(frame: np.ndarray) -> float:
    """Estimate image sharpness via Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
