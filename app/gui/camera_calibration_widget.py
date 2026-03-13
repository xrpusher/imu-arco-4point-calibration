"""Camera calibration tab."""

from __future__ import annotations

from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.camera_calibration import (
    approximate_calibration,
    calibrate_from_chessboard_samples,
    find_chessboard_corners,
    load_camera_calibration,
    save_camera_calibration,
)
from app.core.camera_manager import (
    COMMON_RESOLUTIONS,
    CameraManager,
    enumerate_camera_indices,
    parse_resolution_label,
    resolution_label,
)
from app.utils.logging_utils import exception_to_text
from app.utils.paths import ensure_dir


def _frame_to_pixmap(frame) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(
        rgb.data,
        width,
        height,
        channels * width,
        QImage.Format.Format_RGB888,
    )
    return QPixmap.fromImage(image.copy())


class CameraCalibrationWidget(QWidget):
    """Widget with quick and chessboard-based camera calibration workflows."""

    log_message = pyqtSignal(str)
    calibration_ready = pyqtSignal(object, str)

    def __init__(self, default_output_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default_output_dir = ensure_dir(default_output_dir)
        self.camera = CameraManager()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self.latest_corners = None
        self.latest_frame_size: tuple[int, int] | None = None
        self.captured_corners: list = []
        self.preview_pixmap: QPixmap | None = None

        self._build_ui()
        self._refresh_cameras()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel(
            "Рекомендуемый режим: калибровка по шахматной доске. Быстрый режим использует приближённую "
            "модель камеры и подходит только для чернового теста."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        troubleshooting = QLabel(
            "Если вместо картинки виден замок или серый экран, проверьте разрешения Windows на камеру, "
            "физическую шторку ноутбука и не занята ли камера другой программой."
        )
        troubleshooting.setWordWrap(True)
        layout.addWidget(troubleshooting)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        layout.addLayout(content_layout, stretch=1)

        left_panel = QWidget()
        left_panel.setMinimumWidth(620)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        top_group = QGroupBox("Камера предпросмотра")
        top_layout = QFormLayout(top_group)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)
        self.resolution_combo = QComboBox()
        for resolution in COMMON_RESOLUTIONS:
            self.resolution_combo.addItem(resolution_label(resolution))
        default_resolution = self.resolution_combo.findText("1280x720")
        if default_resolution >= 0:
            self.resolution_combo.setCurrentIndex(default_resolution)

        top_layout.addRow("Индекс камеры", self.camera_combo)
        top_layout.addRow("Разрешение", self.resolution_combo)

        preview_button_row = QHBoxLayout()
        refresh_button = QPushButton("Обновить список камер")
        refresh_button.clicked.connect(self._refresh_cameras)
        start_button = QPushButton("Старт предпросмотра")
        start_button.clicked.connect(self._start_preview)
        stop_button = QPushButton("Стоп предпросмотра")
        stop_button.clicked.connect(self._stop_preview)
        for button in (refresh_button, start_button, stop_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            preview_button_row.addWidget(button)
        preview_button_widget = QWidget()
        preview_button_widget.setLayout(preview_button_row)
        top_layout.addRow(preview_button_widget)
        left_layout.addWidget(top_group)

        quick_group = QGroupBox("Быстрая калибровка")
        quick_layout = QFormLayout(quick_group)
        self.hfov_spin = QDoubleSpinBox()
        self.hfov_spin.setRange(30.0, 120.0)
        self.hfov_spin.setValue(69.0)
        self.hfov_spin.setDecimals(1)
        self.hfov_spin.setSuffix(" град")
        quick_button = QPushButton("Создать быструю калибровку")
        quick_button.clicked.connect(self._create_quick_calibration)
        quick_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        quick_layout.addRow("Предполагаемый горизонтальный FOV", self.hfov_spin)
        quick_layout.addRow(quick_button)
        left_layout.addWidget(quick_group)

        chessboard_group = QGroupBox("Калибровка по шахматной доске")
        chessboard_layout = QFormLayout(chessboard_group)
        self.board_cols_spin = QSpinBox()
        self.board_cols_spin.setRange(3, 20)
        self.board_cols_spin.setValue(9)
        self.board_rows_spin = QSpinBox()
        self.board_rows_spin.setRange(3, 20)
        self.board_rows_spin.setValue(6)
        self.square_size_spin = QDoubleSpinBox()
        self.square_size_spin.setRange(1.0, 100.0)
        self.square_size_spin.setValue(25.0)
        self.square_size_spin.setDecimals(2)
        self.square_size_spin.setSuffix(" мм")

        chessboard_layout.addRow("Внутренние углы (столбцы)", self.board_cols_spin)
        chessboard_layout.addRow("Внутренние углы (строки)", self.board_rows_spin)
        chessboard_layout.addRow("Размер клетки", self.square_size_spin)

        capture_row = QHBoxLayout()
        capture_button = QPushButton("Сохранить кадр")
        capture_button.clicked.connect(self._capture_frame)
        clear_button = QPushButton("Очистить кадры")
        clear_button.clicked.connect(self._clear_captures)
        calibrate_button = QPushButton("Калибровать и сохранить")
        calibrate_button.clicked.connect(self._run_chessboard_calibration)
        load_button = QPushButton("Загрузить готовый файл")
        load_button.clicked.connect(self._load_existing_calibration)
        for button in (capture_button, clear_button, calibrate_button, load_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            capture_row.addWidget(button)
        capture_widget = QWidget()
        capture_widget.setLayout(capture_row)
        chessboard_layout.addRow(capture_widget)
        left_layout.addWidget(chessboard_group)

        self.captures_label = QLabel("Сохранено кадров: 0")
        self.status_label = QLabel("Готово.")
        self.status_label.setWordWrap(True)
        left_layout.addWidget(self.captures_label)
        left_layout.addWidget(self.status_label)
        left_layout.addStretch(1)

        right_panel = QWidget()
        right_panel.setMaximumWidth(780)
        right_layout = QVBoxLayout(right_panel)
        preview_group = QGroupBox("Предпросмотр камеры")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Предпросмотр камеры")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(430, 310)
        self.preview_label.setMaximumHeight(820)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background: #1f1f1f;")
        preview_layout.addWidget(self.preview_label)
        right_layout.addWidget(preview_group, stretch=1)

        tips = QLabel(
            "Совет: держите шахматную доску полностью в кадре, снимайте 8-12 разных ракурсов, "
            "не допускайте смаза и бликов."
        )
        tips.setWordWrap(True)
        right_layout.addWidget(tips)

        content_layout.addWidget(left_panel, stretch=0)
        content_layout.addWidget(right_panel, stretch=1)

    def _refresh_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = enumerate_camera_indices()
        if not cameras:
            self.camera_combo.addItem("0")
            self.status_label.setText("Не удалось получить список камер. Попробуйте индекс 0 вручную.")
            self.log_message.emit("Во вкладке калибровки не найден список камер.")
            return
        for index in cameras:
            self.camera_combo.addItem(str(index))
        self.status_label.setText(
            f"Доступные индексы: {', '.join(str(index) for index in cameras)}. Обычно сначала стоит попробовать 0."
        )

    def _start_preview(self) -> None:
        try:
            index = int(self.camera_combo.currentText())
            resolution = parse_resolution_label(self.resolution_combo.currentText())
            self.status_label.setText("Открываю камеру...")
            QApplication.processEvents()
            actual = self.camera.open(index, resolution)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка камеры", exception_to_text(exc))
            self.log_message.emit(f"Не удалось запустить предпросмотр: {exception_to_text(exc)}")
            self.status_label.setText("Не удалось открыть камеру.")
            return
        self.timer.start(33)
        self.status_label.setText(f"Предпросмотр запущен: {actual[0]}x{actual[1]}.")
        self.log_message.emit(f"Предпросмотр калибровки запущен на камере {index}: {actual[0]}x{actual[1]}")

    def _stop_preview(self) -> None:
        self.timer.stop()
        self.camera.stop()
        self.preview_label.setText("Предпросмотр камеры")

    def _current_board_size(self) -> tuple[int, int]:
        return self.board_cols_spin.value(), self.board_rows_spin.value()

    def _update_frame(self) -> None:
        try:
            frame = self.camera.read_frame()
        except Exception as exc:
            self._stop_preview()
            self.status_label.setText(exception_to_text(exc))
            self.log_message.emit(f"Предпросмотр остановлен: {exception_to_text(exc)}")
            return

        try:
            found, corners, preview = find_chessboard_corners(frame, self._current_board_size())
        except cv2.error as exc:
            self.latest_corners = None
            self.latest_frame_size = (frame.shape[1], frame.shape[0])
            self.status_label.setText(f"Ошибка поиска шахматной доски: {exc}")
            preview = frame
            found = False
            corners = None
        self.latest_corners = corners.copy() if found and corners is not None else None
        self.latest_frame_size = (frame.shape[1], frame.shape[0])
        self.status_label.setText(
            "Шахматная доска найдена. Если всё резко и целиком видно, можно сохранять кадр."
            if found
            else "Предпросмотр идёт. Покажите шахматную доску целиком."
        )
        pixmap = _frame_to_pixmap(preview)
        self.preview_pixmap = pixmap
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _capture_frame(self) -> None:
        if self.latest_corners is None:
            QMessageBox.warning(self, "Шахматная доска не найдена", "В текущем кадре не найдены корректные углы доски.")
            return
        self.captured_corners.append(self.latest_corners.copy())
        self.captures_label.setText(f"Сохранено кадров: {len(self.captured_corners)}")
        self.log_message.emit(f"Сохранён кадр шахматной доски #{len(self.captured_corners)}")

    def _clear_captures(self) -> None:
        self.captured_corners.clear()
        self.captures_label.setText("Сохранено кадров: 0")
        self.status_label.setText("Список кадров шахматной доски очищен.")

    def _create_quick_calibration(self) -> None:
        try:
            image_size = self.latest_frame_size or parse_resolution_label(self.resolution_combo.currentText())
            calibration = approximate_calibration(image_size, self.hfov_spin.value())
            save_path = self._choose_save_path("camera_calibration_quick.json")
            if not save_path:
                return
            save_camera_calibration(calibration, save_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка быстрой калибровки", exception_to_text(exc))
            self.log_message.emit(f"Ошибка быстрой калибровки: {exception_to_text(exc)}")
            return

        self.status_label.setText(
            f"Быстрая калибровка сохранена: {save_path}. Этот режим менее точен, чем шахматная доска."
        )
        self.calibration_ready.emit(calibration, str(save_path))
        self.log_message.emit(f"Быстрая калибровка камеры сохранена: {save_path}")

    def _run_chessboard_calibration(self) -> None:
        if self.latest_frame_size is None:
            QMessageBox.warning(self, "Нет предпросмотра", "Сначала запустите камеру.")
            return
        if len(self.captured_corners) < 8:
            QMessageBox.warning(
                self,
                "Нужно больше кадров",
                "Для рекомендуемой калибровки сохраните минимум 8 хороших кадров шахматной доски.",
            )
            return

        try:
            calibration = calibrate_from_chessboard_samples(
                image_points=self.captured_corners,
                image_size=self.latest_frame_size,
                board_size=self._current_board_size(),
                square_size_mm=self.square_size_spin.value(),
            )
            save_path = self._choose_save_path("camera_calibration.json")
            if not save_path:
                return
            save_camera_calibration(calibration, save_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка калибровки", exception_to_text(exc))
            self.log_message.emit(f"Ошибка калибровки по шахматной доске: {exception_to_text(exc)}")
            return

        reprojection = calibration.reprojection_error if calibration.reprojection_error is not None else 0.0
        self.status_label.setText(
            f"Калибровка сохранена: {save_path}. Средняя reprojection error: {reprojection:.3f}px"
        )
        self.calibration_ready.emit(calibration, str(save_path))
        self.log_message.emit(f"Калибровка по шахматной доске сохранена: {save_path}")

    def _load_existing_calibration(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Загрузить файл калибровки камеры",
            str(self.default_output_dir),
            "JSON files (*.json)",
        )
        if not file_path:
            return
        try:
            calibration = load_camera_calibration(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка загрузки", exception_to_text(exc))
            self.log_message.emit(f"Не удалось загрузить файл калибровки: {exception_to_text(exc)}")
            return

        self.status_label.setText(f"Загружена калибровка: {file_path}")
        self.calibration_ready.emit(calibration, file_path)
        self.log_message.emit(f"Загружена калибровка камеры: {file_path}")

    def _choose_save_path(self, suggested_name: str) -> str:
        default_path = str(self.default_output_dir / suggested_name)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить калибровку камеры",
            default_path,
            "JSON files (*.json)",
        )
        return file_path

    def shutdown(self) -> None:
        self._stop_preview()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if self.preview_pixmap is not None and not self.preview_pixmap.isNull():
            self.preview_label.setPixmap(
                self.preview_pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
