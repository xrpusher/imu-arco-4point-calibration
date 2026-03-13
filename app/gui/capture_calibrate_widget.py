"""Live capture and calibration session tab."""

from __future__ import annotations

import time
from itertools import combinations
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app.core.aruco_utils import (
    DEFAULT_DICTIONARY,
    available_dictionaries,
    b21s_recommended_tag_size_mm,
    blur_score,
    detect_markers,
    draw_marker_annotations,
    estimate_marker_observations,
    parse_tag_ids,
)
from app.core.camera_calibration import (
    approximate_calibration,
    rescale_calibration,
    save_camera_calibration,
)
from app.core.camera_manager import (
    COMMON_RESOLUTIONS,
    CameraManager,
    enumerate_camera_indices,
    parse_resolution_label,
    resolution_label,
)
from app.core.export_utils import export_all_reports
from app.core.geometry_solver import solve_geometry
from app.core.models import CameraCalibrationData, MarkerPoseObservation
from app.core.session_recorder import SessionRecorder, SessionRecorderConfig
from app.utils.logging_utils import exception_to_text
from app.utils.paths import ensure_dir, timestamped_output_dir


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


class CaptureCalibrateWidget(QWidget):
    """Live camera tab used to record calibration observations."""

    log_message = pyqtSignal(str)
    results_ready = pyqtSignal(object, str)

    def __init__(
        self,
        calibration_provider: Callable[[], tuple[CameraCalibrationData | None, str]],
        default_output_root: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.calibration_provider = calibration_provider
        self.default_output_root = ensure_dir(default_output_root)

        self.camera = CameraManager()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)

        self.frame_index = 0
        self.latest_frame_size: tuple[int, int] | None = None
        self.recorder: SessionRecorder | None = None
        self.session_active = False
        self.last_eval_time = 0.0
        self.recent_weak_frames = 0
        self.current_output_dir: Path | None = None
        self.preview_pixmap: QPixmap | None = None

        self._build_ui()
        self._refresh_cameras()
        self._update_reference_combos_from_text()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel(
            "Эта вкладка показывает живое видео, детектирует 4 ArUco-тега и собирает хорошие кадры "
            "для финальной оценки расстояний и геометрии между 4 датчиками."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        troubleshooting = QLabel(
            "Если вместо видео виден замок или серый экран, проверьте разрешения Windows на камеру, "
            "физическую шторку ноутбука и не занята ли камера другой программой."
        )
        troubleshooting.setWordWrap(True)
        layout.addWidget(troubleshooting)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(14)
        layout.addLayout(content_layout, stretch=1)

        left_panel = QWidget()
        left_panel.setMinimumWidth(660)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(12)

        config_group = QGroupBox("Камера и детект")
        config_layout = QFormLayout(config_group)

        self.camera_combo = QComboBox()
        self.camera_combo.setEditable(True)

        self.resolution_combo = QComboBox()
        for resolution in COMMON_RESOLUTIONS:
            self.resolution_combo.addItem(resolution_label(resolution))
        default_resolution = self.resolution_combo.findText("1280x720")
        if default_resolution >= 0:
            self.resolution_combo.setCurrentIndex(default_resolution)

        self.dictionary_combo = QComboBox()
        self.aruco_error: str | None = None
        try:
            dictionaries = available_dictionaries()
        except Exception as exc:
            dictionaries = [DEFAULT_DICTIONARY]
            self.aruco_error = exception_to_text(exc)
        self.dictionary_combo.addItems(dictionaries)
        default_dictionary = self.dictionary_combo.findText(DEFAULT_DICTIONARY)
        if default_dictionary >= 0:
            self.dictionary_combo.setCurrentIndex(default_dictionary)

        self.tag_ids_edit = QLineEdit("10, 20, 30, 40")
        self.tag_ids_edit.editingFinished.connect(self._update_reference_combos_from_text)

        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(10.0, 300.0)
        self.tag_size_spin.setValue(round(b21s_recommended_tag_size_mm(), 1))
        self.tag_size_spin.setDecimals(1)
        self.tag_size_spin.setSuffix(" мм")

        config_layout.addRow("Индекс камеры", self.camera_combo)
        config_layout.addRow("Разрешение", self.resolution_combo)
        config_layout.addRow("Словарь ArUco", self.dictionary_combo)
        config_layout.addRow("Размер тега", self.tag_size_spin)
        config_layout.addRow("Нужные ID тегов", self.tag_ids_edit)

        control_row = QHBoxLayout()
        refresh_button = QPushButton("Обновить камеры")
        refresh_button.clicked.connect(self._refresh_cameras)
        self.start_camera_button = QPushButton("Старт камеры")
        self.start_camera_button.clicked.connect(self._start_camera)
        self.stop_camera_button = QPushButton("Стоп камеры")
        self.stop_camera_button.clicked.connect(self._stop_camera)
        for button in (refresh_button, self.start_camera_button, self.stop_camera_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            control_row.addWidget(button)
        control_widget = QWidget()
        control_widget.setLayout(control_row)
        config_layout.addRow(control_widget)
        left_layout.addWidget(config_group)

        b21s_hint = QLabel(
            "По умолчанию уже выставлен размер под B21S 50x30 мм / 203 dpi: примерно 27.0 мм. "
            "Если используешь другой макет, просто измени это поле вручную."
        )
        b21s_hint.setWordWrap(True)
        left_layout.addWidget(b21s_hint)

        solver_group = QGroupBox("Сессия и геометрия")
        solver_layout = QFormLayout(solver_group)
        self.solver_combo = QComboBox()
        self.solver_combo.addItem("Расстояния между 4 датчиками (рекомендуется для рук в стороны)", "simple")
        self.solver_combo.addItem("Локальная система по 3 опорным тегам", "torso")

        self.upper_combo = QComboBox()
        self.lower_combo = QComboBox()
        self.plane_combo = QComboBox()

        output_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(str(self.default_output_root))
        browse_output_button = QPushButton("Выбрать папку")
        browse_output_button.clicked.connect(self._browse_output_root)
        browse_output_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        output_row.addWidget(self.output_dir_edit)
        output_row.addWidget(browse_output_button)
        output_widget = QWidget()
        output_widget.setLayout(output_row)

        solver_layout.addRow("Режим расчёта", self.solver_combo)
        solver_layout.addRow("Опорный тег A (начало)", self.upper_combo)
        solver_layout.addRow("Опорный тег B (направление оси)", self.lower_combo)
        solver_layout.addRow("Опорный тег C (плоскость)", self.plane_combo)
        solver_layout.addRow("Корневая папка результатов", output_widget)
        left_layout.addWidget(solver_group)

        frame_hint = QLabel(
            "Если теги расположены на разведённых руках и тебе важны именно расстояния между 4 датчиками, "
            "используй первый режим. Локальная система нужна только если тебе реально нужны локальные оси и координаты."
        )
        frame_hint.setWordWrap(True)
        left_layout.addWidget(frame_hint)

        session_group = QGroupBox("Управление сессией")
        session_layout = QVBoxLayout(session_group)
        session_row = QHBoxLayout()
        self.start_session_button = QPushButton("Начать сессию")
        self.start_session_button.clicked.connect(self._start_session)
        self.finish_session_button = QPushButton("Завершить сессию")
        self.finish_session_button.clicked.connect(self._finish_session)
        for button in (self.start_session_button, self.finish_session_button):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            session_row.addWidget(button)
        session_layout.addLayout(session_row)

        self.good_frames_label = QLabel("Хороших кадров собрано: 0")
        self.rejected_frames_label = QLabel("Отклонено кандидатов: 0")
        self.last_decision_label = QLabel("Последняя оценка кадра: ожидание")
        self.last_decision_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        session_layout.addWidget(self.good_frames_label)
        session_layout.addWidget(self.rejected_frames_label)
        session_layout.addWidget(self.last_decision_label)
        session_layout.addWidget(self.progress_bar)
        left_layout.addWidget(session_group)

        summary_group = QGroupBox("Сводка сессии")
        summary_layout = QVBoxLayout(summary_group)
        self.session_summary = QPlainTextEdit()
        self.session_summary.setReadOnly(True)
        self.session_summary.setMinimumHeight(180)
        self.session_summary.setPlaceholderText("Здесь появится краткая сводка по захвату и экспорту результатов.")
        summary_layout.addWidget(self.session_summary)
        left_layout.addWidget(summary_group, stretch=1)

        right_panel = QWidget()
        right_panel.setMaximumWidth(780)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(12)

        preview_group = QGroupBox("Живое превью")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("Превью камеры")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(430, 310)
        self.preview_label.setMaximumHeight(820)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background: #1f1f1f;")
        preview_layout.addWidget(self.preview_label)
        right_layout.addWidget(preview_group, stretch=1)

        phone_test_label = QLabel(
            "Если тестируете с телефона: откройте отдельный PNG-тег или сильно приблизьте изображение. "
            "Показывать весь лист 2x2 на экране телефона часто слишком мелко: камера может увидеть только 1-2 тега."
        )
        phone_test_label.setWordWrap(True)
        right_layout.addWidget(phone_test_label)

        status_group = QGroupBox("Текущее состояние")
        status_layout = QGridLayout(status_group)
        self.found_status_label = QLabel("Найдено 0/4")
        self.found_ids_label = QLabel("ID в кадре: -")
        self.calibration_label = QLabel("Калибровка камеры: временное быстрое приближение")
        self.tag_size_label = QLabel("Примерный размер тега: -")
        self.blur_label = QLabel("Оценка резкости: -")
        self.angle_label = QLabel("Предупреждение по углу: -")
        self.guidance_label = QLabel(
            "Повернитесь немного влево/вправо. Держите все 4 тега в кадре несколько секунд."
        )
        self.guidance_label.setWordWrap(True)

        status_layout.addWidget(self.found_status_label, 0, 0)
        status_layout.addWidget(self.found_ids_label, 0, 1)
        status_layout.addWidget(self.calibration_label, 1, 0, 1, 2)
        status_layout.addWidget(self.tag_size_label, 2, 0)
        status_layout.addWidget(self.blur_label, 2, 1)
        status_layout.addWidget(self.angle_label, 3, 0)
        status_layout.addWidget(self.guidance_label, 3, 1)
        right_layout.addWidget(status_group)

        live_distances_group = QGroupBox("Текущие попарные расстояния")
        live_distances_layout = QVBoxLayout(live_distances_group)
        self.live_distances_box = QPlainTextEdit()
        self.live_distances_box.setReadOnly(True)
        self.live_distances_box.setMinimumHeight(120)
        self.live_distances_box.setPlaceholderText(
            "Когда в кадре видны нужные теги, здесь появятся текущие расстояния между ними."
        )
        live_distances_layout.addWidget(self.live_distances_box)
        right_layout.addWidget(live_distances_group)

        content_layout.addWidget(left_panel, stretch=0)
        content_layout.addWidget(right_panel, stretch=1)

        if self.aruco_error:
            self.start_camera_button.setEnabled(False)
            self.start_session_button.setEnabled(False)
            self.guidance_label.setText(
                "ArUco-детект недоступен. Установите opencv-contrib-python, чтобы включить эту вкладку."
            )
            self.session_summary.setPlainText(f"Не хватает зависимости: {self.aruco_error}")

    def _refresh_cameras(self) -> None:
        self.camera_combo.clear()
        cameras = enumerate_camera_indices()
        if not cameras:
            self.camera_combo.addItem("0")
            self.session_summary.setPlainText(
                "Не удалось автоматически получить список камер. Попробуйте индекс 0, затем 1."
            )
            self.log_message.emit("Во вкладке захвата не найден список камер.")
            return
        for index in cameras:
            self.camera_combo.addItem(str(index))
        self.session_summary.setPlainText(
            "Камеры обновлены. Обычно сначала стоит попробовать индекс 0. "
            "Если видно замок, проблема обычно не в коде, а в настройках Windows или шторке камеры."
        )

    def _browse_output_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку результатов",
            self.output_dir_edit.text(),
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _update_reference_combos_from_text(self) -> None:
        try:
            tag_ids = parse_tag_ids(self.tag_ids_edit.text())
        except Exception:
            tag_ids = [10, 20, 30, 40]

        combos = [self.upper_combo, self.lower_combo, self.plane_combo]
        previous = [combo.currentText() for combo in combos]
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            for tag_id in tag_ids:
                combo.addItem(str(tag_id))
            combo.blockSignals(False)

        defaults = [0, 1, 2]
        for combo, previous_text, default_index in zip(combos, previous, defaults):
            restored = combo.findText(previous_text)
            if restored >= 0:
                combo.setCurrentIndex(restored)
            elif combo.count() > default_index:
                combo.setCurrentIndex(default_index)

    def _start_camera(self) -> None:
        try:
            index = int(self.camera_combo.currentText())
            resolution = parse_resolution_label(self.resolution_combo.currentText())
            self.session_summary.setPlainText("Открываю камеру...")
            QApplication.processEvents()
            actual = self.camera.open(index, resolution)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка камеры", exception_to_text(exc))
            self.log_message.emit(f"Не удалось запустить камеру: {exception_to_text(exc)}")
            self.session_summary.setPlainText("Не удалось открыть камеру. Попробуйте другой индекс.")
            return

        self.frame_index = 0
        self.timer.start(33)
        self.session_summary.setPlainText(
            "Камера запущена.\n"
            "- Повернитесь немного влево/вправо.\n"
            "- Держите теги крупными и резкими.\n"
            "- Для телефона лучше открыть отдельный PNG-тег крупно.\n"
            "- Для полноценной калибровки лучше использовать распечатанные PNG."
        )
        self.log_message.emit(f"Камера захвата запущена: {actual[0]}x{actual[1]}")

    def _stop_camera(self) -> None:
        self.timer.stop()
        self.camera.stop()
        self.preview_label.setText("Превью камеры")

    def _get_active_calibration(
        self,
        image_size: tuple[int, int],
    ) -> tuple[CameraCalibrationData, str]:
        calibration, calibration_file = self.calibration_provider()
        if calibration is None:
            temporary = approximate_calibration(image_size)
            return temporary, "Быстрое приближение (временно)"
        return rescale_calibration(calibration, image_size), calibration_file or calibration.mode

    def _start_session(self) -> None:
        if not self.camera.is_open():
            QMessageBox.warning(self, "Нужна камера", "Сначала запустите камеру, затем начинайте сессию.")
            return
        try:
            tag_ids = parse_tag_ids(self.tag_ids_edit.text())
            if len(tag_ids) != 4:
                raise ValueError("В этом MVP нужно указать ровно 4 ID тега.")
        except Exception as exc:
            QMessageBox.warning(self, "Некорректные ID тегов", exception_to_text(exc))
            return

        self._update_reference_combos_from_text()
        self.recorder = SessionRecorder(SessionRecorderConfig(required_ids=tag_ids))
        self.recorder.start()
        self.session_active = True
        self.last_eval_time = 0.0
        self.progress_bar.setValue(0)
        self.good_frames_label.setText("Хороших кадров собрано: 0")
        self.rejected_frames_label.setText("Отклонено кандидатов: 0")
        self.last_decision_label.setText("Последняя оценка кадра: сессия запущена, ждём хороший кадр")
        self.session_summary.setPlainText(
            "Сессия началась.\n"
            "Цель: собрать хотя бы 3, а лучше 5-8 разных хороших кадров.\n"
            "Если нужны только расстояния, оставь режим по умолчанию: без осей frame."
        )
        self.log_message.emit("Сессия калибровки начата.")

    def _finish_session(self) -> None:
        if not self.session_active or self.recorder is None:
            QMessageBox.warning(self, "Нет активной сессии", "Сначала начните сессию.")
            return
        if len(self.recorder.accepted_frames) < 3:
            QMessageBox.warning(
                self,
                "Слишком мало кадров",
                "Перед завершением нужно собрать минимум 3 хороших кадра.",
            )
            return
        if self.latest_frame_size is None:
            QMessageBox.warning(self, "Нет данных кадра", "Нет актуального изображения с камеры.")
            return

        tag_ids = parse_tag_ids(self.tag_ids_edit.text())
        solver_mode = str(self.solver_combo.currentData())
        notes: list[str] = []
        reference_ids = {
            "upper": int(self.upper_combo.currentText()),
            "lower": int(self.lower_combo.currentText()),
            "plane": int(self.plane_combo.currentText()),
        }
        if solver_mode == "torso" and len(set(reference_ids.values())) < 3:
            solver_mode = "simple"
            notes.append("Локальная система отключена: опорные теги совпадают, использован режим расстояний.")

        output_root = ensure_dir(self.output_dir_edit.text())
        output_dir = timestamped_output_dir(output_root, "calibration_session")
        calibration, _ = self._get_active_calibration(self.latest_frame_size)
        calibration_file = save_camera_calibration(
            calibration,
            output_dir / "camera_calibration_used.json",
        )

        try:
            result = solve_geometry(
                frames=self.recorder.accepted_frames,
                tag_ids=tag_ids,
                aruco_dictionary=self.dictionary_combo.currentText(),
                tag_size_mm=self.tag_size_spin.value(),
                camera_calibration_file=str(calibration_file),
                rejected_frames_count=len(self.recorder.rejected_frames),
                solver_mode=solver_mode,
                reference_ids=reference_ids,
            )
        except Exception as exc:
            if solver_mode == "torso":
                try:
                    result = solve_geometry(
                        frames=self.recorder.accepted_frames,
                        tag_ids=tag_ids,
                        aruco_dictionary=self.dictionary_combo.currentText(),
                        tag_size_mm=self.tag_size_spin.value(),
                        camera_calibration_file=str(calibration_file),
                        rejected_frames_count=len(self.recorder.rejected_frames),
                        solver_mode="simple",
                        reference_ids=reference_ids,
                    )
                    notes.append(
                        "Локальная система не собралась устойчиво, поэтому автоматически использован режим расстояний."
                    )
                    notes.append(f"Причина fallback: {exception_to_text(exc)}")
                except Exception as fallback_exc:
                    QMessageBox.critical(self, "Ошибка калибровки", exception_to_text(fallback_exc))
                    self.log_message.emit(f"Ошибка расчёта геометрии: {exception_to_text(fallback_exc)}")
                    return
            else:
                QMessageBox.critical(self, "Ошибка калибровки", exception_to_text(exc))
                self.log_message.emit(f"Ошибка расчёта геометрии: {exception_to_text(exc)}")
                return

        if notes:
            result.quality_notes = list(result.quality_notes) + notes

        export_paths = export_all_reports(result, output_dir)
        self.session_active = False
        self.recorder.stop()
        self.current_output_dir = output_dir
        summary_lines = [
            f"Сессия завершена. Папка результатов: {output_dir}",
            f"Режим результата: {result.solver_mode}",
            f"Использовано кадров: {result.used_frames}",
            f"Отброшено кадров: {result.rejected_frames}",
            f"Итоговая оценка качества: {result.quality_overall_score:.2f}",
            f"JSON: {export_paths['json']}",
            f"CSV: {export_paths['csv']}",
            f"TXT: {export_paths['txt']}",
        ]
        if result.quality_notes:
            summary_lines.append("Примечания:")
            summary_lines.extend(f"- {note}" for note in result.quality_notes)
        self.session_summary.setPlainText("\n".join(summary_lines))
        self.log_message.emit(f"Результаты сессии сохранены в {output_dir}")
        self.results_ready.emit(result, str(output_dir))

    def _update_frame(self) -> None:
        try:
            frame = self.camera.read_frame()
        except Exception as exc:
            self._stop_camera()
            self.log_message.emit(f"Камера остановлена: {exception_to_text(exc)}")
            return

        self.frame_index += 1
        self.latest_frame_size = (frame.shape[1], frame.shape[0])
        calibration, calibration_source = self._get_active_calibration(self.latest_frame_size)
        camera_matrix = calibration.camera_matrix_np()
        dist_coeffs = calibration.dist_coeffs_np()

        try:
            corners, ids, _ = detect_markers(frame, self.dictionary_combo.currentText())
            observations = estimate_marker_observations(
                corners,
                ids,
                camera_matrix,
                dist_coeffs,
                self.tag_size_spin.value(),
            )
        except Exception as exc:
            self._stop_camera()
            QMessageBox.critical(self, "Ошибка детекта", exception_to_text(exc))
            self.log_message.emit(f"Ошибка ArUco-детекта: {exception_to_text(exc)}")
            return

        blur_value = blur_score(frame)
        tag_ids = self._safe_required_ids()
        required_ids_set = set(tag_ids)
        found_required = sorted(required_ids_set.intersection(observations))
        mean_tag_size = (
            float(np.mean([observations[tag_id].tag_size_px for tag_id in found_required]))
            if found_required
            else 0.0
        )
        max_angle = max((observations[tag_id].angle_deg for tag_id in found_required), default=0.0)

        self.calibration_label.setText(f"Калибровка камеры: {calibration_source}")
        self.found_status_label.setText(f"Найдено {len(found_required)}/{len(tag_ids)}")
        self.found_ids_label.setText(
            "ID в кадре: " + (", ".join(str(tag_id) for tag_id in sorted(observations)) or "-")
        )
        self.tag_size_label.setText(f"Примерный размер тега: {mean_tag_size:.1f}px")
        self.blur_label.setText(
            f"Оценка резкости: {blur_value:.1f}" + (" | кадр смазан" if blur_value < 80.0 else "")
        )
        self.angle_label.setText(
            f"Предупреждение по углу: {max_angle:.1f} град"
            + (" | угол слишком острый" if max_angle > 65.0 else "")
        )
        self.guidance_label.setText(
            self._guidance_text(mean_tag_size, blur_value, max_angle, found_required, tag_ids)
        )
        self.live_distances_box.setPlainText(self._pairwise_text(observations, tag_ids))

        annotated = draw_marker_annotations(
            frame,
            corners,
            ids,
            observations=observations,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            axis_length_mm=self.tag_size_spin.value() / 2.0,
        )
        annotated = self._draw_live_distance_vectors(
            annotated,
            corners,
            ids,
            observations,
            tag_ids,
        )

        if self.session_active and self.recorder is not None:
            self._evaluate_session_frame(observations, blur_value)

        pixmap = _frame_to_pixmap(annotated)
        self.preview_pixmap = pixmap
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _pairwise_text(
        self,
        observations: dict[int, MarkerPoseObservation],
        tag_ids: list[int],
    ) -> str:
        visible = [tag_id for tag_id in tag_ids if tag_id in observations]
        if len(visible) < 2:
            return "Нужно видеть хотя бы 2 тега, чтобы показать расстояния."

        lines = ["Текущие расстояния в координатах камеры:"]
        for tag_a, tag_b in combinations(visible, 2):
            point_a = np.asarray(observations[tag_a].position_mm, dtype=np.float64)
            point_b = np.asarray(observations[tag_b].position_mm, dtype=np.float64)
            distance_mm = float(np.linalg.norm(point_a - point_b))
            lines.append(f"{tag_a}-{tag_b}: {distance_mm:.1f} мм")
        if len(visible) == len(tag_ids):
            lines.append("")
            lines.append("Эти расстояния уже можно использовать даже без локальной системы координат.")
        return "\n".join(lines)

    def _draw_live_distance_vectors(
        self,
        frame: np.ndarray,
        corners: list[np.ndarray],
        ids: list[int],
        observations: dict[int, MarkerPoseObservation],
        tag_ids: list[int],
    ) -> np.ndarray:
        """Draw thick pairwise distance vectors and labels over the live preview."""
        centers: dict[int, np.ndarray] = {}
        for tag_id, marker_corners in zip(ids, corners):
            centers[int(tag_id)] = np.asarray(marker_corners, dtype=np.float32).reshape(-1, 2).mean(axis=0)

        visible = [tag_id for tag_id in tag_ids if tag_id in observations and tag_id in centers]
        if len(visible) < 2:
            return frame

        overlay = frame.copy()
        font_scale = max(0.72, min(frame.shape[0], frame.shape[1]) / 820.0)
        line_thickness = max(2, int(round(font_scale * 2.6)))
        text_thickness = max(2, line_thickness)
        colors = [
            (40, 80, 240),
            (30, 170, 80),
            (230, 150, 20),
            (190, 70, 190),
            (0, 160, 210),
            (220, 90, 70),
        ]

        for pair_index, (tag_a, tag_b) in enumerate(combinations(visible, 2)):
            start = centers[tag_a].astype(np.float64)
            end = centers[tag_b].astype(np.float64)
            distance_mm = float(
                np.linalg.norm(
                    np.asarray(observations[tag_a].position_mm, dtype=np.float64)
                    - np.asarray(observations[tag_b].position_mm, dtype=np.float64)
                )
            )

            color = colors[pair_index % len(colors)]
            start_point = (int(round(start[0])), int(round(start[1])))
            end_point = (int(round(end[0])), int(round(end[1])))
            cv2.line(overlay, start_point, end_point, color, line_thickness, cv2.LINE_AA)

            direction = end - start
            norm = float(np.linalg.norm(direction))
            if norm < 1e-6:
                continue
            normal = np.array([-direction[1], direction[0]], dtype=np.float64) / norm
            offset_scale = 26.0 + 18.0 * (pair_index % 3)
            if pair_index % 2 == 1:
                offset_scale *= -1.0
            midpoint = (start + end) / 2.0 + normal * offset_scale

            label = f"{tag_a}-{tag_b}: {distance_mm:.0f} мм"
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_thickness,
            )
            x = int(np.clip(midpoint[0] - text_width / 2.0, 8, max(8, frame.shape[1] - text_width - 16)))
            y = int(np.clip(midpoint[1] + text_height / 2.0, text_height + 12, max(text_height + 12, frame.shape[0] - 12)))
            top_left = (x - 8, y - text_height - 8)
            bottom_right = (x + text_width + 8, y + baseline + 8)
            cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), -1)
            cv2.rectangle(overlay, top_left, bottom_right, color, 2)
            cv2.putText(
                overlay,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (20, 20, 20),
                text_thickness,
                cv2.LINE_AA,
            )

        cv2.putText(
            overlay,
            "Живые расстояния между тегами",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.78, font_scale),
            (255, 255, 255),
            text_thickness + 1,
            cv2.LINE_AA,
        )

        return cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0)

    def _guidance_text(
        self,
        mean_tag_size: float,
        blur_value: float,
        max_angle: float,
        found_required: list[int],
        required_ids: list[int],
    ) -> str:
        guidance = [
            "Повернитесь немного влево/вправо.",
            "Подержите все 4 тега в кадре несколько секунд.",
        ]
        if mean_tag_size and mean_tag_size < 25.0:
            guidance.append("Подойдите ближе")
        if mean_tag_size and mean_tag_size < 18.0:
            guidance.append("Для теста с телефона откройте отдельный PNG-тег крупно, а не весь лист 2x2")
        if blur_value < 80.0:
            guidance.append("Улучшите освещение или уменьшите движение")
        if max_angle > 65.0:
            guidance.append("Сделайте угол обзора менее острым")
        if len(found_required) < len(required_ids):
            missing = [tag_id for tag_id in required_ids if tag_id not in found_required]
            guidance.append(f"Сейчас видны не все нужные теги. Не хватает: {missing}")
        if len(found_required) <= 2 and mean_tag_size < 25.0:
            guidance.append("Контакт-лист на экране телефона часто слишком мелкий для уверенного детекта")
        if len(found_required) < len(required_ids) or blur_value < 80.0:
            self.recent_weak_frames += 1
        else:
            self.recent_weak_frames = 0
        if self.recent_weak_frames >= 5:
            guidance.append("Теги часто теряются: попробуйте лучшее освещение, меньше движение и больше размер в кадре")
        return " | ".join(dict.fromkeys(guidance))

    def _evaluate_session_frame(
        self,
        observations: dict[int, MarkerPoseObservation],
        blur_value: float,
    ) -> None:
        now = time.monotonic()
        if now - self.last_eval_time < 0.25:
            return
        self.last_eval_time = now

        accepted, reason = self.recorder.process_frame(self.frame_index, observations, blur_value)
        good_frames = len(self.recorder.accepted_frames)
        rejected_frames = len(self.recorder.rejected_frames)
        self.good_frames_label.setText(f"Хороших кадров собрано: {good_frames}")
        self.rejected_frames_label.setText(f"Отклонено кандидатов: {rejected_frames}")
        target = self.recorder.config.target_frames
        self.progress_bar.setValue(min(100, int(100.0 * good_frames / max(target, 1))))

        if accepted:
            self.last_decision_label.setText(
                f"Последняя оценка кадра: принят кадр #{good_frames} (frame {self.frame_index})"
            )
            self.log_message.emit(f"Принят хороший кадр #{good_frames} на frame {self.frame_index}")
        else:
            self.last_decision_label.setText(f"Последняя оценка кадра: отклонён, причина: {reason}")
            if len(self.recorder.rejected_frames) % 10 == 0:
                self.log_message.emit(
                    f"Отброшено кандидатов: {len(self.recorder.rejected_frames)} | последняя причина: {reason}"
                )

    def _safe_required_ids(self) -> list[int]:
        try:
            tag_ids = parse_tag_ids(self.tag_ids_edit.text())
            return tag_ids if len(tag_ids) == 4 else [10, 20, 30, 40]
        except Exception:
            return [10, 20, 30, 40]

    def shutdown(self) -> None:
        self._stop_camera()

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
