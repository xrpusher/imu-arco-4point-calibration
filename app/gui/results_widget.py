"""Results tab for viewing and exporting calibration data."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QDesktopServices,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.export_utils import (
    write_calibration_json,
    write_calibration_report_csv,
    write_calibration_report_txt,
)
from app.core.models import CalibrationComputationResult
from app.utils.logging_utils import exception_to_text
from app.utils.paths import ensure_dir


class TorsoSchematicWidget(QWidget):
    """Draw an arms-out body schematic with solved points and pairwise distances."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: CalibrationComputationResult | None = None
        self.setMinimumHeight(460)
        self.setStyleSheet("background: #fafafa; border: 1px solid #d6d6d6;")

    def set_result(self, result: CalibrationComputationResult | None) -> None:
        """Store a result object and repaint the widget."""
        self._result = result
        self.update()

    def _normalized_points(self) -> dict[str, QPointF]:
        """Normalize local coordinates into the 0..1 range for drawing."""
        if self._result is None:
            return {}

        raw_points = {
            str(tag_id): self._result.points_local[str(tag_id)]
            for tag_id in self._result.tag_ids
            if str(tag_id) in self._result.points_local
        }
        if not raw_points:
            return {}

        y_values = [coords[1] for coords in raw_points.values()]
        z_values = [coords[2] for coords in raw_points.values()]
        use_z_for_vertical = (max(y_values) - min(y_values) < 1e-6) or (
            max(z_values) - min(z_values) > (max(y_values) - min(y_values)) * 1.25
        )

        points_2d = np.asarray(
            [
                [coords[0], coords[2] if use_z_for_vertical else coords[1]]
                for coords in raw_points.values()
            ],
            dtype=np.float64,
        )
        centered = points_2d - points_2d.mean(axis=0, keepdims=True)
        if len(points_2d) >= 2:
            covariance = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            order = np.argsort(eigenvalues)[::-1]
            principal = eigenvectors[:, order[0]]
            secondary = np.array([-principal[1], principal[0]], dtype=np.float64)
            rotation = np.column_stack([principal, secondary])
            aligned = centered @ rotation
        else:
            aligned = centered

        min_x, min_y = aligned.min(axis=0)
        max_x, max_y = aligned.max(axis=0)
        span_x = max(float(max_x - min_x), 1.0)
        span_y = max(float(max_y - min_y), 1.0)

        normalized: dict[str, QPointF] = {}
        for index, tag_id in enumerate(raw_points.keys()):
            draw_x = float((aligned[index, 0] - min_x) / span_x)
            draw_y = float((aligned[index, 1] - min_y) / span_y)
            normalized[tag_id] = QPointF(draw_x, draw_y)
        return normalized

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Paint the arms-out silhouette and overlay solved data."""
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        canvas = self.rect().adjusted(16, 16, -16, -16)
        painter.fillRect(canvas, QColor("#fcfcfc"))

        title_font = QFont(painter.font())
        title_font.setPointSize(title_font.pointSize() + 1)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#2f2f2f"))
        painter.drawText(
            canvas.adjusted(8, 8, -8, -8),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            "Схема 4 датчиков на позе с разведёнными руками",
        )

        body_rect = QRectF(
            canvas.left() + 40,
            canvas.top() + 56,
            max(320.0, canvas.width() - 80.0),
            max(260.0, canvas.height() - 120.0),
        )

        body_pen = QPen(QColor("#b9c4cf"), 8)
        painter.setPen(body_pen)
        painter.setBrush(QColor("#eef2f6"))

        head_center = QPointF(body_rect.center().x(), body_rect.top() + body_rect.height() * 0.13)
        painter.drawEllipse(head_center, 26, 26)

        torso_top = QPointF(body_rect.center().x(), body_rect.top() + body_rect.height() * 0.24)
        torso_bottom = QPointF(body_rect.center().x(), body_rect.top() + body_rect.height() * 0.84)
        shoulder_y = body_rect.top() + body_rect.height() * 0.30
        arm_left = QPointF(body_rect.left() + body_rect.width() * 0.05, shoulder_y)
        arm_right = QPointF(body_rect.right() - body_rect.width() * 0.05, shoulder_y)
        elbow_left = QPointF(body_rect.left() + body_rect.width() * 0.24, shoulder_y + 4)
        elbow_right = QPointF(body_rect.right() - body_rect.width() * 0.24, shoulder_y + 4)
        hip_left = QPointF(body_rect.center().x() - 55, body_rect.top() + body_rect.height() * 0.72)
        hip_right = QPointF(body_rect.center().x() + 55, body_rect.top() + body_rect.height() * 0.72)

        path = QPainterPath()
        path.moveTo(arm_left)
        path.lineTo(elbow_left)
        path.lineTo(QPointF(body_rect.center().x() - 54, shoulder_y))
        path.lineTo(QPointF(body_rect.center().x() + 54, shoulder_y))
        path.lineTo(elbow_right)
        path.lineTo(arm_right)
        path.moveTo(torso_top)
        path.lineTo(torso_bottom)
        path.moveTo(hip_left)
        path.lineTo(QPointF(body_rect.center().x(), body_rect.bottom() - 10))
        path.lineTo(hip_right)
        painter.drawPath(path)

        painter.setPen(QPen(QColor("#dfe6ee"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            QRectF(
                body_rect.center().x() - 54,
                torso_top.y(),
                108,
                hip_left.y() - torso_top.y(),
            ),
            18,
            18,
        )

        if self._result is None:
            painter.setPen(QColor("#6d7781"))
            painter.drawText(
                body_rect.adjusted(0, 10, 0, 0),
                int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter),
                "После завершения сессии здесь появится схема 4 точек и расстояний между ними.",
            )
            return

        draw_area = QRectF(
            body_rect.left() + 18,
            body_rect.top() + 10,
            body_rect.width() - 36,
            body_rect.height() - 36,
        )
        normalized = self._normalized_points()
        if not normalized:
            painter.setPen(QColor("#6d7781"))
            painter.drawText(
                body_rect.adjusted(0, 10, 0, 0),
                int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter),
                "Нет координат для построения схемы.",
            )
            return

        palette = ["#ef4444", "#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#0f766e"]
        point_map: dict[str, QPointF] = {}
        for index, tag_id in enumerate(self._result.tag_ids):
            normalized_point = normalized.get(str(tag_id))
            if normalized_point is None:
                continue
            point_map[str(tag_id)] = QPointF(
                draw_area.left() + normalized_point.x() * draw_area.width(),
                draw_area.bottom() - normalized_point.y() * draw_area.height(),
            )

        line_pen = QPen(QColor(68, 92, 120, 130), 2)
        small_font = QFont(painter.font())
        small_font.setPointSize(max(8, small_font.pointSize() - 1))
        painter.setFont(small_font)
        for left_id, right_id in combinations(self._result.tag_ids, 2):
            left_key = str(left_id)
            right_key = str(right_id)
            if left_key not in point_map or right_key not in point_map:
                continue
            pair_key = f"{min(left_id, right_id)}-{max(left_id, right_id)}"
            distance = self._result.pairwise_distances_mm.get(pair_key)
            if distance is None:
                continue
            start = point_map[left_key]
            end = point_map[right_key]
            painter.setPen(line_pen)
            painter.drawLine(start, end)
            midpoint = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
            label_rect = QRectF(midpoint.x() - 44, midpoint.y() - 12, 88, 24)
            painter.fillRect(label_rect, QColor(255, 255, 255, 220))
            painter.setPen(QColor("#334155"))
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignCenter), f"{distance:.0f} мм")

        id_font = QFont(painter.font())
        id_font.setPointSize(id_font.pointSize() + 1)
        id_font.setBold(True)
        painter.setFont(id_font)
        for index, tag_id in enumerate(self._result.tag_ids):
            point = point_map.get(str(tag_id))
            if point is None:
                continue
            color = QColor(palette[index % len(palette)])
            painter.setPen(QPen(QColor("#ffffff"), 2))
            painter.setBrush(color)
            painter.drawEllipse(point, 10, 10)
            painter.setPen(QColor("#111827"))
            painter.drawText(
                QRectF(point.x() + 12, point.y() - 18, 84, 24),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                f"ID {tag_id}",
            )

        painter.setPen(QColor("#6b7280"))
        painter.setFont(small_font)
        painter.drawText(
            QRectF(canvas.left() + 12, canvas.bottom() - 32, canvas.width() - 24, 24),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            "Силуэт условный, но поза специально показана с руками в стороны. Точки и расстояния берутся из реального расчёта.",
        )


class ResultsWidget(QWidget):
    """Display final geometry results and allow exporting them again."""

    log_message = pyqtSignal(str)

    def __init__(self, default_output_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default_output_root = ensure_dir(default_output_root)
        self.current_result: CalibrationComputationResult | None = None
        self.current_output_dir: Path | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.summary_label = QLabel("Результатов пока нет. Завершите сессию калибровки.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        button_row = QHBoxLayout()
        self.save_json_button = QPushButton("Сохранить JSON")
        self.save_json_button.clicked.connect(self._save_json)
        self.save_csv_button = QPushButton("Сохранить CSV")
        self.save_csv_button.clicked.connect(self._save_csv)
        self.save_txt_button = QPushButton("Сохранить TXT")
        self.save_txt_button.clicked.connect(self._save_txt)
        self.open_folder_button = QPushButton("Открыть папку результатов")
        self.open_folder_button.clicked.connect(self._open_output_folder)
        button_row.addWidget(self.save_json_button)
        button_row.addWidget(self.save_csv_button)
        button_row.addWidget(self.save_txt_button)
        button_row.addWidget(self.open_folder_button)
        layout.addLayout(button_row)

        schematic_group = QGroupBox("Наглядная схема")
        schematic_layout = QVBoxLayout(schematic_group)
        schematic_hint = QLabel(
            "Ниже показана условная поза с разведёнными руками, как в твоём сценарии с 4 датчиками. "
            "Схема нужна для наглядного просмотра точек и расстояний."
        )
        schematic_hint.setWordWrap(True)
        self.schematic_widget = TorsoSchematicWidget()
        schematic_layout.addWidget(schematic_hint)
        schematic_layout.addWidget(self.schematic_widget)
        layout.addWidget(schematic_group)

        points_group = QGroupBox("Локальные 3D-координаты")
        points_layout = QVBoxLayout(points_group)
        self.points_table = QTableWidget(0, 7)
        self.points_table.setHorizontalHeaderLabels(
            ["ID тега", "X (мм)", "Y (мм)", "Z (мм)", "Std X", "Std Y", "Std Z"]
        )
        self.points_table.verticalHeader().setVisible(False)
        self.points_table.setMinimumHeight(220)
        self.points_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        points_layout.addWidget(self.points_table)
        layout.addWidget(points_group)

        pairwise_group = QGroupBox("Попарные расстояния")
        pairwise_layout = QVBoxLayout(pairwise_group)
        self.pairwise_table = QTableWidget(0, 7)
        self.pairwise_table.setHorizontalHeaderLabels(
            ["Пара", "Расстояние", "Среднее", "Медиана", "Std", "MAD", "Сэмплов"]
        )
        self.pairwise_table.verticalHeader().setVisible(False)
        self.pairwise_table.setMinimumHeight(220)
        self.pairwise_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        pairwise_layout.addWidget(self.pairwise_table)
        layout.addWidget(pairwise_group)

        warnings_group = QGroupBox("Предупреждения")
        warnings_layout = QVBoxLayout(warnings_group)
        self.warnings_box = QPlainTextEdit()
        self.warnings_box.setReadOnly(True)
        self.warnings_box.setPlaceholderText("После расчёта здесь появятся замечания и предупреждения.")
        warnings_layout.addWidget(self.warnings_box)
        layout.addWidget(warnings_group)

    def set_result(self, result: CalibrationComputationResult, output_dir: str) -> None:
        """Show a new computed result in the UI."""
        self.current_result = result
        self.current_output_dir = Path(output_dir)
        mode_name = "Локальная система torso" if result.solver_mode == "torso" else "Расстояния и форма"
        b21s_note = " | B21S-масштаб" if abs(result.tag_size_mm - 27.0) < 1.5 else ""
        self.summary_label.setText(
            f"Режим: {mode_name}{b21s_note} | использовано кадров: {result.used_frames} | "
            f"отброшено кадров: {result.rejected_frames} | итоговая оценка: {result.quality_overall_score:.2f}"
        )
        self._populate_points_table(result)
        self._populate_pairwise_table(result)
        self.schematic_widget.set_result(result)
        warnings = result.quality_notes or ["Предупреждений нет."]
        self.warnings_box.setPlainText("\n".join(warnings))

    def _populate_points_table(self, result: CalibrationComputationResult) -> None:
        self.points_table.setRowCount(len(result.tag_ids))
        for row, tag_id in enumerate(result.tag_ids):
            coordinates = result.points_local[str(tag_id)]
            variance = result.point_variance_mm.get(str(tag_id), [0.0, 0.0, 0.0])
            values = [
                str(tag_id),
                f"{coordinates[0]:.3f}",
                f"{coordinates[1]:.3f}",
                f"{coordinates[2]:.3f}",
                f"{variance[0]:.3f}",
                f"{variance[1]:.3f}",
                f"{variance[2]:.3f}",
            ]
            for column, value in enumerate(values):
                self.points_table.setItem(row, column, QTableWidgetItem(value))

    def _populate_pairwise_table(self, result: CalibrationComputationResult) -> None:
        pairs = sorted(result.pairwise_distances_mm.keys())
        self.pairwise_table.setRowCount(len(pairs))
        for row, pair_key in enumerate(pairs):
            stats = result.pairwise_stats_mm[pair_key]
            values = [
                pair_key,
                f"{result.pairwise_distances_mm[pair_key]:.3f}",
                f"{stats.mean:.3f}",
                f"{stats.median:.3f}",
                f"{stats.std:.3f}",
                f"{stats.mad:.3f}",
                str(stats.samples),
            ]
            for column, value in enumerate(values):
                self.pairwise_table.setItem(row, column, QTableWidgetItem(value))

    def _save_json(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "Нет результата", "Результат калибровки ещё не рассчитан.")
            return
        default_path = str((self.current_output_dir or self.default_output_root) / "calibration.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить JSON", default_path, "JSON files (*.json)")
        if not file_path:
            return
        try:
            write_calibration_json(self.current_result, file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", exception_to_text(exc))
            return
        self.log_message.emit(f"Сохранён calibration JSON: {file_path}")

    def _save_csv(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "Нет результата", "Результат калибровки ещё не рассчитан.")
            return
        default_path = str((self.current_output_dir or self.default_output_root) / "calibration_report.csv")
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", default_path, "CSV files (*.csv)")
        if not file_path:
            return
        try:
            write_calibration_report_csv(self.current_result, file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", exception_to_text(exc))
            return
        self.log_message.emit(f"Сохранён calibration CSV: {file_path}")

    def _save_txt(self) -> None:
        if self.current_result is None:
            QMessageBox.warning(self, "Нет результата", "Результат калибровки ещё не рассчитан.")
            return
        default_path = str((self.current_output_dir or self.default_output_root) / "calibration_report.txt")
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить TXT", default_path, "Text files (*.txt)")
        if not file_path:
            return
        try:
            write_calibration_report_txt(self.current_result, file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", exception_to_text(exc))
            return
        self.log_message.emit(f"Сохранён calibration TXT: {file_path}")

    def _open_output_folder(self) -> None:
        if self.current_output_dir is None:
            QMessageBox.warning(self, "Нет папки результатов", "Сначала завершите сессию калибровки.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_output_dir)))
