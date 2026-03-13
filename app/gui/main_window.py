"""Main application window."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QMainWindow,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.models import CameraCalibrationData
from app.gui.camera_calibration_widget import CameraCalibrationWidget
from app.gui.capture_calibrate_widget import CaptureCalibrateWidget
from app.gui.generate_tags_widget import GenerateTagsWidget
from app.gui.results_widget import ResultsWidget
from app.utils.logging_utils import format_log_message
from app.utils.paths import output_root


class MainWindow(QMainWindow):
    """Main PyQt application window with four tabs and a shared log panel."""

    def __init__(self) -> None:
        super().__init__()
        self.current_calibration: CameraCalibrationData | None = None
        self.current_calibration_file: str = ""

        self.setWindowTitle("Калибровка IMU по ArUco")
        self.resize(1480, 980)
        self.setMinimumSize(1100, 780)

        central = QWidget()
        central_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.log_panel = QPlainTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setPlaceholderText("Журнал приложения")

        base_output = output_root()
        self.generate_tags_widget = GenerateTagsWidget(base_output / "tags")
        self.camera_calibration_widget = CameraCalibrationWidget(base_output / "camera_calibration")
        self.capture_widget = CaptureCalibrateWidget(
            calibration_provider=self._get_current_calibration,
            default_output_root=base_output / "sessions",
        )
        self.results_widget = ResultsWidget(base_output / "sessions")

        self.tabs.addTab(self._make_scroll_tab(self.generate_tags_widget), "Генерация тегов")
        self.tabs.addTab(self._make_scroll_tab(self.camera_calibration_widget), "Калибровка камеры")
        self.tabs.addTab(self._make_scroll_tab(self.capture_widget), "Захват и калибровка")
        self.tabs.addTab(self._make_scroll_tab(self.results_widget), "Результаты")

        splitter.addWidget(self.tabs)
        splitter.addWidget(self.log_panel)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([820, 180])
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.generate_tags_widget.log_message.connect(self._append_log)
        self.camera_calibration_widget.log_message.connect(self._append_log)
        self.camera_calibration_widget.calibration_ready.connect(self._store_calibration)
        self.capture_widget.log_message.connect(self._append_log)
        self.capture_widget.results_ready.connect(self._push_results)
        self.results_widget.log_message.connect(self._append_log)

        self._append_log("Приложение запущено.")

    def _make_scroll_tab(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidget(widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll

    def _append_log(self, message: str) -> None:
        self.log_panel.appendPlainText(format_log_message(message))

    def _store_calibration(self, calibration: object, file_path: str) -> None:
        self.current_calibration = calibration if isinstance(calibration, CameraCalibrationData) else None
        self.current_calibration_file = file_path
        self._append_log(f"Активная калибровка камеры: {file_path}")

    def _get_current_calibration(self) -> tuple[CameraCalibrationData | None, str]:
        return self.current_calibration, self.current_calibration_file

    def _push_results(self, result: object, output_dir: str) -> None:
        self.results_widget.set_result(result, output_dir)
        self.tabs.setCurrentIndex(3)
        self._append_log(f"Результаты обновлены из {output_dir}")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.camera_calibration_widget.shutdown()
        self.capture_widget.shutdown()
        super().closeEvent(event)
