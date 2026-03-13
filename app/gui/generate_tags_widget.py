"""Tag generation tab."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
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
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core import aruco_utils
from app.utils.logging_utils import exception_to_text
from app.utils.paths import ensure_dir


class GenerateTagsWidget(QWidget):
    """Widget for generating printable ArUco tags."""

    log_message = pyqtSignal(str)

    def __init__(self, default_output_dir: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default_output_dir = ensure_dir(default_output_dir)
        self.preview_pixmap: QPixmap | None = None
        self._build_ui()
        self._apply_layout_preset()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        intro = QLabel(
            "Сгенерируйте 4 ArUco-тега для печати. По умолчанию выбран пресет для термопринтера B21S "
            "с этикеткой 50x30 мм и плотностью 203 dpi."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        note = QLabel(
            "Для печати лучше использовать PNG. Для теста с телефона лучше открывать отдельные файлы "
            "`tag_10.png`, `tag_20.png` и т.д. крупно, а не весь contact-sheet сразу."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        controls_group = QGroupBox("Параметры генерации")
        controls_layout = QFormLayout(controls_group)

        self.dictionary_combo = QComboBox()
        self.aruco_error: str | None = None
        try:
            dictionaries = aruco_utils.available_dictionaries()
        except Exception as exc:
            dictionaries = [aruco_utils.DEFAULT_DICTIONARY]
            self.aruco_error = exception_to_text(exc)
        self.dictionary_combo.addItems(dictionaries)
        default_index = self.dictionary_combo.findText(aruco_utils.DEFAULT_DICTIONARY)
        if default_index >= 0:
            self.dictionary_combo.setCurrentIndex(default_index)

        self.layout_combo = QComboBox()
        for layout_name, label in aruco_utils.available_layouts():
            self.layout_combo.addItem(label, layout_name)
        self.layout_combo.currentIndexChanged.connect(self._apply_layout_preset)

        self.ids_edit = QLineEdit("10, 20, 30, 40")

        self.tag_size_spin = QDoubleSpinBox()
        self.tag_size_spin.setRange(10.0, 300.0)
        self.tag_size_spin.setDecimals(1)
        self.tag_size_spin.setValue(round(aruco_utils.b21s_recommended_tag_size_mm(), 1))
        self.tag_size_spin.setSuffix(" мм")

        canvas_grid = QGridLayout()
        self.image_width_spin = QSpinBox()
        self.image_width_spin.setRange(200, 4000)
        self.image_width_spin.setValue(1000)
        self.image_width_spin.setSuffix(" px")
        self.image_height_spin = QSpinBox()
        self.image_height_spin.setRange(200, 4000)
        self.image_height_spin.setValue(1000)
        self.image_height_spin.setSuffix(" px")
        canvas_grid.addWidget(QLabel("Ширина"), 0, 0)
        canvas_grid.addWidget(self.image_width_spin, 0, 1)
        canvas_grid.addWidget(QLabel("Высота"), 0, 2)
        canvas_grid.addWidget(self.image_height_spin, 0, 3)
        canvas_widget = QWidget()
        canvas_widget.setLayout(canvas_grid)

        folder_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit(str(self.default_output_dir))
        browse_button = QPushButton("Выбрать папку")
        browse_button.clicked.connect(self._browse_output_dir)
        folder_row.addWidget(self.output_dir_edit)
        folder_row.addWidget(browse_button)
        folder_widget = QWidget()
        folder_widget.setLayout(folder_row)

        controls_layout.addRow("Словарь ArUco", self.dictionary_combo)
        controls_layout.addRow("Макет печати", self.layout_combo)
        controls_layout.addRow("ID тегов", self.ids_edit)
        controls_layout.addRow("Физический размер тега", self.tag_size_spin)
        controls_layout.addRow("Размер выходного изображения", canvas_widget)
        controls_layout.addRow("Папка сохранения", folder_widget)
        layout.addWidget(controls_group)

        self.layout_info_label = QLabel()
        self.layout_info_label.setWordWrap(True)
        layout.addWidget(self.layout_info_label)
        default_layout = self.layout_combo.findData(aruco_utils.LAYOUT_B21S_50X30)
        if default_layout >= 0:
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(default_layout)
            self.layout_combo.blockSignals(False)

        self.generate_button = QPushButton("Сгенерировать теги")
        self.generate_button.clicked.connect(self._generate_tags)
        self.generate_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(self.generate_button)

        self.status_label = QLabel("Готово к генерации.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        preview_group = QGroupBox("Предпросмотр листа")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel("После генерации здесь появится preview.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(520)
        self.preview_label.setMaximumHeight(1200)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background: #f3f3f3;")
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group, stretch=1)

        if self.aruco_error:
            self.generate_button.setEnabled(False)
            self.status_label.setText(
                "Генерация ArUco недоступна. Установите opencv-contrib-python.\n"
                f"Подробности: {self.aruco_error}"
            )

    def _apply_layout_preset(self) -> None:
        layout_name = str(self.layout_combo.currentData())
        if layout_name == aruco_utils.LAYOUT_B21S_50X30:
            width_px, height_px = aruco_utils.layout_canvas_size(layout_name)
            recommended_mm = aruco_utils.b21s_recommended_tag_size_mm()
            self.image_width_spin.setValue(width_px)
            self.image_height_spin.setValue(height_px)
            self.image_width_spin.setEnabled(False)
            self.image_height_spin.setEnabled(False)
            self.tag_size_spin.setValue(round(recommended_mm, 1))
            self.tag_size_spin.setEnabled(False)
            self.layout_info_label.setText(
                "Пресет B21S: точный размер файла 400x240 px для этикетки 50x30 мм при 203 dpi. "
                f"Фактический размер квадратного маркера на наклейке будет около {recommended_mm:.1f} мм. "
                "Это намного практичнее для твоего принтера, чем исходные 60 мм."
            )
            return

        self.image_width_spin.setEnabled(True)
        self.image_height_spin.setEnabled(True)
        self.tag_size_spin.setEnabled(True)
        if self.image_width_spin.value() == aruco_utils.B21S_CANVAS_WIDTH_PX:
            self.image_width_spin.setValue(1000)
        if self.image_height_spin.value() == aruco_utils.B21S_CANVAS_HEIGHT_PX:
            self.image_height_spin.setValue(1000)
        if abs(self.tag_size_spin.value() - aruco_utils.b21s_recommended_tag_size_mm()) < 1.0:
            self.tag_size_spin.setValue(60.0)
        self.layout_info_label.setText(
            "Свободный макет: можно задать любую канву и физический размер. "
            "Этот режим удобен для обычной печати на листе."
        )

    def _browse_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку сохранения",
            self.output_dir_edit.text(),
        )
        if directory:
            self.output_dir_edit.setText(directory)

    def _generate_tags(self) -> None:
        try:
            tag_ids = aruco_utils.parse_tag_ids(self.ids_edit.text())
            if len(tag_ids) != 4:
                raise ValueError("В этом MVP нужно указать ровно 4 ID тега.")

            canvas_size_px = (self.image_width_spin.value(), self.image_height_spin.value())
            result = aruco_utils.generate_tags(
                output_dir=self.output_dir_edit.text(),
                dictionary_name=self.dictionary_combo.currentText(),
                tag_ids=tag_ids,
                tag_size_mm=self.tag_size_spin.value(),
                canvas_size_px=canvas_size_px,
                layout_name=str(self.layout_combo.currentData()),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка генерации тегов", exception_to_text(exc))
            self.log_message.emit(f"Ошибка генерации тегов: {exception_to_text(exc)}")
            return

        self.status_label.setText(
            f"Сохранены 4 тега в PNG и JPG. Канва: {result.canvas_width_px}x{result.canvas_height_px} px. "
            f"Фактический размер тега: около {result.rendered_tag_size_mm or result.tag_size_mm:.1f} мм. "
            "Для печати рекомендуется PNG."
        )
        self._set_preview(Path(result.preview_path))
        self.log_message.emit(f"Сгенерирован набор тегов: {result.output_dir}")

    def _set_preview(self, preview_path: Path) -> None:
        pixmap = QPixmap(str(preview_path))
        if pixmap.isNull():
            self.preview_label.setText(f"Не удалось загрузить preview: {preview_path}")
            return
        self.preview_pixmap = pixmap
        scaled = pixmap.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        current_pixmap = self.preview_pixmap
        if current_pixmap is not None and not current_pixmap.isNull():
            self.preview_label.setPixmap(
                current_pixmap.scaled(
                    self.preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
