"""Application entrypoint for the IMU ArUco calibration MVP."""

from __future__ import annotations

import os
import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication


def _configure_runtime() -> None:
    """Tune process-wide runtime settings before the UI starts."""
    os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
    try:
        import cv2

        if hasattr(cv2, "setLogLevel"):
            try:
                cv2.setLogLevel(0)
            except Exception:
                pass
    except Exception:
        pass


def main() -> int:
    """Run the desktop application."""
    _configure_runtime()
    from app.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("Калибровка IMU по ArUco")
    app.setFont(QFont("Segoe UI", 11))
    app.setStyleSheet(
        """
        QPushButton {
            min-height: 36px;
            padding: 6px 16px;
            min-width: 220px;
        }
        QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {
            min-height: 32px;
        }
        """
    )
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
