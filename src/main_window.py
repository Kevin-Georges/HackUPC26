from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
    QLabel,
)
from PyQt5.QtCore import Qt
from constants import BG, PANEL, BORDER, TEXT, DIM
from model_viewer import ModelViewer
from charts_panel import ChartsPanel
from health_panel import HealthPanel


class MainWindow(QMainWindow):
    def __init__(self, model_path: str):
        super().__init__()
        self.setWindowTitle("HP Metal Jet S100 — Digital Twin")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(f"QWidget{{background:{BG}; color:{TEXT};}}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Left: 3-D viewer ──────────────────────────────────────────────────
        left = QFrame()
        left.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;"
        )
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(4)

        tip = QLabel(
            "Left-drag · rotate X/Y     Right-drag · rotate Z     Scroll · zoom"
        )
        tip.setAlignment(Qt.AlignCenter)
        tip.setStyleSheet(
            f"color:{DIM}; font-size:8pt; background:transparent; border:none;"
        )
        ll.addWidget(tip)
        ll.addWidget(ModelViewer(model_path))
        root.addWidget(left, stretch=3)

        # ── Right: chart + health table ───────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        right.addWidget(ChartsPanel(), stretch=2)

        health = HealthPanel()
        health.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;"
        )
        right.addWidget(health, stretch=1)

        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, stretch=2)
