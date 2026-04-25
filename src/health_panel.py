from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QFont
from constants import GREEN, YELLOW, RED, TEXT, DIM
from degradation_engine import DegradationEngine


def _pct_to_status(pct_str: str) -> str:
    try:
        value = int(pct_str.rstrip("%"))
    except ValueError:
        return "Not Working"
    if value > 70:
        return "Fully Functional"
    if value > 30:
        return "Degraded"
    return "Not Working"


def _health_color(pct_str: str) -> str:
    try:
        value = int(pct_str.rstrip("%"))
    except ValueError:
        return RED
    if value > 70:
        return GREEN
    if value > 30:
        return YELLOW
    return RED


class HealthPanel(QFrame):

    def __init__(self):
        super().__init__()

        self._engine = DegradationEngine()
        components = self._engine.tick(21.0, 0.0, 0.0)
        self._ROWS = [(c.name, c.pct_str) for c in components]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        title = QLabel("Component Health")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT}; border:none;")
        lay.addWidget(title)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(20)
        self._grid.setVerticalSpacing(2)

        for col, header in enumerate(("Component", "Status", "Health")):
            lbl = QLabel(header)
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setStyleSheet(f"color:{DIM}; border:none;")
            self._grid.addWidget(lbl, 0, col)

        self._status_labels: list[QLabel] = []
        self._health_labels: list[QLabel] = []

        for row, (name, pct) in enumerate(self._ROWS, 1):
            color  = _health_color(pct)
            status = _pct_to_status(pct)

            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 8))
            name_lbl.setStyleSheet(f"color:{TEXT}; border:none;")
            self._grid.addWidget(name_lbl, row, 0)

            status_lbl = QLabel(status)
            status_lbl.setFont(QFont("Segoe UI", 8))
            status_lbl.setStyleSheet(f"color:{color}; font-weight:bold; border:none;")
            self._grid.addWidget(status_lbl, row, 1)
            self._status_labels.append(status_lbl)

            health_lbl = QLabel(pct)
            health_lbl.setFont(QFont("Segoe UI", 8))
            health_lbl.setStyleSheet(f"color:{color}; border:none;")
            self._grid.addWidget(health_lbl, row, 2)
            self._health_labels.append(health_lbl)

        lay.addLayout(self._grid)

    def update(
        self,
        temperature:      float,
        humidity:         float,
        operational_load: float,
    ) -> None:
        """Advance the degradation engine one tick and refresh all labels."""
        components = self._engine.tick(temperature, humidity, operational_load)
        self._ROWS = [(c.name, c.pct_str) for c in components]

        for i, (_, pct) in enumerate(self._ROWS):
            color  = _health_color(pct)
            status = _pct_to_status(pct)

            self._status_labels[i].setText(status)
            self._status_labels[i].setStyleSheet(
                f"color:{color}; font-weight:bold; border:none;"
            )
            self._health_labels[i].setText(pct)
            self._health_labels[i].setStyleSheet(f"color:{color}; border:none;")
