from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QFont
from constants import GREEN, YELLOW, RED, TEXT, DIM


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
    _ROWS = [
        ("Recoater Blade", "70%"),
        ("Nozzle Plate", "50%"),
        ("Heating Elements", "100%"),
    ]

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        title = QLabel("Component Health  [placeholder — driven by Phase 1 engine]")
        title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        title.setStyleSheet(f"color:{TEXT}; border:none;")
        lay.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(2)

        for col, header in enumerate(("Component", "Status", "Health")):
            lbl = QLabel(header)
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setStyleSheet(f"color:{DIM}; border:none;")
            grid.addWidget(lbl, 0, col)

        for row, (name, pct) in enumerate(self._ROWS, 1):
            color = _health_color(pct)
            status = _pct_to_status(pct)
            for col, (val, style) in enumerate(
                [
                    (name, f"color:{TEXT};"),
                    (status, f"color:{color}; font-weight:bold;"),
                    (pct, f"color:{color};"),
                ]
            ):
                lbl = QLabel(val)
                lbl.setFont(QFont("Segoe UI", 8))
                lbl.setStyleSheet(style + " border:none;")
                grid.addWidget(lbl, row, col)

        lay.addLayout(grid)
