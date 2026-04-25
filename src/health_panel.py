from collections import deque

from PyQt5.QtWidgets import QFrame, QVBoxLayout, QLabel, QGridLayout, QWidget, QSizePolicy
from PyQt5.QtGui import QFont, QPainter, QColor, QFontMetrics
from PyQt5.QtCore import Qt
from constants import GREEN, YELLOW, RED, TEXT, DIM
from degradation_engine import DegradationEngine

_N_CELLS  = 90   # number of discrete time cells in each bar
_CELL_GAP = 1    # px gap between cells

_COLOR_HEALTHY   = QColor(GREEN)
_COLOR_DEGRADED  = QColor(YELLOW)
_COLOR_CRITICAL  = QColor(RED)
_COLOR_NOW_LINE  = QColor(TEXT)


def _health_to_qcolor(pct: int) -> QColor:
    if pct >= 70:
        return _COLOR_HEALTHY
    if pct >= 30:
        return _COLOR_DEGRADED
    return _COLOR_CRITICAL


class HealthBarWidget(QWidget):
    """Horizontal segmented timeline bar for one component's health history.

    Newest reading on the right, oldest on the left.  Cells not yet populated
    (buffer shorter than _N_CELLS) are left transparent so the bar grows in
    from the right as data arrives.  A thin vertical "now" line sits at the
    right edge of the last filled cell.
    """

    _FONT     = QFont("Segoe UI", 8)
    _TEXT_GAP = 6   # px between now-line and percentage text

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: deque[int] = deque(maxlen=_N_CELLS)
        self._current_pct: int = 0
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumWidth(150)
        self.setMinimumHeight(10)
        self.setMaximumHeight(20)

    def push(self, health_pct: int) -> None:
        self._current_pct = health_pct
        self._history.append(health_pct)
        self.update()

    def paintEvent(self, _) -> None:
        if not self._history:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        W = self.width()
        H = self.height()
        pad_v = max(1, H // 5)
        bar_h = H - 2 * pad_v

        # Reserve space on the right for "100%" so the bar width is stable
        fm        = QFontMetrics(self._FONT)
        label_w   = fm.horizontalAdvance("100%")
        bar_w     = W - label_w - self._TEXT_GAP

        history   = list(self._history)
        n_filled  = len(history)

        slot   = (bar_w - _CELL_GAP) / _N_CELLS
        cell_w = slot - _CELL_GAP

        for i, pct in enumerate(history):
            x0 = round(i * slot)
            x1 = round(x0 + cell_w)
            painter.fillRect(x0, pad_v, max(1, x1 - x0), bar_h,
                             _health_to_qcolor(pct))

        now_x = round((n_filled - 1) * slot + cell_w)
        painter.setPen(_COLOR_NOW_LINE)
        painter.drawLine(now_x, pad_v, now_x, pad_v + bar_h - 1)

        # Percentage label — right of the bar, vertically centred
        painter.setFont(self._FONT)
        painter.setPen(_health_to_qcolor(self._current_pct))
        text_x = bar_w + self._TEXT_GAP
        painter.drawText(text_x, 0, label_w, H, Qt.AlignVCenter | Qt.AlignRight,
                         f"{self._current_pct}%")

        painter.end()


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
        self._day = 0
        components = self._engine.tick(21.0, 0.0, 0.0)
        self._ROWS = [(c.name, c.pct_str) for c in components]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(4)

        self._title = QLabel(f"Component Health — Day {self._day}")
        self._title.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self._title.setStyleSheet(f"color:{TEXT}; border:none;")
        lay.addWidget(self._title)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(20)
        self._grid.setVerticalSpacing(2)

        for col, header in enumerate(("Component", "Status", "Health")):
            lbl = QLabel(header)
            lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            lbl.setStyleSheet(f"color:{DIM}; border:none;")
            self._grid.addWidget(lbl, 0, col)

        self._status_labels: list[QLabel] = []
        self._health_bars: list[HealthBarWidget] = []

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

            bar = HealthBarWidget()
            bar.push(int(pct.rstrip("%")))
            self._grid.addWidget(bar, row, 2)
            self._health_bars.append(bar)

        lay.addLayout(self._grid)

    def update(
        self,
        temperature:             float,
        humidity:                float,
        operational_load:        float,
        powder_quality:          float = 1.0,
        binder_viscosity_stress: float = 0.0,
        voltage_stress:          float = 0.0,
    ) -> None:
        """Advance the degradation engine one day and refresh all labels."""
        self._day += 1
        self._title.setText(f"Component Health — Day {self._day}")
        components = self._engine.tick(
            temperature,
            humidity,
            operational_load,
            powder_quality,
            binder_viscosity_stress,
            voltage_stress,
        )
        self._ROWS = [(c.name, c.pct_str) for c in components]

        for i, (_, pct) in enumerate(self._ROWS):
            color  = _health_color(pct)
            status = _pct_to_status(pct)

            self._status_labels[i].setText(status)
            self._status_labels[i].setStyleSheet(
                f"color:{color}; font-weight:bold; border:none;"
            )
            self._health_bars[i].push(int(pct.rstrip("%")))
