from collections import deque

import numpy as np
from PyQt5.QtCore import QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QSizePolicy,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
)
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from constants import PANEL, BG, RED, YELLOW, GREEN, TEXT, DIM, BORDER
from input_drivers import DriverSnapshot, DriverSuite, HumidityContaminationDriver
from temp_chart import TempChart, _WINDOW_DAYS

_TIMER_MS = 100


# ── Humidity / Contamination chart ────────────────────────────────────────────


class HumidityChart(FigureCanvasQTAgg):
    """Live humidity & contamination index chart. Driven by update(snap)."""

    def __init__(self):
        fig = Figure(facecolor=PANEL)
        fig.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.90)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._t_buf = deque(maxlen=_WINDOW_DAYS)
        self._y_buf = deque(maxlen=_WINDOW_DAYS)
        self._tick = 0

        ax = fig.add_subplot(111, facecolor=BG)
        self._ax = ax

        (self._line,) = ax.plot(
            [], [], color="#58a6ff", lw=0.9, alpha=0.9, label="Contamination index"
        )

        ax.set_xlim(0, _WINDOW_DAYS)
        ax.set_ylim(0.0, 0.45)
        self._title = ax.set_title(
            "Humidity / Contamination — waiting…", color=TEXT, fontsize=9, pad=5
        )
        ax.set_xlabel("Day", color=DIM, fontsize=8)
        ax.set_ylabel("Index (0–1)", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(
            fontsize=7,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            loc="upper left",
        )
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    def push(self, snap: DriverSnapshot) -> None:
        val = snap.humidity_contamination
        self._t_buf.append(float(self._tick))
        self._y_buf.append(val)
        self._tick += 1

        t_arr = np.asarray(self._t_buf)
        y_arr = np.asarray(self._y_buf)
        self._line.set_data(t_arr, y_arr)

        x_end = max(_WINDOW_DAYS, self._tick)
        self._ax.set_xlim(x_end - _WINDOW_DAYS, x_end)

        if val >= HumidityContaminationDriver.CRITICAL:
            label, color = "CRITICAL", RED
        elif val >= HumidityContaminationDriver.WARNING:
            label, color = "WARNING", YELLOW
        else:
            label, color = "nominal", "#58a6ff"
        self._title.set_text(f"Humidity / Contamination — {val:.3f}  [{label}]")
        self._title.set_color(color)
        self.draw_idle()

    def reset(self) -> None:
        self._t_buf.clear()
        self._y_buf.clear()
        self._tick = 0
        self._line.set_data([], [])
        self._ax.set_xlim(0, _WINDOW_DAYS)
        self._title.set_text("Humidity / Contamination — waiting…")
        self._title.set_color(TEXT)
        self.draw_idle()


# ── Printer usage (operational load) chart ────────────────────────────────────


class UsageChart(FigureCanvasQTAgg):
    """Live instantaneous print-rate chart. Driven by update(snap)."""

    def __init__(self):
        fig = Figure(facecolor=PANEL)
        fig.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.90)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._t_buf = deque(maxlen=_WINDOW_DAYS)
        self._y_buf = deque(maxlen=_WINDOW_DAYS)
        self._tick = 0
        self._prev_load = 0.0

        ax = fig.add_subplot(111, facecolor=BG)
        self._ax = ax

        (self._line,) = ax.plot(
            [], [], color="#a371f7", lw=0.9, alpha=0.9, label="Cycles / day"
        )

        ax.set_xlim(0, _WINDOW_DAYS)
        ax.set_ylim(0.0, 2.1)
        self._title = ax.set_title(
            "Printer Usage — waiting…", color=TEXT, fontsize=9, pad=5
        )
        ax.set_xlabel("Day", color=DIM, fontsize=8)
        ax.set_ylabel("Cycles / day", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(
            fontsize=7,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            loc="upper left",
        )
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    def push(self, snap: DriverSnapshot) -> None:
        # Instantaneous rate = change in cumulative load since last step
        rate = snap.operational_load - self._prev_load
        self._prev_load = snap.operational_load

        self._t_buf.append(float(self._tick))
        self._y_buf.append(rate)
        self._tick += 1

        t_arr = np.asarray(self._t_buf)
        y_arr = np.asarray(self._y_buf)
        self._line.set_data(t_arr, y_arr)

        x_end = max(_WINDOW_DAYS, self._tick)
        self._ax.set_xlim(x_end - _WINDOW_DAYS, x_end)

        self._title.set_text(
            f"Printer Usage — {rate:.2f} cycles/day  "
            f"({snap.operational_load:,.0f} total)"
        )
        self._title.set_color(TEXT)
        self.draw_idle()

    def reset(self) -> None:
        self._t_buf.clear()
        self._y_buf.clear()
        self._tick = 0
        self._prev_load = 0.0
        self._line.set_data([], [])
        self._ax.set_xlim(0, _WINDOW_DAYS)
        self._title.set_text("Printer Usage — waiting…")
        self._title.set_color(TEXT)
        self.draw_idle()


# ── Tabbed container ──────────────────────────────────────────────────────────

_TAB_STYLE = f"""
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {PANEL};
}}
QTabBar::tab {{
    background: {BG};
    color: {DIM};
    padding: 5px 18px;
    border: 1px solid {BORDER};
    border-bottom: none;
    margin-right: 2px;
    font-size: 8pt;
}}
QTabBar::tab:selected {{
    background: {PANEL};
    color: {TEXT};
    border-bottom: 2px solid {GREEN};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}
"""


_BTN_STYLE = f"""
QPushButton {{
    background: {BG};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 12px;
    font-size: 8pt;
}}
QPushButton:hover {{ background: {PANEL}; color: {TEXT}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {DIM}; }}
"""


class ChartsPanel(QWidget):
    """Charts + playback controls.

    Owns the shared DriverSuite and QTimer so all charts advance in lockstep.
    Emits snapshot_ready(DriverSnapshot) on every tick for external consumers.

    Speed steps: 0.25×  0.5×  1×  2×  4×  8×
    A fractional accumulator means slow speeds skip ticks smoothly rather than
    firing at a fixed reduced rate.
    """

    snapshot_ready = pyqtSignal(object)
    reset_requested = pyqtSignal()

    _SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    _SPEED_IDX_DEFAULT = 2  # 1×

    def __init__(self, parent=None):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── Tab widget ────────────────────────────────────────────────────────
        tabs = QTabWidget()
        tabs.setStyleSheet(_TAB_STYLE)

        self._temp_chart = TempChart()
        self._humidity_chart = HumidityChart()
        self._usage_chart = UsageChart()

        tabs.addTab(self._temp_chart, "Temperature")
        tabs.addTab(self._humidity_chart, "Humidity")
        tabs.addTab(self._usage_chart, "Printer Usage")

        root.addWidget(tabs)

        # ── Playback controls ─────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(6)

        self._btn_pause = QPushButton("⏸  Pause")
        self._btn_pause.setStyleSheet(_BTN_STYLE)
        self._btn_pause.clicked.connect(self._toggle_pause)

        btn_slow = QPushButton("◀◀  Slower")
        btn_slow.setStyleSheet(_BTN_STYLE)
        btn_slow.clicked.connect(self._slower)

        btn_fast = QPushButton("▶▶  Faster")
        btn_fast.setStyleSheet(_BTN_STYLE)
        btn_fast.clicked.connect(self._faster)

        btn_reset = QPushButton("↺  Reset")
        btn_reset.setStyleSheet(_BTN_STYLE)
        btn_reset.clicked.connect(self._reset)

        self._speed_label = QLabel("1×")
        self._speed_label.setFont(QFont("Segoe UI", 8))
        self._speed_label.setStyleSheet(f"color:{TEXT};")

        bar.addStretch()
        bar.addWidget(btn_slow)
        bar.addWidget(self._btn_pause)
        bar.addWidget(btn_fast)
        bar.addWidget(btn_reset)
        bar.addWidget(self._speed_label)
        bar.addStretch()

        root.addLayout(bar)

        # ── Simulation state ──────────────────────────────────────────────────
        self._suite = DriverSuite(seed=42)
        self._paused = False
        self._speed_idx = self._SPEED_IDX_DEFAULT
        self._tick_acc = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TIMER_MS)

    # ── Playback control slots ────────────────────────────────────────────────

    def pause(self) -> None:
        self._paused = True
        self._btn_pause.setText("▶  Resume")

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._btn_pause.setText("▶  Resume" if self._paused else "⏸  Pause")

    def _slower(self) -> None:
        if self._speed_idx > 0:
            self._speed_idx -= 1
            self._update_speed_label()

    def _faster(self) -> None:
        if self._speed_idx < len(self._SPEEDS) - 1:
            self._speed_idx += 1
            self._update_speed_label()

    def _reset(self) -> None:
        self._suite = DriverSuite(seed=42)
        self._tick_acc = 0.0
        self._paused = False
        self._speed_idx = self._SPEED_IDX_DEFAULT
        self._btn_pause.setText("⏸  Pause")
        self._update_speed_label()
        self._temp_chart.reset()
        self._humidity_chart.reset()
        self._usage_chart.reset()
        self.reset_requested.emit()

    def _update_speed_label(self) -> None:
        spd = self._SPEEDS[self._speed_idx]
        self._speed_label.setText(f"{spd:g}×")

    # ── Tick ──────────────────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        if self._paused:
            return
        self._tick_acc += self._SPEEDS[self._speed_idx]
        n = int(self._tick_acc)
        self._tick_acc -= n
        for _ in range(n):
            snap = self._suite.tick()
            self._temp_chart.push(snap)
            self._humidity_chart.push(snap)
            self._usage_chart.push(snap)
            self.snapshot_ready.emit(snap)
