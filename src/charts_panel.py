from collections import deque

import numpy as np
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QSizePolicy, QTabWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from constants import PANEL, BG, RED, YELLOW, GREEN, TEXT, DIM, BORDER
from input_drivers import DriverSnapshot, DriverSuite, HumidityContaminationDriver
from temp_chart import TempChart, _WINDOW_STEPS

_TIMER_MS = 100


# ── Humidity / Contamination chart ────────────────────────────────────────────

class HumidityChart(FigureCanvasQTAgg):
    """Live humidity & contamination index chart. Driven by update(snap)."""

    def __init__(self):
        fig = Figure(facecolor=PANEL)
        fig.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.90)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._t_buf = deque(maxlen=_WINDOW_STEPS)
        self._y_buf = deque(maxlen=_WINDOW_STEPS)
        self._tick  = 0

        ax = fig.add_subplot(111, facecolor=BG)
        self._ax = ax

        ax.axhspan(0.0, HumidityContaminationDriver.NOMINAL,
                   color=GREEN, alpha=0.07, label="Nominal range")
        ax.axhline(HumidityContaminationDriver.CRITICAL,
                   color=RED, lw=0.9, ls="--", alpha=0.8,
                   label=f"Critical  {HumidityContaminationDriver.CRITICAL:.2f}")
        ax.axhline(HumidityContaminationDriver.WARNING,
                   color=YELLOW, lw=0.9, ls="--", alpha=0.8,
                   label=f"Warning  {HumidityContaminationDriver.WARNING:.2f}")

        (self._line,) = ax.plot([], [], color="#58a6ff", lw=0.9, alpha=0.9,
                                label="Contamination index")

        ax.set_xlim(0, _WINDOW_STEPS)
        ax.set_ylim(0.0, 1.0)
        self._title = ax.set_title("Humidity / Contamination — waiting…",
                                   color=TEXT, fontsize=9, pad=5)
        ax.set_xlabel("Simulation step", color=DIM, fontsize=8)
        ax.set_ylabel("Index (0–1)", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
                  loc="upper left")
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

        x_end = max(_WINDOW_STEPS, self._tick)
        self._ax.set_xlim(x_end - _WINDOW_STEPS, x_end)

        if val >= HumidityContaminationDriver.CRITICAL:
            label, color = "CRITICAL", RED
        elif val >= HumidityContaminationDriver.WARNING:
            label, color = "WARNING", YELLOW
        else:
            label, color = "nominal", "#58a6ff"
        self._title.set_text(
            f"Humidity / Contamination — {val:.3f}  [{label}]")
        self._title.set_color(color)
        self.draw_idle()


# ── Printer usage (operational load) chart ────────────────────────────────────

class UsageChart(FigureCanvasQTAgg):
    """Live instantaneous print-rate chart. Driven by update(snap)."""

    def __init__(self):
        fig = Figure(facecolor=PANEL)
        fig.subplots_adjust(left=0.14, bottom=0.14, right=0.97, top=0.90)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._t_buf    = deque(maxlen=_WINDOW_STEPS)
        self._y_buf    = deque(maxlen=_WINDOW_STEPS)
        self._tick     = 0
        self._prev_load = 0.0

        ax = fig.add_subplot(111, facecolor=BG)
        self._ax = ax

        ax.axhline(1.0, color=DIM, lw=0.7, ls="--", alpha=0.5, label="Baseline (1.0)")

        (self._line,) = ax.plot([], [], color="#a371f7", lw=0.9, alpha=0.9,
                                label="Cycles / step")

        ax.set_xlim(0, _WINDOW_STEPS)
        ax.set_ylim(0.0, 2.2)
        self._title = ax.set_title("Printer Usage — waiting…",
                                   color=TEXT, fontsize=9, pad=5)
        ax.set_xlabel("Simulation step", color=DIM, fontsize=8)
        ax.set_ylabel("Cycles / step", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
                  loc="upper left")
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

        x_end = max(_WINDOW_STEPS, self._tick)
        self._ax.set_xlim(x_end - _WINDOW_STEPS, x_end)

        self._title.set_text(
            f"Printer Usage — {rate:.2f} cycles/step  "
            f"({snap.operational_load:,.0f} total)"
        )
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


class ChartsPanel(QTabWidget):
    """Tabbed widget holding all three live driver charts.

    Owns the shared DriverSuite and QTimer so all charts advance in lockstep.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_TAB_STYLE)

        self._temp_chart     = TempChart()
        self._humidity_chart = HumidityChart()
        self._usage_chart    = UsageChart()

        self.addTab(self._temp_chart,     "Temperature")
        self.addTab(self._humidity_chart, "Humidity")
        self.addTab(self._usage_chart,    "Printer Usage")

        self._suite = DriverSuite(seed=42)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start(_TIMER_MS)

    def _on_tick(self) -> None:
        snap = self._suite.tick()
        self._temp_chart.push(snap)
        self._humidity_chart.push(snap)
        self._usage_chart.push(snap)
