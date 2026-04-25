from collections import deque

import numpy as np
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from constants import PANEL, BG, RED, YELLOW, GREEN, TEXT, DIM, BORDER
from input_drivers import DriverSnapshot, TemperatureStressDriver

_WINDOW_STEPS = 300


class TempChart(FigureCanvasQTAgg):
    """Live temperature chart. Driven externally via update(snap)."""

    def __init__(self):
        fig = Figure(facecolor=PANEL)
        fig.subplots_adjust(left=0.12, bottom=0.14, right=0.97, top=0.90)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._t_buf   = deque(maxlen=_WINDOW_STEPS)
        self._tmp_buf = deque(maxlen=_WINDOW_STEPS)
        self._tick    = 0

        ax = fig.add_subplot(111, facecolor=BG)
        self._ax = ax

        ax.axhspan(
            TemperatureStressDriver.OPTIMAL_LOW,
            TemperatureStressDriver.OPTIMAL_HIGH,
            color=GREEN, alpha=0.07, label="Optimal range",
        )
        ax.axhline(
            TemperatureStressDriver.CRITICAL,
            color=RED, lw=0.9, ls="--", alpha=0.8,
            label=f"Critical  {TemperatureStressDriver.CRITICAL:.0f} °C",
        )
        ax.axhline(
            TemperatureStressDriver.WARNING,
            color=YELLOW, lw=0.9, ls="--", alpha=0.8,
            label=f"Warning  {TemperatureStressDriver.WARNING:.0f} °C",
        )
        (self._line,) = ax.plot([], [], color="#ff7043", lw=0.9, alpha=0.9,
                                label="Ambient temp")

        ax.set_xlim(0, _WINDOW_STEPS)
        ax.set_ylim(10, 55)
        self._title = ax.set_title("Temperature Stress — waiting…", color=TEXT,
                                   fontsize=9, pad=5)
        ax.set_xlabel("Simulation step", color=DIM, fontsize=8)
        ax.set_ylabel("°C", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT,
                  loc="upper left")
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    def push(self, snap: DriverSnapshot) -> None:
        temp = snap.temperature_stress
        self._t_buf.append(float(self._tick))
        self._tmp_buf.append(temp)
        self._tick += 1

        t_arr = np.asarray(self._t_buf)
        y_arr = np.asarray(self._tmp_buf)
        self._line.set_data(t_arr, y_arr)

        x_end = max(_WINDOW_STEPS, self._tick)
        self._ax.set_xlim(x_end - _WINDOW_STEPS, x_end)

        if temp >= TemperatureStressDriver.CRITICAL:
            label, color = "CRITICAL", RED
        elif temp >= TemperatureStressDriver.WARNING:
            label, color = "WARNING", YELLOW
        else:
            label, color = "nominal", "#ff7043"
        self._title.set_text(f"Temperature Stress — {temp:.1f} °C  [{label}]")
        self._title.set_color(color)
        self.draw_idle()
