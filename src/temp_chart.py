import numpy as np
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from constants import PANEL, BG, RED, YELLOW, TEXT, DIM, BORDER


class TempChart(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(facecolor=PANEL, tight_layout=True)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        rng = np.random.default_rng(42)
        t = np.linspace(0, 120, 300)
        tmp = 195 + 12 * np.sin(t / 18) + 4 * np.sin(t / 5) + rng.normal(0, 1.5, 300)

        ax = fig.add_subplot(111, facecolor=BG)
        ax.plot(t, tmp, color="#ff7043", lw=1.2, label="Nozzle temp")
        ax.axhline(220, color=RED, lw=0.8, ls="--", alpha=0.7, label="Max limit")
        ax.axhline(180, color=YELLOW, lw=0.8, ls="--", alpha=0.7, label="Min limit")
        ax.set_title(
            "Build Temperature  [placeholder — live data in Phase 2]",
            color=TEXT,
            fontsize=9,
            pad=5,
        )
        ax.set_xlabel("Time (s)", color=DIM, fontsize=8)
        ax.set_ylabel("°C", color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
