"""RUL Panel — two-tab widget in the top-left dashboard area.

  Predict tab  — enter temperature / humidity / print volume, get a table
                 of predicted days-to-degradation for every component.
  Heatmap tab  — 2-D sweep across any two variables for one component.

Requires ml/models/rul_model.joblib (run: python -m ml.train).
"""
import pathlib
import sys
import numpy as np

from PyQt5.QtWidgets import (
    QFrame, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QSlider, QSizePolicy, QGridLayout, QScrollArea,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from constants import PANEL, BORDER, TEXT, DIM, GREEN, YELLOW, ORANGE, RED

# ── paths ─────────────────────────────────────────────────────────────────────
_MODEL_PATH = pathlib.Path(__file__).parent.parent / "ml" / "models" / "rul_model.joblib"
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── feature metadata ──────────────────────────────────────────────────────────
_FEAT_LABELS = {
    "temperature_c":           "Temperature (°C)",
    "humidity_index":          "Humidity",
    "operational_load":        "Print Volume",
    "maintenance_level":       "Maintenance",
    "powder_quality":          "Powder Quality",
    "binder_viscosity_stress": "Binder Viscosity",
    "voltage_stress":          "Voltage Stress",
    "health_pct":              "Health (%)",
}

_ENV_FEATURES = list(_FEAT_LABELS.keys())   # excludes health_pct / component_id

_RANGES = {
    "temperature_c":           (15.0,  35.0),
    "humidity_index":          (0.0,   1.0),
    "operational_load":        (0.0,   3.0),
    "maintenance_level":       (0.0,   1.0),
    "powder_quality":          (0.5,   1.0),
    "binder_viscosity_stress": (0.0,   0.5),
    "voltage_stress":          (0.0,   0.5),
    "health_pct":              (1.0,   100.0),
}

_DEFAULTS = {
    "temperature_c":           21.0,
    "humidity_index":          0.3,
    "operational_load":        1.5,
    "maintenance_level":       0.8,
    "powder_quality":          0.9,
    "binder_viscosity_stress": 0.1,
    "voltage_stress":          0.1,
    "health_pct":              100.0,
}

_GRID_SIZE = 40

_DARK_STYLE = f"""
    QWidget  {{ background: {PANEL}; color: {TEXT}; border: none; }}
    QTabWidget::pane  {{ border: 1px solid {BORDER}; border-radius: 3px; }}
    QTabBar::tab {{
        background: #1f2937; color: {DIM};
        padding: 3px 10px; border-radius: 3px 3px 0 0;
        font-size: 10px;
    }}
    QTabBar::tab:selected {{ background: {PANEL}; color: {TEXT}; }}
    QComboBox {{
        background: #1f2937; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 3px;
        padding: 1px 4px; font-size: 10px;
    }}
    QComboBox QAbstractItemView {{ background: #1f2937; color: {TEXT}; }}
    QSlider::groove:horizontal {{
        height: 4px; background: {BORDER}; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 10px; height: 10px; margin: -3px 0;
        background: #4a9eff; border-radius: 5px;
    }}
    QSlider::sub-page:horizontal {{ background: #4a9eff; border-radius: 2px; }}
    QScrollArea {{ border: none; }}
"""


# ── shared slider widget ───────────────────────────────────────────────────────

class _SliderRow(QWidget):
    def __init__(self, feat: str, label_width: int = 110, parent=None):
        super().__init__(parent)
        self.feat = feat
        lo, hi = _RANGES[feat]
        self._lo, self._hi = lo, hi

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        lbl = QLabel(_FEAT_LABELS[feat])
        lbl.setFont(QFont("Segoe UI", 7))
        lbl.setStyleSheet(f"color:{DIM};")
        lbl.setFixedWidth(label_width)
        row.addWidget(lbl)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 200)
        self._slider.setValue(self._to_slider(_DEFAULTS[feat]))
        self._slider.setFixedHeight(14)
        row.addWidget(self._slider, stretch=1)

        self._val_lbl = QLabel(self._fmt(_DEFAULTS[feat]))
        self._val_lbl.setFont(QFont("Segoe UI", 7))
        self._val_lbl.setStyleSheet(f"color:{TEXT};")
        self._val_lbl.setFixedWidth(38)
        self._val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self._val_lbl)

        self._slider.valueChanged.connect(self._on_change)

    def _to_slider(self, v: float) -> int:
        return round((v - self._lo) / (self._hi - self._lo) * 200)

    def value(self) -> float:
        return self._lo + self._slider.value() / 200.0 * (self._hi - self._lo)

    def _fmt(self, v: float) -> str:
        span = self._hi - self._lo
        if span > 20:  return f"{v:.0f}"
        if span > 2:   return f"{v:.2f}"
        return f"{v:.3f}"

    def _on_change(self, _):
        self._val_lbl.setText(self._fmt(self.value()))

    def connect_changed(self, slot):
        self._slider.valueChanged.connect(slot)


# ── Predict tab ───────────────────────────────────────────────────────────────

class _PredictTab(QWidget):
    """Three primary sliders + optional advanced sliders → per-component RUL table."""

    _PRIMARY = ["temperature_c", "humidity_index", "operational_load"]
    _ADVANCED = ["maintenance_level", "powder_quality", "binder_viscosity_stress", "voltage_stress"]

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model
        self._health: dict[str, int] = {}   # updated from live simulation

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # ── primary sliders ───────────────────────────────────────────────────
        self._primary_sliders: dict[str, _SliderRow] = {}
        for feat in self._PRIMARY:
            sr = _SliderRow(feat)
            sr.connect_changed(self._schedule_predict)
            self._primary_sliders[feat] = sr
            root.addWidget(sr)

        # ── advanced toggle ───────────────────────────────────────────────────
        adv_toggle = QLabel("▸ Advanced")
        adv_toggle.setFont(QFont("Segoe UI", 7))
        adv_toggle.setStyleSheet(f"color:{DIM}; cursor: pointer;")
        adv_toggle.setCursor(Qt.PointingHandCursor)
        root.addWidget(adv_toggle)

        self._adv_container = QWidget()
        adv_lay = QVBoxLayout(self._adv_container)
        adv_lay.setContentsMargins(0, 0, 0, 0)
        adv_lay.setSpacing(2)
        self._adv_sliders: dict[str, _SliderRow] = {}
        for feat in self._ADVANCED:
            sr = _SliderRow(feat)
            sr.connect_changed(self._schedule_predict)
            self._adv_sliders[feat] = sr
            adv_lay.addWidget(sr)
        self._adv_container.setVisible(False)
        root.addWidget(self._adv_container)

        def _toggle(_):
            visible = not self._adv_container.isVisible()
            self._adv_container.setVisible(visible)
            adv_toggle.setText("▾ Advanced" if visible else "▸ Advanced")
        adv_toggle.mousePressEvent = _toggle

        # ── results table ─────────────────────────────────────────────────────
        sep = QLabel()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{BORDER};")
        root.addWidget(sep)

        headers = QGridLayout()
        headers.setContentsMargins(0, 2, 0, 0)
        headers.setHorizontalSpacing(8)
        for col, txt in enumerate(("Component", "Health", "Days to Degradation")):
            h = QLabel(txt)
            h.setFont(QFont("Segoe UI", 7, QFont.Bold))
            h.setStyleSheet(f"color:{DIM};")
            headers.addWidget(h, 0, col)
        headers.setColumnStretch(2, 1)
        root.addLayout(headers)

        self._row_widgets: list[tuple[QLabel, QLabel, QLabel]] = []
        from ml.data import COMPONENT_LABELS
        self._component_labels = COMPONENT_LABELS
        for i, name in enumerate(COMPONENT_LABELS):
            name_lbl = QLabel(name)
            name_lbl.setFont(QFont("Segoe UI", 8))
            name_lbl.setStyleSheet(f"color:{TEXT};")

            health_lbl = QLabel("—")
            health_lbl.setFont(QFont("Segoe UI", 8, QFont.Bold))
            health_lbl.setAlignment(Qt.AlignCenter)

            rul_lbl = QLabel("—")
            rul_lbl.setFont(QFont("Segoe UI", 8))

            self._row_widgets.append((name_lbl, health_lbl, rul_lbl))

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        for i, (n, h, r) in enumerate(self._row_widgets):
            grid.addWidget(n, i, 0)
            grid.addWidget(h, i, 1)
            grid.addWidget(r, i, 2)
        grid.setColumnStretch(2, 1)
        root.addLayout(grid)
        root.addStretch()

        # ── debounce timer ────────────────────────────────────────────────────
        self._day = 0

        self._day_lbl = QLabel("Simulation day: 0")
        self._day_lbl.setFont(QFont("Segoe UI", 7))
        self._day_lbl.setStyleSheet(f"color:{DIM};")
        root.addWidget(self._day_lbl)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._run_predict)
        self._run_predict()

    def update_health(self, rows: list[tuple[str, str]], day: int = 0) -> None:
        """Accept current health from the live simulation."""
        self._health = {name: int(pct.rstrip("%")) for name, pct in rows}
        self._day = day
        self._update_day_label()
        self._schedule_predict()

    def _update_day_label(self):
        self._day_lbl.setText(f"Simulation day: {self._day}  —  health reflects current state")

    def _schedule_predict(self, *_):
        self._timer.start()

    def _run_predict(self):
        if self._model is None:
            for n, h, r in self._row_widgets:
                r.setText("Model not trained")
                r.setStyleSheet(f"color:{DIM};")
            return

        from ml.data import ALL_FEATURES, COMPONENT_LABELS
        feat_vals = {}
        for feat in self._PRIMARY:
            feat_vals[feat] = self._primary_sliders[feat].value()
        for feat in self._ADVANCED:
            feat_vals[feat] = self._adv_sliders[feat].value()

        for i, name in enumerate(COMPONENT_LABELS):
            health = self._health.get(name, 100)
            feat_vals["health_pct"] = float(health)
            feat_vals["component_id"] = float(i)

            X = np.array([[feat_vals[f] for f in ALL_FEATURES]], dtype=np.float32)
            try:
                rul = int(round(float(self._model.predict(X)[0])))
            except Exception:
                rul = -1

            n_lbl, h_lbl, r_lbl = self._row_widgets[i]
            h_lbl.setText(f"{health}%")
            h_lbl.setStyleSheet(f"color:{self._health_color(health)}; font-weight:bold;")

            if rul < 0:
                r_lbl.setText("error")
                r_lbl.setStyleSheet(f"color:{DIM};")
            elif rul == 0:
                r_lbl.setText(f"Already degrading ({health}%)")
                r_lbl.setStyleSheet(f"color:{RED};")
            elif rul <= 7:
                r_lbl.setText(f"{rul} day{'s' if rul != 1 else ''}")
                r_lbl.setStyleSheet(f"color:{ORANGE};")
            elif rul <= 30:
                r_lbl.setText(f"{rul} days")
                r_lbl.setStyleSheet(f"color:{YELLOW};")
            else:
                r_lbl.setText(f"{rul} days")
                r_lbl.setStyleSheet(f"color:{GREEN};")

    @staticmethod
    def _health_color(pct: int) -> str:
        if pct > 70: return GREEN
        if pct > 50: return YELLOW
        if pct > 30: return ORANGE
        return RED


# ── Heatmap tab ───────────────────────────────────────────────────────────────

class _HeatmapTab(QWidget):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self._model = model

        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(150)
        self._render_timer.timeout.connect(self._render)

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(3)

        # ── control row ───────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        def _lbl(t):
            l = QLabel(t)
            l.setFont(QFont("Segoe UI", 7))
            l.setStyleSheet(f"color:{DIM};")
            return l

        ctrl.addWidget(_lbl("Component:"))
        self._cmp_box = QComboBox()
        self._cmp_box.setFixedWidth(130)
        from ml.data import COMPONENT_LABELS
        for lbl in COMPONENT_LABELS:
            self._cmp_box.addItem(lbl)
        ctrl.addWidget(self._cmp_box)

        ctrl.addSpacing(6)
        ctrl.addWidget(_lbl("X:"))
        self._x_box = QComboBox()
        self._x_box.setFixedWidth(110)
        ctrl.addWidget(self._x_box)

        ctrl.addWidget(_lbl("Y:"))
        self._y_box = QComboBox()
        self._y_box.setFixedWidth(110)
        ctrl.addWidget(self._y_box)
        ctrl.addStretch()
        root.addLayout(ctrl)

        for feat in _ENV_FEATURES:
            self._x_box.addItem(_FEAT_LABELS[feat], feat)
            self._y_box.addItem(_FEAT_LABELS[feat], feat)
        self._x_box.setCurrentIndex(0)
        self._y_box.setCurrentIndex(1)

        # ── fixed-variable sliders ────────────────────────────────────────────
        self._sliders: dict[str, _SliderRow] = {}
        self._slider_container = QWidget()
        sl = QVBoxLayout(self._slider_container)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(1)
        for feat in _ENV_FEATURES:
            sr = _SliderRow(feat)
            sr.connect_changed(self._schedule_render)
            self._sliders[feat] = sr
            sl.addWidget(sr)
        root.addWidget(self._slider_container)

        # ── canvas ────────────────────────────────────────────────────────────
        self._fig, self._ax = plt.subplots(figsize=(4, 2.8))
        self._fig.patch.set_facecolor("#0d1117")
        self._ax.set_facecolor("#161b22")
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._canvas, stretch=1)

        self._cmp_box.currentIndexChanged.connect(self._schedule_render)
        self._x_box.currentIndexChanged.connect(self._on_axes_changed)
        self._y_box.currentIndexChanged.connect(self._on_axes_changed)
        self._on_axes_changed()

    def _on_axes_changed(self):
        x_feat = self._x_box.currentData()
        y_feat = self._y_box.currentData()
        for feat, sr in self._sliders.items():
            sr.setVisible(feat != x_feat and feat != y_feat)
        self._schedule_render()

    def _schedule_render(self, *_):
        self._render_timer.start()

    def _render(self):
        self._fig.clf()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor("#161b22")

        if self._model is None:
            self._ax.text(0.5, 0.5, "Model not trained.\nRun:  python -m ml.train",
                          ha="center", va="center", transform=self._ax.transAxes,
                          color="#8b949e", fontsize=8)
            self._canvas.draw()
            return

        x_feat = self._x_box.currentData()
        y_feat = self._y_box.currentData()
        comp_idx = self._cmp_box.currentIndex()
        if x_feat is None or y_feat is None or x_feat == y_feat:
            self._canvas.draw()
            return

        x_lo, x_hi = _RANGES[x_feat]
        y_lo, y_hi = _RANGES[y_feat]
        xs = np.linspace(x_lo, x_hi, _GRID_SIZE)
        ys = np.linspace(y_lo, y_hi, _GRID_SIZE)
        XX, YY = np.meshgrid(xs, ys)
        n = _GRID_SIZE * _GRID_SIZE

        from ml.data import ALL_FEATURES
        feat_vals = {}
        for feat in ALL_FEATURES[:-1]:
            if feat == x_feat:
                feat_vals[feat] = XX.ravel()
            elif feat == y_feat:
                feat_vals[feat] = YY.ravel()
            else:
                feat_vals[feat] = np.full(n, self._sliders[feat].value())

        rows = [feat_vals[f] for f in ALL_FEATURES[:-1]]
        rows.append(np.full(n, float(comp_idx)))
        X_grid = np.column_stack(rows).astype(np.float32)

        try:
            Z = self._model.predict(X_grid).reshape(_GRID_SIZE, _GRID_SIZE)
        except Exception as exc:
            print(f"[Heatmap] predict error: {exc}")
            self._canvas.draw()
            return

        im = self._ax.imshow(Z, origin="lower", aspect="auto",
                             extent=[x_lo, x_hi, y_lo, y_hi],
                             cmap="RdYlGn", vmin=0, vmax=max(1, Z.max()))
        self._ax.set_title(f"Predicted RUL — {self._cmp_box.currentText()}",
                           color=TEXT, fontsize=7, pad=3)
        self._ax.set_xlabel(_FEAT_LABELS[x_feat], color="#8b949e", fontsize=6)
        self._ax.set_ylabel(_FEAT_LABELS[y_feat], color="#8b949e", fontsize=6)
        self._ax.tick_params(colors="#8b949e", labelsize=5)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)

        cb = self._fig.colorbar(im, ax=self._ax, pad=0.02)
        cb.ax.tick_params(colors="#8b949e", labelsize=5)
        cb.set_label("RUL (days)", color="#8b949e", fontsize=5)
        cb.outline.set_edgecolor(BORDER)

        self._fig.tight_layout(pad=0.4)
        self._canvas.draw()


# ── Public panel ──────────────────────────────────────────────────────────────

class RULHeatmapPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_DARK_STYLE)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._model = self._load_model()

        tabs = QTabWidget()
        tabs.setDocumentMode(True)

        self._predict_tab = _PredictTab(self._model)
        self._heatmap_tab = _HeatmapTab(self._model)

        tabs.addTab(self._predict_tab, "Predict")
        tabs.addTab(self._heatmap_tab, "Heatmap")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(tabs)

    def update_health(self, rows: list[tuple[str, str]], day: int = 0) -> None:
        """Called by MainWindow on each simulation tick with current health."""
        self._predict_tab.update_health(rows, day)

    @staticmethod
    def _load_model():
        try:
            from ml.model import RULModel
            return RULModel.load(_MODEL_PATH)
        except FileNotFoundError:
            return None
        except Exception as exc:
            print(f"[RULPanel] model load error: {exc}")
            return None
