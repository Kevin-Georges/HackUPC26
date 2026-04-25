#!/usr/bin/env python3
"""HP Metal Jet S100 — Digital Twin Dashboard (Phase 1 scaffold)"""

import sys
import os
import warnings
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QFrame, QSizePolicy, QGridLayout, QOpenGLWidget,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QSurfaceFormat
from OpenGL.GL import *   # noqa: F401,F403  – legacy pipeline; wildcard avoids missed symbols
from OpenGL.GLU import gluPerspective
import trimesh
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# ── Palette ──────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"
DIM    = "#8b949e"
GREEN  = "#3fb950"
YELLOW = "#d29922"
RED    = "#f85149"


# ── 3-D wireframe viewer ─────────────────────────────────────────────────────
class ModelViewer(QOpenGLWidget):
    """Renders a GLB mesh as a green wireframe.

    Left-drag  → rotate around X/Y
    Right-drag → rotate around Z
    Scroll     → zoom
    """

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.rot_x  = 20.0
        self.rot_y  = 30.0
        self.rot_z  = 0.0
        self._dist  = 5.0
        self._last  = None
        self._ready = False
        self._vbo   = 0
        self._n_draw = 0
        self.setMinimumSize(400, 400)

    # ── OpenGL lifecycle ──────────────────────────────────────────────────────
    def initializeGL(self):
        glClearColor(0.05, 0.07, 0.09, 1.0)
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        self._load_model()

    def _load_model(self):
        try:
            # force="mesh" applies all scene-graph transforms before concatenating,
            # so the bounding box and centroid are correct in world space.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh = trimesh.load(self.model_path, force="mesh")

            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                print("[ModelViewer] not a valid Trimesh after load")
                return

            c = mesh.centroid
            s = float(mesh.extents.max()) or 1.0

            # Pre-expand triangles into a flat (N*3, 3) float32 array so we
            # need only one VBO and a plain glDrawArrays call — no index buffer.
            flat = ((mesh.vertices[mesh.faces.reshape(-1)] - c) / s * 1.6).astype(np.float32)
            self._n_draw = len(flat)

            self._vbo = int(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            self._ready = True
            print(f"[ModelViewer] {len(mesh.vertices):,} verts  "
                  f"{len(mesh.faces):,} faces  extents={mesh.extents.round(2)}")
        except Exception as exc:
            import traceback
            print(f"[ModelViewer] load error: {exc}")
            traceback.print_exc()

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, max(h, 1))
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / max(h, 1), 0.1, 200.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -self._dist)
        glRotatef(self.rot_x, 1, 0, 0)
        glRotatef(self.rot_y, 0, 1, 0)
        glRotatef(self.rot_z, 0, 0, 1)

        if self._ready:
            glColor3f(0.18, 0.63, 0.30)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawArrays(GL_TRIANGLES, 0, self._n_draw)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_VERTEX_ARRAY)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            # Fallback: coloured axes so the panel isn't black on load failure
            glBegin(GL_LINES)
            glColor3f(1.0, 0.2, 0.2); glVertex3f(0, 0, 0); glVertex3f(1, 0, 0)
            glColor3f(0.2, 1.0, 0.2); glVertex3f(0, 0, 0); glVertex3f(0, 1, 0)
            glColor3f(0.2, 0.2, 1.0); glVertex3f(0, 0, 0); glVertex3f(0, 0, 1)
            glEnd()

    # ── mouse / wheel interaction ─────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._last = e.pos()

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        dx = e.x() - self._last.x()
        dy = e.y() - self._last.y()
        if e.buttons() & Qt.LeftButton:
            self.rot_y += dx * 0.4
            self.rot_x += dy * 0.4
        if e.buttons() & Qt.RightButton:
            self.rot_z += dx * 0.4
        self._last = e.pos()
        self.update()

    def mouseReleaseEvent(self, e):
        self._last = None

    def wheelEvent(self, e):
        self._dist = max(1.5, min(20.0, self._dist - e.angleDelta().y() / 240.0))
        self.update()


# ── placeholder temperature chart ────────────────────────────────────────────
class TempChart(FigureCanvasQTAgg):
    def __init__(self):
        fig = Figure(facecolor=PANEL, tight_layout=True)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        rng = np.random.default_rng(42)
        t   = np.linspace(0, 120, 300)
        tmp = 195 + 12 * np.sin(t / 18) + 4 * np.sin(t / 5) + rng.normal(0, 1.5, 300)

        ax = fig.add_subplot(111, facecolor=BG)
        ax.plot(t, tmp, color="#ff7043", lw=1.2, label="Nozzle temp")
        ax.axhline(220, color=RED,    lw=0.8, ls="--", alpha=0.7, label="Max limit")
        ax.axhline(180, color=YELLOW, lw=0.8, ls="--", alpha=0.7, label="Min limit")
        ax.set_title("Build Temperature  [placeholder — live data in Phase 2]",
                     color=TEXT, fontsize=9, pad=5)
        ax.set_xlabel("Time (s)", color=DIM, fontsize=8)
        ax.set_ylabel("°C",       color=DIM, fontsize=8)
        ax.tick_params(colors=DIM, labelsize=7)
        ax.legend(fontsize=7, facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)


# ── placeholder component-health table ───────────────────────────────────────
class HealthPanel(QFrame):
    _ROWS = [
        ("Recoater Blade",      "FUNCTIONAL", "88%", GREEN),
        ("Drive Motor",         "FUNCTIONAL", "91%", GREEN),
        ("Nozzle Plate",        "DEGRADED",   "61%", YELLOW),
        ("Heating Elements",    "FUNCTIONAL", "79%", GREEN),
        ("Temperature Sensors", "FUNCTIONAL", "95%", GREEN),
        ("Insulation Panels",   "CRITICAL",   "33%", RED),
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

        for row, (name, status, pct, color) in enumerate(self._ROWS, 1):
            for col, (val, style) in enumerate([
                (name,   f"color:{TEXT};"),
                (status, f"color:{color}; font-weight:bold;"),
                (pct,    f"color:{color};"),
            ]):
                lbl = QLabel(val)
                lbl.setFont(QFont("Segoe UI", 8))
                lbl.setStyleSheet(style + " border:none;")
                grid.addWidget(lbl, row, col)

        lay.addLayout(grid)


# ── main window ───────────────────────────────────────────────────────────────
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

        tip = QLabel("Left-drag · rotate X/Y     Right-drag · rotate Z     Scroll · zoom")
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

        chart_frame = QFrame()
        chart_frame.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;"
        )
        QVBoxLayout(chart_frame).addWidget(TempChart())
        right.addWidget(chart_frame, stretch=2)

        health = HealthPanel()
        health.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;"
        )
        right.addWidget(health, stretch=1)

        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, stretch=2)


# ── entry point ───────────────────────────────────────────────────────────────
def main():
    # Request a compatibility-profile context so legacy GL functions work
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    here  = os.path.dirname(os.path.abspath(__file__))
    model = os.path.join(here, "Proper HP Printer.glb")
    win   = MainWindow(model)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
