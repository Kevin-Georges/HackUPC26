import colorsys
import warnings
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GLU import gluPerspective
import trimesh
from constants import BLUE

# ── GLSL sources ──────────────────────────────────────────────────────────────
_VERT_SRC = """
#version 120
attribute vec3 aPos;
void main() {
    gl_Position = gl_ModelViewProjectionMatrix * vec4(aPos, 1.0);
}
"""

_FRAG_SRC = """
#version 120
uniform bool  uUseHighlight;
uniform vec3  uHighlightColor;
uniform vec3  uDefaultColor;
void main() {
    if (uUseHighlight)
        gl_FragColor = vec4(uHighlightColor, 1.0);
    else
        gl_FragColor = vec4(uDefaultColor, 1.0);
}
"""

# ── Name-translation: health-panel label → glb node name ─────────────────────
_LABEL_TO_NODE: dict[str, str] = {
    "Nozzle Plate":      "Nozzle Plates",
    "Recoater Blade":    "Recoater Roller",
    "Heating Elements":  "Heating Lamps",
    "Vacuum Pump":       "Vacuum Pump",
    "Vacuum Area":       "Vacuum Area",
}



def _pct_to_rgb(pct_str: str) -> tuple[float, float, float]:
    try:
        value = int(pct_str.rstrip("%"))
    except ValueError:
        return colorsys.hsv_to_rgb(0.0, 1.0, 1.0)
    t = max(0.0, min(1.0, value / 100.0))
    hue = t / 3.0  # 0.0 = red, 1/3 = green
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


def build_color_map(health_rows) -> dict[str, tuple[float, float, float]]:
    """Convert HealthPanel._ROWS into {glb_node_name: (r, g, b)}."""
    result = {}
    for label, pct in health_rows:
        node = _LABEL_TO_NODE.get(label)
        if node:
            result[node] = _pct_to_rgb(pct)
    return result


class ModelViewer(QOpenGLWidget):
    """Renders a GLB mesh.

    Non-highlighted nodes → blue wireframe.
    Highlighted nodes     → solid fill in status color.

    Left-drag  → rotate X/Y
    Right-drag → rotate Z
    Scroll     → zoom
    """

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.rot_x = 20.0
        self.rot_y = 30.0
        self.rot_z = 0.0
        self._dist = 5.0
        self._last = None
        self._ready = False
        self._meshes: list[tuple[int, int, str]] = []  # (vbo, n_verts, node_name)
        self._prog = 0
        self._component_colors: dict[str, tuple[float, float, float]] = {}
        self.setMinimumSize(400, 400)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._auto_rotate_step)

        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(2000)
        self._idle_timer.timeout.connect(self._anim_timer.start)
        self._idle_timer.start()

    def set_component_colors(self, colors: dict[str, tuple[float, float, float]]):
        self._component_colors = colors
        self.update()

    # ── GL lifecycle ──────────────────────────────────────────────────────────

    def initializeGL(self):
        glClearColor(0.05, 0.07, 0.09, 1.0)
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        self._build_shader()
        self._load_model()

    def _build_shader(self):
        def compile_shader(src, kind):
            sh = glCreateShader(kind)
            glShaderSource(sh, src)
            glCompileShader(sh)
            if not glGetShaderiv(sh, GL_COMPILE_STATUS):
                raise RuntimeError(glGetShaderInfoLog(sh).decode())
            return sh

        vert = compile_shader(_VERT_SRC, GL_VERTEX_SHADER)
        frag = compile_shader(_FRAG_SRC, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vert)
        glAttachShader(prog, frag)
        glBindAttribLocation(prog, 0, "aPos")
        glLinkProgram(prog)
        if not glGetProgramiv(prog, GL_LINK_STATUS):
            raise RuntimeError(glGetProgramInfoLog(prog).decode())
        glDeleteShader(vert)
        glDeleteShader(frag)
        self._prog = prog

    def _load_model(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scene = trimesh.load(self.model_path)

            if isinstance(scene, trimesh.Trimesh):
                meshes_by_node: dict[str, trimesh.Trimesh] = {"": scene}
            elif isinstance(scene, trimesh.Scene):
                meshes_by_node = {}
                for node_name in scene.graph.nodes_geometry:
                    transform, geo_name = scene.graph[node_name]
                    if geo_name is None or geo_name not in scene.geometry:
                        continue
                    transformed = scene.geometry[geo_name].copy()
                    transformed.apply_transform(transform)
                    meshes_by_node[node_name] = transformed
            else:
                print(f"[ModelViewer] unexpected type: {type(scene)}")
                return

            valid = {k: v for k, v in meshes_by_node.items() if len(v.faces) > 0}
            if not valid:
                print("[ModelViewer] no geometry found")
                return

            all_verts = np.concatenate([m.vertices for m in valid.values()])
            c = all_verts.mean(axis=0)
            s = float((all_verts.max(axis=0) - all_verts.min(axis=0)).max()) or 1.0

            for node_name, mesh in valid.items():
                flat = (
                    (mesh.vertices[mesh.faces.reshape(-1)] - c) / s * 1.6
                ).astype(np.float32)
                vbo = int(glGenBuffers(1))
                glBindBuffer(GL_ARRAY_BUFFER, vbo)
                glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
                glBindBuffer(GL_ARRAY_BUFFER, 0)
                self._meshes.append((vbo, len(flat), node_name))

            self._ready = True
            print(
                f"[ModelViewer] {len(valid)} nodes loaded  "
                f"({sum(len(m.faces) for m in valid.values()):,} faces total)"
            )
            print(f"[ModelViewer] node names: {sorted(valid.keys())}")
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
        glTranslatef(0.4, 0.0, 0.0)

        if not self._ready:
            glBegin(GL_LINES)
            glColor3f(1.0, 0.2, 0.2)
            glVertex3f(0, 0, 0)
            glVertex3f(1, 0, 0)
            glColor3f(0.2, 1.0, 0.2)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 1, 0)
            glColor3f(0.2, 0.2, 1.0)
            glVertex3f(0, 0, 0)
            glVertex3f(0, 0, 1)
            glEnd()
            return

        glUseProgram(self._prog)
        loc_use = glGetUniformLocation(self._prog, "uUseHighlight")
        loc_col = glGetUniformLocation(self._prog, "uHighlightColor")
        loc_def = glGetUniformLocation(self._prog, "uDefaultColor")
        glUniform3f(loc_def, *BLUE)

        for vbo, n_verts, node_name in self._meshes:
            rgb = self._component_colors.get(node_name)
            if rgb is not None:
                glUniform1i(loc_use, 1)
                glUniform3f(loc_col, *rgb)
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            else:
                glUniform1i(loc_use, 0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

            glEnableVertexAttribArray(0)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
            glDrawArrays(GL_TRIANGLES, 0, n_verts)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableVertexAttribArray(0)

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glUseProgram(0)

    # ── Input ─────────────────────────────────────────────────────────────────

    def _reset_idle(self) -> None:
        self._anim_timer.stop()
        self._idle_timer.start(2000)

    def _auto_rotate_step(self) -> None:
        self.rot_y += 0.3
        self.update()

    def mousePressEvent(self, e):
        self._reset_idle()
        self._last = e.pos()

    def mouseMoveEvent(self, e):
        if self._last is None:
            return
        self._reset_idle()
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
        self._reset_idle()

    def wheelEvent(self, e):
        self._reset_idle()
        self._dist = max(1.5, min(20.0, self._dist - e.angleDelta().y() / 240.0))
        self.update()
