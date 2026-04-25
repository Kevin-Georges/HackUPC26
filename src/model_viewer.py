import warnings
import numpy as np
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GLU import gluPerspective
import trimesh
from constants import BLUE


class ModelViewer(QOpenGLWidget):
    """Renders a GLB mesh as a green wireframe.

    Left-drag  → rotate around X/Y
    Right-drag → rotate around Z
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
        self._vbo = 0
        self._n_draw = 0
        self.setMinimumSize(400, 400)

    def initializeGL(self):
        glClearColor(0.05, 0.07, 0.09, 1.0)
        glEnable(GL_DEPTH_TEST)
        glLineWidth(1.0)
        self._load_model()

    def _load_model(self):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mesh = trimesh.load(self.model_path, force="mesh")

            if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
                print("[ModelViewer] not a valid Trimesh after load")
                return

            c = mesh.centroid
            s = float(mesh.extents.max()) or 1.0

            flat = ((mesh.vertices[mesh.faces.reshape(-1)] - c) / s * 1.6).astype(
                np.float32
            )
            self._n_draw = len(flat)

            self._vbo = int(glGenBuffers(1))
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, flat.nbytes, flat, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)

            self._ready = True
            print(
                f"[ModelViewer] {len(mesh.vertices):,} verts  "
                f"{len(mesh.faces):,} faces  extents={mesh.extents.round(2)}"
            )
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
            glColor3f(*BLUE)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glEnableClientState(GL_VERTEX_ARRAY)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glVertexPointer(3, GL_FLOAT, 0, None)
            glDrawArrays(GL_TRIANGLES, 0, self._n_draw)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glDisableClientState(GL_VERTEX_ARRAY)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
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
