#!/usr/bin/env python3
import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QSurfaceFormat
from main_window import MainWindow


def main():
    # Request a compatibility-profile context so legacy GL functions work
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)
    fmt.setProfile(QSurfaceFormat.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    here = os.path.dirname(os.path.abspath(__file__))
    model = os.path.join(here, "../assets/Proper HP Printer All Components.glb")
    win = MainWindow(model)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
