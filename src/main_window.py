import csv
import pathlib

from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QFrame,
)
from constants import BG, PANEL, BORDER, TEXT
from model_viewer import ModelViewer, build_color_map
from charts_panel import ChartsPanel
from health_panel import HealthPanel

_CSV_DIR = pathlib.Path(__file__).parent.parent
_CSV_HEADER = [
    "run_id",
    "day",
    "temperature_c",
    "humidity_index",
    "operational_load",
    "maintenance_level",
    "powder_quality",
    "binder_viscosity_stress",
    "voltage_stress",
    "recoater_blade_pct",
    "nozzle_plate_pct",
    "heating_elements_pct",
    "drive_motor_rails_pct",
    "cleaning_thermal_iface_pct",
    "insulation_sensors_pct",
]


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

        self._top_area = QWidget()   # reserved — free space for later use
        self._top_area.setStyleSheet("background: transparent; border: none;")
        ll.addWidget(self._top_area, stretch=1)

        viewer = ModelViewer(model_path)
        ll.addWidget(viewer, stretch=2)
        root.addWidget(left, stretch=3)

        # ── Right: chart + health table ───────────────────────────────────────
        right = QVBoxLayout()
        right.setSpacing(8)

        charts = ChartsPanel()
        right.addWidget(charts, stretch=2)

        health = HealthPanel()
        health.setStyleSheet(
            f"background:{PANEL}; border:1px solid {BORDER}; border-radius:5px;"
        )
        right.addWidget(health, stretch=1)

        viewer.set_component_colors(build_color_map(health._ROWS))

        self._health = health
        self._viewer = viewer
        self._charts = charts
        charts.snapshot_ready.connect(self._on_snap)
        charts.reset_requested.connect(health.reset)
        charts.reset_requested.connect(self._open_csv)
        health.component_failed.connect(self._on_component_failed)
        viewer.component_selected.connect(health.highlight)

        self._csv_file   = None
        self._csv_writer = None
        self._run_id     = 0
        self._open_csv()

        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, stretch=2)

    def _open_csv(self) -> None:
        if self._csv_file:
            self._csv_file.close()
        self._run_id += 1
        path = _CSV_DIR / f"sim_details_{self._run_id}.csv"
        self._csv_file   = open(path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(_CSV_HEADER)
        self._csv_file.flush()

    def _on_component_failed(self, name: str) -> None:
        self._charts.pause()

    def _on_snap(self, snap) -> None:
        self._health.update(
            snap.temperature_stress,
            snap.humidity_contamination,
            snap.operational_load,
            snap.powder_quality,
            snap.binder_viscosity_stress,
            snap.voltage_stress,
            snap.maintenance_level,
        )
        self._viewer.set_component_colors(build_color_map(self._health._ROWS))

        h = {name: pct for name, pct in self._health._ROWS}
        self._csv_writer.writerow([
            self._run_id,
            self._health._day,
            f"{snap.temperature_stress:.4f}",
            f"{snap.humidity_contamination:.4f}",
            f"{snap.operational_load:.4f}",
            f"{snap.maintenance_level:.4f}",
            f"{snap.powder_quality:.4f}",
            f"{snap.binder_viscosity_stress:.4f}",
            f"{snap.voltage_stress:.4f}",
            h.get("Recoater Blade",           ""),
            h.get("Nozzle Plate",             ""),
            h.get("Heating Elements",         ""),
            h.get("Drive Motor & Rails",      ""),
            h.get("Cleaning & Thermal Iface", ""),
            h.get("Insulation & Sensors",     ""),
        ])
        self._csv_file.flush()

    def closeEvent(self, event) -> None:
        if self._csv_file:
            self._csv_file.close()
        super().closeEvent(event)
