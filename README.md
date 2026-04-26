# HP Metal Jet S100 — Digital Twin

A real-time Digital Twin simulation of the HP Metal Jet S100 industrial 3D printer, built for HackUPC 2026. The simulator models component degradation over time and visualises it on a 3D model of the printer with live health charts.

## Features

- 3D model of the HP Metal Jet S100 built in Blender from public HP documentation, with 25+ individually named components
- Real-time health simulation across six components (Nozzle Plates, Recoater Blade, Heating Elements, Drive Motor & Rails, Cleaning & Thermal Interface, Insulation & Sensors)
- Components colour-shift green → red as health degrades; click any part to highlight it
- Live charts for temperature, humidity/contamination, and printer usage
- Adjustable simulation speed (0.25× – 8×) with pause and reset
- CSV export of full telemetry per simulation run

## Setup

**Requirements:** Python 3.10+

1. Clone the repo:
   ```bash
   git clone https://github.com/Kevin-Georges/HackUPC26.git
   cd HackUPC26
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   cd src
   python main.py
   ```

> On some systems PyOpenGL may require an additional system-level OpenGL driver. If the 3D viewer fails to initialise, ensure your GPU drivers are up to date.
