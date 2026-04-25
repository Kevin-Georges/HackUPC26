"""
Degradation Engine — HP Metal Jet S100 Digital Twin (Phase 1)

Rule-based degradation models for the three required subsystems:

    Recoater Blade    — Abrasive Wear (exponential decay)
    Nozzle Plate      — Clogging + Thermal Fatigue (combined exponential)
    Heating Elements  — Electrical Degradation (Weibull survival)

Inputs
------
temperature  : float  — build-chamber / ambient temperature in °C
humidity     : float  — contamination index 0–1 (0 = clean/dry, 1 = max)
print_volume : float  — cumulative print cycles since commissioning

All models are deterministic: same inputs → same outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_OPTIMAL_TEMP_LOW  = 18.0   # °C — lower bound of the safe operating band
_OPTIMAL_TEMP_HIGH = 25.0   # °C — upper bound of the safe operating band


def _temp_stress(temperature: float) -> float:
    """Multiplicative stress factor: 1.0 inside optimal band, rises outside it."""
    if _OPTIMAL_TEMP_LOW <= temperature <= _OPTIMAL_TEMP_HIGH:
        return 1.0
    excess = max(temperature - _OPTIMAL_TEMP_HIGH, _OPTIMAL_TEMP_LOW - temperature)
    return 1.0 + 0.05 * excess


@dataclass(frozen=True)
class ComponentHealth:
    name:       str
    health_pct: int   # 0–100

    @property
    def pct_str(self) -> str:
        return f"{self.health_pct}%"


class DegradationEngine:
    """Compute component health given current environmental and operational inputs."""

    # ── Recoater Blade (Subsystem A — Abrasive Wear) ───────────────────────────
    # Each cycle drags the blade through metal powder.  Humidity causes powder
    # clumping which grinds the edge harder; excess temperature softens the
    # blade material, amplifying the abrasion rate.
    #   Health = exp(-k_blade * cycles * (1 + amp_H * humidity) * temp_stress)
    _BLADE_BASE_RATE    = 8.0e-5   # wear rate per cycle at nominal conditions
    _BLADE_HUMIDITY_AMP = 2.5      # humidity multiplier on wear rate

    # ── Nozzle Plate (Subsystem B — Clogging + Thermal Fatigue) ───────────────
    # Two independent damage mechanisms accumulate additively in the exponent:
    #   clog damage    ∝ humidity × cycles  (moisture/contamination blocks jets)
    #   thermal damage ∝ cycles × temp_stress  (fatigue from heating/cooling)
    #   Health = exp(-(clog_damage + thermal_damage))
    _NOZZLE_CLOG_RATE    = 0.20    # clog damage per humidity unit per 10 k cycles
    _NOZZLE_THERMAL_RATE = 1.5e-5  # thermal fatigue per cycle at optimal temp
    _NOZZLE_THERMAL_AMP  = 3.0     # amplifier applied when temperature is stressed

    # ── Heating Elements (Subsystem C — Weibull Electrical Degradation) ────────
    # Classic two-parameter Weibull survival function S(t) = exp(-(t/η)^β).
    # Cold ambient forces elements to work harder, shortening the characteristic
    # life η proportionally to the temperature deficit.
    #   Health = exp(-(cycles / eff_life)^shape)
    _HEATER_CHAR_LIFE    = 18_000.0  # η — characteristic life in cycles
    _HEATER_SHAPE        = 2.2       # β — shape parameter (>1 = wear-out regime)
    _HEATER_COLD_PENALTY = 0.025     # fractional life reduction per °C below optimal

    def compute(
        self,
        temperature:  float,
        humidity:     float,
        print_volume: float,
    ) -> list[ComponentHealth]:
        """Return health for each component given current conditions.

        Parameters
        ----------
        temperature  : °C — build-chamber temperature
        humidity     : 0–1 — contamination / moisture index
        print_volume : cumulative print cycles
        """
        tf = _temp_stress(temperature)

        # Recoater Blade
        blade_h = math.exp(
            -self._BLADE_BASE_RATE
            * print_volume
            * (1.0 + self._BLADE_HUMIDITY_AMP * humidity)
            * tf
        )

        # Nozzle Plate
        clog    = self._NOZZLE_CLOG_RATE * humidity * (print_volume / 10_000.0)
        thermal = self._NOZZLE_THERMAL_RATE * print_volume * tf * self._NOZZLE_THERMAL_AMP
        nozzle_h = math.exp(-(clog + thermal))

        # Heating Elements (Weibull)
        cold_excess = max(0.0, _OPTIMAL_TEMP_LOW - temperature)
        eff_life    = self._HEATER_CHAR_LIFE * (1.0 - self._HEATER_COLD_PENALTY * cold_excess)
        heater_h    = math.exp(-(print_volume / max(eff_life, 1.0)) ** self._HEATER_SHAPE)

        def _pct(h: float) -> int:
            return round(max(0.0, min(1.0, h)) * 100)

        return [
            ComponentHealth("Recoater Blade",  _pct(blade_h)),
            ComponentHealth("Nozzle Plate",    _pct(nozzle_h)),
            ComponentHealth("Heating Elements", _pct(heater_h)),
        ]
