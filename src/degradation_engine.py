"""
Degradation Engine — HP Metal Jet S100 Digital Twin (Phase 1)

One well-known hardware-degradation model per subsystem:

    Recoater Blade    — Archard Wear Model          (tribology / abrasive contact)
    Nozzle Plate      — Coffin-Manson Fatigue Law   (thermal-cycle fatigue + Miner's rule)
    Heating Elements  — Arrhenius Degradation Model (thermally-activated electrical decay)

The engine is stateful: call tick() once per simulation step.
Damage for each component accumulates across ticks so the full
degradation history is preserved in the engine's state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Physical constants
_BOLTZMANN_EV = 8.617e-5    # Boltzmann constant in eV/K
_KELVIN       = 273.15      # °C → K offset

_OPTIMAL_TEMP = 21.5        # °C — midpoint of the safe operating band (18–25 °C)


@dataclass(frozen=True)
class ComponentHealth:
    name:       str
    health_pct: int   # 0–100

    @property
    def pct_str(self) -> str:
        return f"{self.health_pct}%"


class DegradationEngine:
    """Stateful Phase 1 Logic Engine.  Call tick() once per simulation step."""

    # ── Model 1 — Archard Wear (Recoater Blade) ────────────────────────────────
    #
    # W = K * F * s / H
    #
    #   W  — wear volume (normalised, 0→W_max = failure)
    #   K  — dimensionless wear coefficient (material pair: tool-steel blade / SS powder)
    #   F  — normal force (N); rises with humidity because wet powder clumps and
    #         resists the blade, increasing the contact force
    #   s  — total sliding distance (m) = cycles × stroke_length
    #   H  — Vickers hardness of the blade material (MPa); decreases linearly
    #         above the optimal temperature as the steel softens
    #
    # Reference: Archard, J.F. (1953). "Contact and Rubbing of Flat Surfaces."
    #            Journal of Applied Physics, 24(8), 981–988.
    _ARCHARD_K            = 1.2e-4   # wear coefficient for tool-steel on metal powder
    _ARCHARD_F0           = 10.0     # baseline normal force (N) at zero contamination
    _ARCHARD_HUMIDITY_K   = 5.0      # force amplification factor per unit humidity (0–1)
    _ARCHARD_H0           = 800.0    # baseline hardness (MPa) at optimal temperature
    _ARCHARD_H_TEMP_SLOPE = 8.0      # hardness reduction rate (MPa per °C above optimal)
    _ARCHARD_STROKE       = 0.35     # blade stroke per cycle (m)
    _ARCHARD_W_MAX        = 0.015    # normalised wear volume at end-of-life

    # ── Model 2 — Coffin-Manson Fatigue Law (Nozzle Plate) ────────────────────
    #
    # N_f = C * (ΔT_eff)^(-m)            — cycles to failure at thermal range ΔT_eff
    # D   += 1 / N_f   per cycle         — Miner's linear damage accumulation rule
    # D = 1  →  failure
    #
    #   ΔT_eff  — effective thermal strain range (°C):
    #               temperature deviation from optimal
    #             + contamination contribution (humidity amplifies binder viscosity
    #               variation, adding an equivalent thermal strain)
    #   C, m    — Coffin-Manson material constants for the nozzle-plate alloy
    #
    # Reference: Coffin, L.F. (1954). Trans. ASME, 76, 931–950.
    #            Manson, S.S. (1953). NACA TN 2933.
    _CM_BASE_DT     = 5.0       # intrinsic thermal cycle amplitude per firing (°C)
    _CM_AMBIENT_K   = 0.5       # fraction of ambient deviation added to ΔT
    _CM_HUMIDITY_K  = 2.0       # humidity → equivalent ΔT contribution (°C per unit)
    _CM_C           = 500_000.0 # material constant tuned for ~17 k cycle life at nominal
    _CM_M           = 2.0       # fatigue ductility exponent

    # ── Model 3 — Arrhenius Degradation Model (Heating Elements) ──────────────
    #
    # Degradation rate:   r(T) = A * exp(-Ea / (k_B * T_K))
    # Lifetime at T:      L(T) = L_ref * exp( (Ea/k_B) * (1/T_K - 1/T_ref_K) )
    # Damage per cycle:   d    = 1 / L(T)
    #
    #   Ea      — activation energy (eV); represents the energy barrier for
    #             oxide-layer growth / electromigration in the resistive element
    #   T_K     — absolute temperature of the element (K)
    #             = ambient + self-heating offset (heating elements run hotter
    #               than ambient; a cold room forces more self-heating → higher T_K)
    #   L_ref   — characteristic life (cycles) at the reference temperature
    #
    # Reference: Arrhenius, S. (1889). Z. Phys. Chem. 4, 226–248.
    #            MIL-HDBK-217F — Reliability Prediction of Electronic Equipment.
    _ARR_EA           = 0.85      # activation energy (eV) — metal-oxide resistor
    _ARR_T_REF_C      = 25.0      # reference temperature (°C)
    _ARR_L_REF        = 20_000.0  # characteristic life at T_ref (cycles)
    _ARR_SELF_HEAT_K  = 1.5       # extra self-heating per °C below optimal (°C/°C)

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._blade_wear      = 0.0   # Archard: cumulative normalised wear volume
        self._nozzle_damage   = 0.0   # Coffin-Manson: cumulative Miner's damage (0→1)
        self._heater_damage   = 0.0   # Arrhenius: cumulative damage fraction (0→1)
        self._prev_load       = 0.0   # last operational_load to derive delta_cycles

    def tick(
        self,
        temperature:      float,
        humidity:         float,
        operational_load: float,
    ) -> list[ComponentHealth]:
        """Advance one simulation step and return updated health values.

        Parameters
        ----------
        temperature      : °C  — build-chamber / ambient temperature
        humidity         : 0–1 — contamination / moisture index
        operational_load : cumulative print cycles (monotonically increasing)
        """
        delta = max(0.0, operational_load - self._prev_load)
        self._prev_load = operational_load

        self._blade_wear    += self._archard_increment(temperature, humidity, delta)
        self._nozzle_damage += self._coffin_manson_increment(temperature, humidity, delta)
        self._heater_damage += self._arrhenius_increment(temperature, delta)

        def _pct(dmg: float, capacity: float) -> int:
            return round(max(0.0, min(1.0, 1.0 - dmg / capacity)) * 100)

        return [
            ComponentHealth("Recoater Blade",   _pct(self._blade_wear,    self._ARCHARD_W_MAX)),
            ComponentHealth("Nozzle Plate",      _pct(self._nozzle_damage, 1.0)),
            ComponentHealth("Heating Elements",  _pct(self._heater_damage, 1.0)),
        ]

    # ── Private model calculations ─────────────────────────────────────────────

    def _archard_increment(self, temperature: float, humidity: float, cycles: float) -> float:
        """W = K * F * s / H  — wear volume for this cycle increment."""
        F = self._ARCHARD_F0 * (1.0 + self._ARCHARD_HUMIDITY_K * humidity)
        T_excess = max(0.0, temperature - _OPTIMAL_TEMP)
        H = max(50.0, self._ARCHARD_H0 - self._ARCHARD_H_TEMP_SLOPE * T_excess)
        s = self._ARCHARD_STROKE * cycles
        return self._ARCHARD_K * F * s / H

    def _coffin_manson_increment(self, temperature: float, humidity: float, cycles: float) -> float:
        """Miner's damage = cycles / N_f,  N_f = C * ΔT_eff^(-m).

        ΔT_eff = base firing amplitude + ambient deviation contribution + humidity strain.
        The base term ensures ambient temperature fluctuations are a modifier,
        not the sole driver of fatigue damage.
        """
        delta_T = (
            self._CM_BASE_DT
            + self._CM_AMBIENT_K * abs(temperature - _OPTIMAL_TEMP)
            + self._CM_HUMIDITY_K * humidity
        )
        N_f = self._CM_C * (delta_T ** -self._CM_M)
        return cycles / max(N_f, 1.0)

    def _arrhenius_increment(self, temperature: float, cycles: float) -> float:
        """Lifetime fraction consumed = cycles / L(T_element)."""
        cold_deficit = max(0.0, _OPTIMAL_TEMP - temperature)
        T_element_C  = temperature + self._ARR_SELF_HEAT_K * cold_deficit
        T_K          = T_element_C + _KELVIN
        T_ref_K      = self._ARR_T_REF_C + _KELVIN
        lifetime = self._ARR_L_REF * math.exp(
            (self._ARR_EA / _BOLTZMANN_EV) * (1.0 / T_K - 1.0 / T_ref_K)
        )
        return cycles / max(lifetime, 1.0)
