"""
Degradation Engine — HP Metal Jet S100 Digital Twin

One well-known hardware-degradation model per subsystem:

    Recoater Blade             — Archard Wear Model          (tribology / abrasive contact)
    Nozzle Plate               — Coffin-Manson Fatigue Law   (thermal-cycle fatigue + Miner's rule)
    Heating Elements           — Arrhenius Degradation Model (thermally-activated electrical decay)
    Drive Motor & Rails        — Paris Law Fatigue           (mechanical fatigue crack growth)
    Cleaning & Thermal Iface   — Kern-Seaton Fouling Model   (asymptotic fouling kinetics)
    Insulation & Sensors       — Moisture-Thermal Degradation (Fickian diffusion + thermal cracking)

The engine is stateful: call tick() once per simulation day.
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
    """Stateful Phase 1 Logic Engine.  Call tick() once per simulation day."""

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
    _ARCHARD_K            = 2.0e-3   # wear coefficient for tool-steel on metal powder
                                      # calibrated so blade reaches ~50 % health at 500 builds
                                      # and fails at ~1 200 builds — consistent with LPBF doctor-blade
                                      # replacement schedules of 300–500 builds (BJ is ~2–3× gentler)
    _ARCHARD_F0           = 10.0     # baseline normal force (N) at zero contamination
    _ARCHARD_HUMIDITY_K   = 5.0      # force amplification factor per unit humidity (0–1)
    _ARCHARD_POWDER_K     = 1.5      # force amplification from degraded powder (irregular particles)
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
    _CM_POWDER_K    = 1.5       # degraded powder → equivalent ΔT contribution (°C per unit)
    _CM_BINDER_K    = 2.0       # binder viscosity stress → equivalent ΔT contribution (°C per unit)
    _CM_C           = 24_000.0  # material constant: ~50 % health at 300 builds, failure ~700–800 builds
                                  # based on ExOne/Voxeljet printhead service intervals of 6–12 months
                                  # at 1–2 builds/day (180–730 builds between major services)
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
    _ARR_L_REF        = 3_000.0   # characteristic life at T_ref (cycles)
                                   # at nominal 21 °C the Arrhenius factor extends this to ~3 000 builds
                                   # grounded in MIL-HDBK-217F / industrial cartridge-heater MTBF of
                                   # 5 000–10 000 hr at rated temp; at ~8 hr/build → 625–1 250 builds
                                   # to MTBF, so safe-life replacement at ~2 000–3 000 builds
    _ARR_SELF_HEAT_K  = 1.5       # extra self-heating per °C below optimal (°C/°C)
    _ARR_VOLTAGE_K    = 15.0      # equivalent temperature rise per unit voltage stress (°C)

    # ── Model 4 — Paris Law Fatigue (Drive Motor & Rails) ─────────────────────
    #
    # da/dN = C * (ΔK)^m          — crack growth per cycle
    #
    #   a    — normalised crack length (0 → a_crit = 1.0 = failure)
    #   ΔK   — stress intensity factor range:
    #           ΔK = ΔK_0 * (1 + k_load * rate) * (1 + k_humid * humidity)
    #          high production rate raises cyclic stress; humidity causes
    #          corrosion-assisted fatigue, broadening the effective ΔK
    #   C, m — Paris material constants (tool-steel drive shaft)
    #
    # Reference: Paris, P.C. & Erdogan, F. (1963). J. Basic Eng., 85, 528–534.
    _PARIS_C         = 5.5e-5   # crack growth coefficient
                                  # calibrated to industrial servo/stepper motor MTBF of 5–10 yr
                                  # (THK linear guide rails: 50 000–100 000 hr rated life)
                                  # at 1 build/day → ~50 % health at ~750 builds, failure ~1 500 builds ≈ 4 yr
    _PARIS_M         = 3.0      # Paris exponent (steel alloys: typically 2.5–4.0)
    _PARIS_DK0       = 1.0      # baseline ΔK at unit load rate
    _PARIS_K_LOAD    = 1.2      # load-rate amplification of ΔK
    _PARIS_K_HUMID   = 0.5      # humidity contribution to ΔK (corrosion-assisted fatigue)
    _PARIS_A_CRIT    = 1.0      # normalised critical crack size at failure

    # ── Model 5 — Kern-Seaton Fouling (Cleaning & Thermal Interface) ──────────
    #
    # dRf/dt = φ_d − φ_r · Rf
    #
    #   Rf   — normalised fouling resistance (0 = clean, 1 = failed)
    #   φ_d  — deposition rate ∝ contamination × load_rate
    #           (more dirt + more cycles → faster fouling)
    #   φ_r  — removal rate ∝ maintenance_level
    #           (good maintenance dissolves foulant deposits)
    #
    # Asymptotic fouling: Rf* = φ_d / φ_r  (equilibrium when dRf/dt = 0)
    #
    # Reference: Kern, D.Q. & Seaton, R.E. (1959). Brit. Chem. Eng., 4, 258–262.
    _KS_PHI_D_K      = 0.025    # deposition rate constant (per unit contamination per cycle)
                                  # inkjet/BJ wiper blades replaced every 3–4 months of active use
                                  # → cleaning interface consumable lifespan ~200 builds to 50 % health,
                                  #   ~500 builds to failure (≈16 months at 1 build/day)
    _KS_PHI_R_K      = 0.004    # removal rate constant (per unit maintenance per day)
    _KS_RF_MAX       = 1.0      # normalised fouling resistance at failure

    # ── Model 6 — Moisture-Thermal Degradation (Insulation & Sensors) ─────────
    #
    # Two coupled damage mechanisms:
    #
    #   Moisture ingress (Fickian diffusion):
    #       dM/dt = D · (humidity − M)
    #       M     — normalised moisture content (0–1)
    #       D     — diffusion constant; moisture approaches humidity equilibrium
    #
    #   Thermal cracking:
    #       d_crack += k_temp · |T − T_opt| · delta_cycles
    #       Temperature swings expand/contract the insulation, opening micro-cracks
    #       that allow further moisture penetration — a positive feedback loop.
    #
    #   Combined damage:
    #       damage += k_moist · M + d_crack
    #
    # When insulation fails the heating elements must compensate for heat loss →
    # their Arrhenius damage rate is amplified proportionally.
    #
    # Reference: IEC 60085 — Thermal classification of electrical insulation.
    _INS_D_MOISTURE  = 0.012    # Fickian diffusion rate (fraction per day toward equilibrium)
    _INS_K_MOIST     = 1.2e-3   # moisture → damage rate
    _INS_K_TEMP      = 8.0e-5   # thermal-cracking rate per °C deviation per cycle
                                  # industrial insulation panels in heated chambers (150–200 °C) rated
                                  # for decades but degraded by thermal cycling and moisture ingress
                                  # (IEC 60085 Class F/H: ~10–20 yr continuous service life)
                                  # → calibrated to ~50 % health at ~2 000 builds, failure ~4 500 builds ≈ 12 yr
    _INS_DAMAGE_MAX  = 1.0      # damage at failure
    _INS_HEATER_AMP  = 0.6      # fraction by which poor insulation amplifies heater damage

    # ──────────────────────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._blade_wear      = 0.0   # Archard
        self._nozzle_damage   = 0.0   # Coffin-Manson
        self._heater_damage   = 0.0   # Arrhenius
        self._motor_crack     = 0.0   # Paris Law: normalised crack length
        self._fouling         = 0.0   # Kern-Seaton: normalised fouling resistance
        self._ins_moisture    = 0.0   # Fickian: current moisture content
        self._ins_damage      = 0.0   # combined insulation damage
        self._prev_load       = 0.0

    def tick(
        self,
        temperature:             float,
        humidity:                float,
        operational_load:        float,
        powder_quality:          float = 1.0,
        binder_viscosity_stress: float = 0.0,
        voltage_stress:          float = 0.0,
        maintenance_level:       float = 1.0,
    ) -> list[ComponentHealth]:
        """Advance one simulation day and return updated health values.

        Parameters
        ----------
        temperature             : °C  — build-chamber / ambient temperature
        humidity                : 0–1 — contamination / moisture index
        operational_load        : cumulative print cycles since day 0
        powder_quality          : 0–1 — feedstock quality (1 = fresh)
        binder_viscosity_stress : 0–1 — binder age/viscosity stress (0 = optimal)
        voltage_stress          : 0–1 — power-supply instability (0 = stable)
        maintenance_level       : 0–1 — service quality coefficient (1 = fully serviced)
        """
        delta = max(0.0, operational_load - self._prev_load)
        self._prev_load = operational_load

        # Insulation health feeds back into heater damage
        ins_health_frac = max(0.0, 1.0 - self._ins_damage / self._INS_DAMAGE_MAX)
        heater_amp      = 1.0 + self._INS_HEATER_AMP * (1.0 - ins_health_frac)

        self._blade_wear    += self._archard_increment(temperature, humidity, powder_quality, delta)
        self._nozzle_damage += self._coffin_manson_increment(temperature, humidity, powder_quality, binder_viscosity_stress, delta)
        self._heater_damage += self._arrhenius_increment(temperature, voltage_stress, delta) * heater_amp
        self._motor_crack   += self._paris_increment(humidity, delta)
        self._fouling       += self._kern_seaton_increment(humidity, maintenance_level, delta)
        self._ins_damage    += self._insulation_increment(temperature, humidity, delta)

        def _pct(dmg: float, capacity: float) -> int:
            return round(max(0.0, min(1.0, 1.0 - dmg / capacity)) * 100)

        return [
            ComponentHealth("Recoater Blade",            _pct(self._blade_wear,    self._ARCHARD_W_MAX)),
            ComponentHealth("Nozzle Plate",               _pct(self._nozzle_damage, 1.0)),
            ComponentHealth("Heating Elements",           _pct(self._heater_damage, 1.0)),
            ComponentHealth("Drive Motor & Rails",        _pct(self._motor_crack,   self._PARIS_A_CRIT)),
            ComponentHealth("Cleaning & Thermal Iface",  _pct(self._fouling,       self._KS_RF_MAX)),
            ComponentHealth("Insulation & Sensors",       _pct(self._ins_damage,    self._INS_DAMAGE_MAX)),
        ]

    # ── Private model calculations ─────────────────────────────────────────────

    def _archard_increment(self, temperature: float, humidity: float, powder_quality: float, cycles: float) -> float:
        """W = K * F * s / H  — wear volume for this cycle increment.

        Degraded powder increases contact force because irregular particles
        resist the blade; combined with humidity-driven clumping.
        """
        F = (
            self._ARCHARD_F0
            * (1.0 + self._ARCHARD_HUMIDITY_K * humidity)
            * (1.0 + self._ARCHARD_POWDER_K * (1.0 - powder_quality))
        )
        T_excess = max(0.0, temperature - _OPTIMAL_TEMP)
        H = max(50.0, self._ARCHARD_H0 - self._ARCHARD_H_TEMP_SLOPE * T_excess)
        s = self._ARCHARD_STROKE * cycles
        return self._ARCHARD_K * F * s / H

    def _coffin_manson_increment(self, temperature: float, humidity: float, powder_quality: float, binder_viscosity_stress: float, cycles: float) -> float:
        """Miner's damage = cycles / N_f,  N_f = C * ΔT_eff^(-m).

        ΔT_eff = base firing amplitude
               + ambient deviation contribution
               + humidity strain
               + degraded-powder strain (irregular particles vary binder drop placement)
               + binder viscosity strain (high viscosity increases firing pressure → more fatigue)
        """
        delta_T = (
            self._CM_BASE_DT
            + self._CM_AMBIENT_K  * abs(temperature - _OPTIMAL_TEMP)
            + self._CM_HUMIDITY_K * humidity
            + self._CM_POWDER_K   * (1.0 - powder_quality)
            + self._CM_BINDER_K   * binder_viscosity_stress
        )
        N_f = self._CM_C * (delta_T ** -self._CM_M)
        return cycles / max(N_f, 1.0)

    def _paris_increment(self, humidity: float, cycles: float) -> float:
        """da = C * (ΔK)^m * cycles  — Paris Law crack growth per day.

        Load rate (cycles/day) raises cyclic stress; humidity promotes
        corrosion-assisted fatigue that accelerates crack tip growth.
        """
        delta_K = self._PARIS_DK0 * (1.0 + self._PARIS_K_LOAD * cycles) * (1.0 + self._PARIS_K_HUMID * humidity)
        return self._PARIS_C * (delta_K ** self._PARIS_M) * cycles

    def _kern_seaton_increment(self, humidity: float, maintenance_level: float, cycles: float) -> float:
        """dRf = φ_d − φ_r · Rf  — Kern-Seaton asymptotic fouling per day.

        Deposition scales with contamination load (humidity × cycles);
        removal scales with maintenance quality.  The fouling asymptote is
        φ_d / φ_r — a heavily used, poorly maintained machine converges to a
        high fouling resistance (clogged nozzle passages, degraded thermal pad).
        """
        phi_d = self._KS_PHI_D_K * humidity * cycles
        phi_r = self._KS_PHI_R_K * maintenance_level
        return phi_d - phi_r * self._fouling

    def _insulation_increment(self, temperature: float, humidity: float, cycles: float) -> float:
        """Combined moisture-diffusion and thermal-cracking damage per day.

        Fickian moisture ingress drives the insulation's moisture content toward
        the ambient humidity equilibrium.  Temperature deviations from optimal
        open micro-cracks proportional to load cycling (more firings per day →
        more thermal expansions and contractions).
        """
        self._ins_moisture += self._INS_D_MOISTURE * (humidity - self._ins_moisture)
        self._ins_moisture  = max(0.0, min(1.0, self._ins_moisture))

        thermal_cracking = self._INS_K_TEMP * abs(temperature - _OPTIMAL_TEMP) * cycles
        moisture_damage  = self._INS_K_MOIST * self._ins_moisture
        return moisture_damage + thermal_cracking

    def _arrhenius_increment(self, temperature: float, voltage_stress: float, cycles: float) -> float:
        """Lifetime fraction consumed = cycles / L(T_element).

        Voltage spikes cause Joule heating inside the element, raising its
        effective operating temperature above the ambient.  This is added on top
        of the cold-compensation self-heating term before entering the Arrhenius
        exponential, so both stressors compound correctly.
        """
        cold_deficit = max(0.0, _OPTIMAL_TEMP - temperature)
        T_element_C  = (
            temperature
            + self._ARR_SELF_HEAT_K * cold_deficit
            + self._ARR_VOLTAGE_K   * voltage_stress
        )
        T_K     = T_element_C + _KELVIN
        T_ref_K = self._ARR_T_REF_C + _KELVIN
        lifetime = self._ARR_L_REF * math.exp(
            (self._ARR_EA / _BOLTZMANN_EV) * (1.0 / T_K - 1.0 / T_ref_K)
        )
        return cycles / max(lifetime, 1.0)
