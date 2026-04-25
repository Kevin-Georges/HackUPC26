# A compilation of all input drivers that will be fed into the system
"""
Input Driver module — HP Metal Jet S100 Digital Twin (Phase 1)

Implements the four Environmental & Operational Vectors from the spec (§1.5.1):

    temperature_stress      — ambient / build-chamber temperature (°C)
    humidity_contamination  — combined humidity + powder purity index (0–1)
    operational_load        — cumulative print cycles since commissioning
    maintenance_level       — service quality coefficient (0–1)

Determinism guarantee: given the same `seed` and the same sequence of
`step()` calls, every driver produces the same output every time.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DriverSnapshot:
    """Immutable snapshot of all four input drivers at one simulation timestep.

    This is the single object passed into the degradation engine each tick.

    Attributes
    ----------
    temperature_stress      : float  — °C.  Optimal 18–25 °C; critical above 40 °C.
    humidity_contamination  : float  — 0–1. 0 = perfectly clean/dry; 1 = max contamination.
    operational_load        : float  — cumulative print cycles (monotonically increasing).
    maintenance_level       : float  — 0–1. 1 = fully serviced; 0 = completely neglected.
    """
    temperature_stress:     float
    humidity_contamination: float
    operational_load:       float
    maintenance_level:      float

    def __str__(self) -> str:
        return (
            f"T={self.temperature_stress:5.1f}°C  "
            f"contam={self.humidity_contamination:.3f}  "
            f"load={self.operational_load:7.1f} cyc  "
            f"maint={self.maintenance_level:.3f}"
        )


# ── Individual drivers ────────────────────────────────────────────────────────

class TemperatureStressDriver:
    """Ambient / build-chamber temperature in Celsius.

    Physics rationale
    -----------------
    The build chamber temperature affects:
      • Recoater blade:    high heat softens the blade edge, accelerating abrasive wear
      • Nozzle plate:      temperature outside optimal bounds increases clog probability
      • Heating elements:  must compensate harder when ambient is cold → more electrical stress

    Model
    -----
    A stable set-point with:
      1. Slow sinusoidal drift simulating a thermal day-cycle.
      2. Gaussian noise (sensor jitter + HVAC fluctuation).
      3. Rare sudden thermal shocks (door opening, cooling failure, etc.).
    """

    OPTIMAL_LOW  = 18.0   # °C
    OPTIMAL_HIGH = 25.0   # °C
    WARNING      = 35.0   # °C
    CRITICAL     = 40.0   # °C

    def __init__(
        self,
        set_point:         float = 21.0,
        noise_std:         float = 1.2,
        drift_amplitude:   float = 3.0,
        drift_period_s:    float = 28_800.0,  # 8-hour HVAC cycle
        shock_probability: float = 0.015,
        shock_magnitude:   float = 14.0,
        seed:              int   = 0,
    ):
        self._set_point      = set_point
        self._noise_std      = noise_std
        self._drift_amp      = drift_amplitude
        self._drift_period   = drift_period_s
        self._shock_prob     = shock_probability
        self._shock_mag      = shock_magnitude
        self._rng            = np.random.default_rng(seed)

    def step(self, t: float) -> float:
        """Return temperature (°C) at simulation time *t* (seconds)."""
        drift = self._drift_amp * np.sin(2.0 * np.pi * t / self._drift_period)
        noise = float(self._rng.normal(0.0, self._noise_std))
        shock = float(self._shock_mag if self._rng.random() < self._shock_prob else 0.0)
        return float(np.clip(self._set_point + drift + noise + shock, -5.0, 80.0))


class HumidityContaminationDriver:
    """Combined humidity and powder contamination index (0–1).

    Physics rationale
    -----------------
    The HP Metal Jet uses fine metallic powder.  Any moisture or airborne
    contamination:
      • Causes powder clumping → uneven recoating → blade wear
      • Increases binder viscosity variation → nozzle clogs
      • Accelerates corrosion on insulation panels

    Model
    -----
    Contamination has internal state.  Each step it:
      1. Accumulates in proportion to operational_load delta and (1 - maintenance).
      2. Is partially cleaned by a maintenance effect (proportional to maintenance_level).
      3. Receives small Gaussian noise (sensor drift, humidity spikes).
    """

    NOMINAL  = 0.10
    WARNING  = 0.40
    CRITICAL = 0.65

    def __init__(
        self,
        baseline:          float = 0.10,
        accum_per_cycle:   float = 3.5e-4,   # contamination added per print cycle
        clean_rate:        float = 8.0e-4,    # contamination removed per maintenance unit
        noise_std:         float = 0.008,
        seed:              int   = 1,
    ):
        self._level          = baseline
        self._accum          = accum_per_cycle
        self._clean          = clean_rate
        self._noise_std      = noise_std
        self._rng            = np.random.default_rng(seed)
        self._prev_load      = 0.0

    def step(self, operational_load: float, maintenance_level: float) -> float:
        """Return contamination index (0–1).

        Parameters
        ----------
        operational_load : cumulative cycle count from OperationalLoadDriver
        maintenance_level: current maintenance coefficient from MaintenanceLevelDriver
        """
        delta_cycles   = max(0.0, operational_load - self._prev_load)
        self._prev_load = operational_load

        accumulation   = self._accum * delta_cycles * (1.0 - maintenance_level)
        cleaning       = self._clean * maintenance_level
        noise          = float(self._rng.normal(0.0, self._noise_std))

        self._level += accumulation - cleaning + noise
        self._level  = float(np.clip(self._level, 0.0, 1.0))
        return self._level


class OperationalLoadDriver:
    """Cumulative print-cycle counter (monotonically increasing).

    Physics rationale
    -----------------
    Every print cycle:
      • Moves the recoater blade across the powder bed → cumulative abrasion
      • Fires the nozzle plate N times → cumulative thermal fatigue
      • Runs the heating elements → cumulative electrical degradation

    Model
    -----
    A simple integrator.  `work_rate` can be < 1.0 to model partial shifts
    or machine downtime (e.g., waiting for powder resupply).
    """

    def __init__(self, work_rate: float = 1.0):
        """
        Parameters
        ----------
        work_rate : cycles added per unit of simulation time (default 1.0).
        """
        self._rate       = max(0.0, work_rate)
        self._cumulative = 0.0

    def step(self, dt: float = 1.0) -> float:
        """Advance by *dt* time units and return the cumulative cycle count."""
        self._cumulative += self._rate * dt
        return self._cumulative

    @property
    def cycles(self) -> float:
        return self._cumulative

    def reset(self) -> None:
        self._cumulative = 0.0


class MaintenanceLevelDriver:
    """Service quality coefficient (0–1).

    Physics rationale
    -----------------
    Good maintenance:
      • Cleans contamination from powder bed and nozzles
      • Lubricates rails → slows motor / rail fatigue
      • Replaces worn blade tips before failure
      • Calibrates heating elements

    A machine that is never serviced converges toward 0.0 (neglect); a
    freshly serviced machine starts near 1.0.

    Model
    -----
    Exponential decay between scheduled service events.  A service event at
    every `service_interval` timesteps boosts the level back toward 1.0.
    An optional noise term models the variability in technician thoroughness.
    """

    GOOD     = 0.75
    ADEQUATE = 0.50
    POOR     = 0.25

    def __init__(
        self,
        initial_level:    float = 0.95,
        decay_rate:       float = 2.5e-4,   # fraction lost per timestep
        service_interval: float = 500.0,    # timesteps between services
        service_restore:  float = 0.85,     # fraction of gap to 1.0 restored on service
        noise_std:        float = 0.005,
        seed:             int   = 2,
    ):
        self._level           = float(np.clip(initial_level, 0.0, 1.0))
        self._decay           = decay_rate
        self._interval        = service_interval
        self._restore         = service_restore
        self._noise_std       = noise_std
        self._rng             = np.random.default_rng(seed)
        self._last_service_t  = 0.0

    def step(self, t: float) -> float:
        """Return maintenance level (0–1) at simulation time *t*."""
        # Exponential decay between services
        self._level *= (1.0 - self._decay)

        # Scheduled service event
        if t - self._last_service_t >= self._interval:
            gap            = 1.0 - self._level
            self._level   += self._restore * gap
            self._last_service_t = t

        # Small noise (technician skill variance, sensor jitter)
        self._level += float(self._rng.normal(0.0, self._noise_std))
        self._level  = float(np.clip(self._level, 0.0, 1.0))
        return self._level

    def force_service(self, t: float) -> float:
        """Trigger an unscheduled full service (e.g. manual maintenance action)."""
        gap = 1.0 - self._level
        self._level += self._restore * gap
        self._last_service_t = t
        return float(np.clip(self._level, 0.0, 1.0))


# ── Aggregate suite ───────────────────────────────────────────────────────────

class DriverSuite:
    """Coordinates all four drivers and emits a DriverSnapshot each tick.

    The evaluation order inside `step()` resolves inter-driver dependencies:
      1. OperationalLoad  — no dependencies
      2. MaintenanceLevel — no dependencies
      3. Temperature      — no dependencies
      4. Contamination    — depends on load + maintenance

    Usage
    -----
    ::

        suite = DriverSuite(seed=42)
        for t in range(2000):
            snap = suite.step(t)
            engine.update(snap)         # pass to Phase 1 degradation engine
    """

    def __init__(
        self,
        seed:                   int   = 42,
        temperature_set_point:  float = 21.0,
        work_rate:              float = 1.0,
        service_interval:       float = 500.0,
    ):
        self.temperature   = TemperatureStressDriver(
            set_point=temperature_set_point,
            seed=seed,
        )
        self.contamination = HumidityContaminationDriver(seed=seed + 1)
        self.load          = OperationalLoadDriver(work_rate=work_rate)
        self.maintenance   = MaintenanceLevelDriver(
            service_interval=service_interval,
            seed=seed + 2,
        )

    def step(self, t: float, dt: float = 1.0) -> DriverSnapshot:
        """Advance all drivers by one timestep and return a DriverSnapshot.

        Parameters
        ----------
        t  : current simulation time (seconds or cycles — must be consistent)
        dt : elapsed time since last step (default 1.0)
        """
        op_load = self.load.step(dt)
        maint   = self.maintenance.step(t)
        temp    = self.temperature.step(t)
        contam  = self.contamination.step(op_load, maint)

        return DriverSnapshot(
            temperature_stress=temp,
            humidity_contamination=contam,
            operational_load=op_load,
            maintenance_level=maint,
        )


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DriverSuite smoke-test — 10 steps\n")
    suite = DriverSuite(seed=42, service_interval=5.0)
    for t in range(10):
        snap = suite.step(float(t))
        print(f"  t={t:3d}  {snap}")

    print("\nDeterminism check (re-run same seed, same output expected):")
    suite2 = DriverSuite(seed=42, service_interval=5.0)
    snaps1 = [suite2.step(float(t)) for t in range(10)]
    suite3 = DriverSuite(seed=42, service_interval=5.0)
    snaps2 = [suite3.step(float(t)) for t in range(10)]
    assert snaps1 == snaps2, "FAIL: outputs differ between runs!"
    print("  PASS — identical outputs for same seed")
