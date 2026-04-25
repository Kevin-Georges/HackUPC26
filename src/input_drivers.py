# A compilation of all input drivers that will be fed into the system
"""
Input Driver module — HP Metal Jet S100 Digital Twin (Phase 1)

Implements seven Environmental & Operational Vectors:

    temperature_stress        — ambient / build-chamber temperature (°C)
    humidity_contamination    — combined humidity + powder purity index (0–1)
    operational_load          — cumulative print cycles since commissioning
    maintenance_level         — service quality coefficient (0–1)
    powder_quality            — feedstock quality index (0–1, 1 = fresh)
    binder_viscosity_stress   — binder age / viscosity stress index (0–1)
    voltage_stress            — power-supply instability index (0–1)

Determinism guarantee: given the same `seed` and the same sequence of
`step()` calls, every driver produces the same output every time.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DriverSnapshot:
    """Immutable snapshot of all seven input drivers at one simulation day.

    This is the single object passed into the degradation engine each tick.

    Attributes
    ----------
    temperature_stress      : float  — °C.  Optimal 18–25 °C; critical above 40 °C.
    humidity_contamination  : float  — 0–1. 0 = perfectly clean/dry; 1 = max contamination.
    operational_load        : float  — cumulative print cycles (monotonically increasing).
    maintenance_level       : float  — 0–1. 1 = fully serviced; 0 = completely neglected.
    powder_quality          : float  — 0–1. 1 = fresh powder; 0 = fully recycled/degraded.
    binder_viscosity_stress : float  — 0–1. 0 = optimal viscosity; 1 = max stress.
    voltage_stress          : float  — 0–1. 0 = stable supply; 1 = max fluctuation.
    """
    temperature_stress:       float
    humidity_contamination:   float
    operational_load:         float
    maintenance_level:        float
    powder_quality:           float
    binder_viscosity_stress:  float
    voltage_stress:           float

    def __str__(self) -> str:
        return (
            f"T={self.temperature_stress:5.1f}°C  "
            f"contam={self.humidity_contamination:.3f}  "
            f"load={self.operational_load:7.1f} cyc  "
            f"maint={self.maintenance_level:.3f}  "
            f"powder={self.powder_quality:.3f}  "
            f"binder={self.binder_viscosity_stress:.3f}  "
            f"volt={self.voltage_stress:.3f}"
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
    A bounded random walk pre-generated at initialisation time.
    Each step moves by a small random delta, clamped so the value
    never leaves [set_point - amplitude, set_point + amplitude].
    The full series is stored in an array; step(t) just indexes into it,
    so the chart always reads from fixed data rather than sampling live.
    """

    OPTIMAL_LOW  = 18.0   # °C
    OPTIMAL_HIGH = 25.0   # °C
    WARNING      = 35.0   # °C
    CRITICAL     = 40.0   # °C

    _PREGENERATE = 10_000  # steps generated at startup

    def __init__(
        self,
        set_point: float = 21.0,
        amplitude: float = 3.0,   # °C  — max deviation from set-point
        step_size: float = 0.3,   # °C  — max random walk delta per step
        seed:      int   = 0,
    ):
        lo, hi = set_point - amplitude, set_point + amplitude
        rng  = np.random.default_rng(seed)
        data = np.empty(self._PREGENERATE)
        data[0] = set_point
        for i in range(1, self._PREGENERATE):
            delta   = rng.uniform(-step_size, step_size)
            data[i] = float(np.clip(data[i - 1] + delta, lo, hi))
        self._data = data

    def step(self, t: float) -> float:
        """Return pre-generated temperature at step index *t*."""
        return float(self._data[int(t) % self._PREGENERATE])


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

    _PREGENERATE = 10_000

    def __init__(self, work_rate: float = 1.0, seed: int = 3):
        """
        Parameters
        ----------
        work_rate : starting rate (cycles per day). The rate then drifts via a
                    bounded random walk so the chart shows realistic busy/idle variation.
        seed      : RNG seed for the pre-generated rate series.
        """
        rng   = np.random.default_rng(seed)
        rates = np.empty(self._PREGENERATE)
        rates[0] = float(np.clip(work_rate, 0.1, 2.0))
        for i in range(1, self._PREGENERATE):
            rates[i] = float(np.clip(rates[i - 1] + rng.uniform(-0.04, 0.04), 0.1, 2.0))
        self._rates      = rates
        self._cumulative = 0.0
        self._idx        = 0

    def step(self, dt: float = 1.0) -> float:
        """Advance by *dt* days and return the cumulative cycle count."""
        self._cumulative += self._rates[self._idx % self._PREGENERATE] * dt
        self._idx += 1
        return self._cumulative

    @property
    def current_rate(self) -> float:
        """Instantaneous work rate at the last completed day (cycles / day)."""
        return float(self._rates[(self._idx - 1) % self._PREGENERATE])

    @property
    def cycles(self) -> float:
        return self._cumulative

    def reset(self) -> None:
        self._cumulative = 0.0
        self._idx        = 0


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
        service_interval: float = 500.0,    # days between services
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


class PowderQualityDriver:
    """Feedstock powder quality index (0–1, 1 = fresh, 0 = fully recycled/degraded).

    Physics rationale
    -----------------
    Metal powder is recycled between builds.  Each cycle slightly degrades the
    particle-size distribution and increases oxide contamination on the surface:
      • Irregular particles → higher blade contact force → faster Archard wear
      • Oxidised surfaces   → weaker binder adhesion → increased nozzle strain

    Model
    -----
    Quality decays proportionally to print cycles (recycling degrades the powder)
    and recovers proportionally to maintenance level (fresh powder injection during
    scheduled service).  A Gaussian noise term models batch-to-batch variability.
    """

    GOOD     = 0.80
    ADEQUATE = 0.60
    POOR     = 0.40

    def __init__(
        self,
        initial_quality:  float = 0.95,
        degrade_per_cycle: float = 3.0e-4,  # quality lost per print cycle
        restore_rate:     float = 3.0e-3,   # fraction of quality gap restored per maintenance unit
        noise_std:        float = 0.004,
        seed:             int   = 4,
    ):
        self._quality    = float(np.clip(initial_quality, 0.0, 1.0))
        self._degrade    = degrade_per_cycle
        self._restore    = restore_rate
        self._noise_std  = noise_std
        self._rng        = np.random.default_rng(seed)
        self._prev_load  = 0.0

    def step(self, operational_load: float, maintenance_level: float) -> float:
        """Return powder quality index (0–1)."""
        delta_cycles     = max(0.0, operational_load - self._prev_load)
        self._prev_load  = operational_load

        self._quality -= self._degrade * delta_cycles
        self._quality += self._restore * maintenance_level * (1.0 - self._quality)
        self._quality += float(self._rng.normal(0.0, self._noise_std))
        self._quality  = float(np.clip(self._quality, 0.0, 1.0))
        return self._quality


class BinderViscosityDriver:
    """Binder age / viscosity stress index (0–1, 0 = optimal, 1 = max stress).

    Physics rationale
    -----------------
    The liquid binder thickens progressively as it ages (solvent evaporation,
    polymerisation).  Viscosity too high or too low impairs jetting:
      • High viscosity → increased firing pressure → thermal fatigue on nozzle plate
      • Cold temperatures amplify this because viscosity rises sharply with cooling

    Model
    -----
    A slowly accumulating base stress (ageing) with a temperature-dependent
    transient offset.  Maintenance resets the base by replacing the cartridge.
    The returned value includes both the accumulated base and the transient
    temperature term so the degradation engine sees the full instantaneous stress.
    """

    NOMINAL  = 0.10
    WARNING  = 0.35
    CRITICAL = 0.60

    _OPTIMAL_TEMP_LOW  = 18.0
    _OPTIMAL_TEMP_HIGH = 25.0

    def __init__(
        self,
        baseline:         float = 0.05,
        age_per_cycle:    float = 6.0e-4,   # stress added per print cycle
        clean_rate:       float = 4.0e-4,   # stress removed per maintenance unit per day
        temp_cold_k:      float = 0.020,    # extra stress per °C below optimal (cold thickens)
        temp_hot_k:       float = 0.008,    # extra stress per °C above optimal (hot thins → spray issues)
        noise_std:        float = 0.005,
        seed:             int   = 5,
    ):
        self._stress      = baseline
        self._age         = age_per_cycle
        self._clean       = clean_rate
        self._temp_cold_k = temp_cold_k
        self._temp_hot_k  = temp_hot_k
        self._noise_std   = noise_std
        self._rng         = np.random.default_rng(seed)
        self._prev_load   = 0.0

    def step(self, temperature: float, operational_load: float, maintenance_level: float) -> float:
        """Return binder viscosity stress index (0–1)."""
        delta_cycles    = max(0.0, operational_load - self._prev_load)
        self._prev_load = operational_load

        self._stress += self._age * delta_cycles
        self._stress -= self._clean * maintenance_level
        self._stress += float(self._rng.normal(0.0, self._noise_std))
        self._stress  = float(np.clip(self._stress, 0.0, 1.0))

        cold_excess = max(0.0, self._OPTIMAL_TEMP_LOW  - temperature)
        hot_excess  = max(0.0, temperature - self._OPTIMAL_TEMP_HIGH)
        temp_offset = self._temp_cold_k * cold_excess + self._temp_hot_k * hot_excess

        return float(np.clip(self._stress + temp_offset, 0.0, 1.0))


class VoltageFluctuationDriver:
    """Power-supply instability index (0–1, 0 = stable, 1 = max fluctuation).

    Physics rationale
    -----------------
    Voltage spikes and sags cause transient Joule heating in the heating elements,
    accelerating electromigration and oxide-layer growth — the primary mechanisms
    modelled by the Arrhenius degradation equation.  Sustained instability also
    stresses the nozzle-plate firing circuitry.

    Model
    -----
    A mean-reverting random walk around a low baseline, with rare large spikes
    drawn from an exponential distribution to model grid transients or brownout
    events.  No dependency on maintenance (power supply is an external factor).
    """

    STABLE   = 0.05
    WARNING  = 0.30
    CRITICAL = 0.60

    def __init__(
        self,
        baseline:    float = 0.05,
        revert_rate: float = 0.12,   # fraction per day to pull back toward baseline
        noise_std:   float = 0.015,
        spike_prob:  float = 0.02,   # probability of a spike each day
        spike_scale: float = 0.35,   # exponential scale of spike magnitude
        seed:        int   = 6,
    ):
        self._stress      = baseline
        self._baseline    = baseline
        self._revert      = revert_rate
        self._noise_std   = noise_std
        self._spike_prob  = spike_prob
        self._spike_scale = spike_scale
        self._rng         = np.random.default_rng(seed)

    def step(self) -> float:
        """Return voltage stress index (0–1)."""
        self._stress += self._revert * (self._baseline - self._stress)
        if self._rng.random() < self._spike_prob:
            self._stress += float(self._rng.exponential(self._spike_scale))
        self._stress += float(self._rng.normal(0.0, self._noise_std))
        self._stress  = float(np.clip(self._stress, 0.0, 1.0))
        return self._stress


# ── Aggregate suite ───────────────────────────────────────────────────────────

class DriverSuite:
    """Coordinates all seven drivers and emits a DriverSnapshot each tick.

    Evaluation order inside `step()` resolves inter-driver dependencies:
      1. OperationalLoad       — no dependencies
      2. MaintenanceLevel      — no dependencies
      3. Temperature           — no dependencies
      4. Contamination         — depends on load + maintenance
      5. PowderQuality         — depends on load + maintenance
      6. BinderViscosity       — depends on temperature + load + maintenance
      7. VoltageFluctuation    — no dependencies

    Usage
    -----
    ::

        suite = DriverSuite(seed=42)
        for t in range(2000):
            snap = suite.step(t)
            engine.tick(snap)           # pass to Phase 1 degradation engine
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
        self.load          = OperationalLoadDriver(work_rate=work_rate, seed=seed + 3)
        self.maintenance   = MaintenanceLevelDriver(
            service_interval=service_interval,
            seed=seed + 2,
        )
        self.powder_quality   = PowderQualityDriver(seed=seed + 4)
        self.binder_viscosity = BinderViscosityDriver(seed=seed + 5)
        self.voltage          = VoltageFluctuationDriver(seed=seed + 6)
        self._t = 0.0

    def tick(self, dt: float = 1.0) -> DriverSnapshot:
        """Auto-incrementing step — no need to track *t* externally.

        Intended for real-time use where a timer calls this once per interval.
        The internal clock advances by *dt* each call.
        """
        snap = self.step(self._t, dt)
        self._t += dt
        return snap

    def step(self, t: float, dt: float = 1.0) -> DriverSnapshot:
        """Advance all drivers by one timestep and return a DriverSnapshot.

        Parameters
        ----------
        t  : current simulation time in days
        dt : elapsed time since last step (default 1.0)
        """
        op_load = self.load.step(dt)
        maint   = self.maintenance.step(t)
        temp    = self.temperature.step(t)
        contam  = self.contamination.step(op_load, maint)
        powder  = self.powder_quality.step(op_load, maint)
        binder  = self.binder_viscosity.step(temp, op_load, maint)
        voltage = self.voltage.step()

        return DriverSnapshot(
            temperature_stress=temp,
            humidity_contamination=contam,
            operational_load=op_load,
            maintenance_level=maint,
            powder_quality=powder,
            binder_viscosity_stress=binder,
            voltage_stress=voltage,
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
