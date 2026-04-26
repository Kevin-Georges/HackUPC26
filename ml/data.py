"""Load and prepare training data from sim_details_*.csv files.

CSV schema (wide):
    run_id, day, temperature_c, humidity_index, operational_load,
    maintenance_level, powder_quality, binder_viscosity_stress, voltage_stress,
    recoater_blade_pct, nozzle_plate_pct, heating_elements_pct,
    drive_motor_rails_pct, cleaning_thermal_iface_pct, insulation_sensors_pct

Health values are strings like "99%".  RUL is derived per (run_id, component)
as (failure_day - current_day), where failure_day = first day health_pct <= 30.
Rows from runs where a component never fails are dropped (right-censored).
"""
import pathlib
import glob

import numpy as np
import pandas as pd

# --- column → display label --------------------------------------------------
COMPONENT_COLS = {
    "recoater_blade_pct":       "Recoater Blade",
    "nozzle_plate_pct":         "Nozzle Plate",
    "heating_elements_pct":     "Heating Elements",
    "drive_motor_rails_pct":    "Drive Motor & Rails",
    "cleaning_thermal_iface_pct": "Cleaning & Thermal Iface",
    "insulation_sensors_pct":   "Insulation & Sensors",
}

COMPONENT_LABELS = list(COMPONENT_COLS.values())

ENV_FEATURES = [
    "temperature_c",
    "humidity_index",
    "operational_load",
    "maintenance_level",
    "powder_quality",
    "binder_viscosity_stress",
    "voltage_stress",
]

FEATURE_RANGES = {
    "temperature_c":           (15.0, 35.0),
    "humidity_index":          (0.0,  1.0),
    "operational_load":        (0.0,  3.0),
    "maintenance_level":       (0.0,  1.0),
    "powder_quality":          (0.5,  1.0),
    "binder_viscosity_stress": (0.0,  0.5),
    "voltage_stress":          (0.0,  0.5),
    "health_pct":              (1.0,  100.0),
}

ALL_FEATURES = ENV_FEATURES + ["health_pct", "component_id"]
FAILURE_THRESHOLD = 80  # health_pct < this → component exits healthy zone (green→degrading)


def _find_csvs(data_dir: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Return sorted list of sim_details_*.csv paths."""
    if data_dir is None:
        # default: project root (parent of ml/)
        data_dir = pathlib.Path(__file__).parent.parent
    paths = sorted(pathlib.Path(data_dir).glob("sim_details_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No sim_details_*.csv found in {data_dir}")
    return paths


def load_training_data(data_dir: pathlib.Path | None = None) -> pd.DataFrame:
    """Load all CSVs and return a long-format DataFrame with RUL column.

    Columns: run_id, day, temperature_c, humidity_index, operational_load,
             maintenance_level, powder_quality, binder_viscosity_stress,
             voltage_stress, component_label, health_pct, component_id, RUL
    """
    paths = _find_csvs(data_dir)
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # strip "%" and convert to int
        for col in COMPONENT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.rstrip("%").astype(int)
        frames.append(df)

    wide = pd.concat(frames, ignore_index=True)

    # melt health columns to long format
    id_vars = ["run_id", "day"] + ENV_FEATURES
    value_vars = [c for c in COMPONENT_COLS if c in wide.columns]
    long = wide.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name="component_col",
        value_name="health_pct",
    )
    long["component_label"] = long["component_col"].map(COMPONENT_COLS)
    long["component_id"] = long["component_label"].map(
        {lbl: i for i, lbl in enumerate(COMPONENT_LABELS)}
    )

    # compute RUL per (run_id, component_col)
    long = long.sort_values(["run_id", "component_col", "day"])
    long["RUL"] = _compute_rul(long)

    # drop right-censored (component never failed in this run)
    long = long.dropna(subset=["RUL"])
    long["RUL"] = long["RUL"].astype(int)

    return long.reset_index(drop=True)


def _compute_rul(df: pd.DataFrame) -> pd.Series:
    """Return RUL series aligned to df; NaN for right-censored rows."""
    rul_values = np.full(len(df), np.nan)

    for (run_id, comp_col), grp in df.groupby(["run_id", "component_col"], sort=False):
        failed = grp[grp["health_pct"] < FAILURE_THRESHOLD]
        if failed.empty:
            continue  # right-censored — leave as NaN
        failure_day = int(failed["day"].min())
        mask = grp.index
        rul_values[df.index.get_indexer(mask)] = (failure_day - grp["day"].values).clip(min=0)

    return pd.Series(rul_values, index=df.index)


def get_features_and_target(df: pd.DataFrame):
    """Return (X: ndarray, y: ndarray, feature_names: list[str])."""
    X = df[ALL_FEATURES].to_numpy(dtype=np.float32)
    y = df["RUL"].to_numpy(dtype=np.float32)
    return X, y, ALL_FEATURES
