"""RUL prediction model wrapping HistGradientBoostingRegressor."""
import pathlib

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

from ml.data import COMPONENT_LABELS, ALL_FEATURES, FEATURE_RANGES

_DEFAULT_PATH = pathlib.Path(__file__).parent / "models" / "rul_model.joblib"


class RULModel:
    """Thin wrapper around HistGradientBoostingRegressor for RUL prediction.

    The last feature (component_id) is treated as categorical.
    """

    def __init__(self):
        n_features = len(ALL_FEATURES)
        self._reg = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=6,
            learning_rate=0.05,
            min_samples_leaf=20,
            categorical_features=[n_features - 1],  # component_id
            random_state=42,
        )
        self.feature_names: list[str] = ALL_FEATURES
        self.component_labels: list[str] = COMPONENT_LABELS
        self.feature_ranges: dict = FEATURE_RANGES
        self._trained = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RULModel":
        self._reg.fit(X, y)
        self._trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("Model not trained — call fit() or load() first")
        return np.clip(self._reg.predict(X), 0, None)

    def save(self, path: pathlib.Path | str = _DEFAULT_PATH) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: pathlib.Path | str = _DEFAULT_PATH) -> "RULModel":
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return joblib.load(path)

    @property
    def trained(self) -> bool:
        return self._trained

    def feature_importances(self, X: np.ndarray, y: np.ndarray,
                             n_repeats: int = 5) -> dict[str, float]:
        """Return permutation importances (mean decrease in MAE)."""
        if not self._trained:
            return {}
        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            self._reg, X, y,
            n_repeats=n_repeats,
            scoring="neg_mean_absolute_error",
            random_state=42,
        )
        return dict(zip(self.feature_names, result.importances_mean))
