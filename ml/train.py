"""Train the RUL model from sim_details_*.csv files and save to ml/models/.

Usage:
    python ml/train.py [--data-dir PATH] [--output PATH]
"""
import argparse
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from ml.data import load_training_data, get_features_and_target
from ml.model import RULModel


def train(data_dir: pathlib.Path | None = None,
          output: pathlib.Path | None = None) -> RULModel:
    print("Loading training data...")
    df = load_training_data(data_dir)
    print(f"  {len(df):,} samples  |  components: {df['component_label'].nunique()}")

    X, y, feature_names = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    print(f"Training on {len(X_train):,} samples, evaluating on {len(X_test):,}...")
    model = RULModel()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2  = r2_score(y_test, preds)
    print(f"\n  MAE : {mae:.1f} days")
    print(f"  R²  : {r2:.4f}")

    print("\nFeature importances (permutation, subsample):")
    sub = min(2000, len(X_test))
    idx = np.random.default_rng(0).choice(len(X_test), sub, replace=False)
    importances = model.feature_importances(X_test[idx], y_test[idx], n_repeats=3)
    for name, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * max(0, int(imp * 200))
        print(f"  {name:<30} {imp:.4f}  {bar}")

    save_path = output or (pathlib.Path(__file__).parent / "models" / "rul_model.joblib")
    model.save(save_path)
    print(f"\nModel saved -> {save_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RUL model")
    parser.add_argument("--data-dir", type=pathlib.Path, default=None,
                        help="Directory containing sim_details_*.csv (default: project root)")
    parser.add_argument("--output", type=pathlib.Path, default=None,
                        help="Output path for saved model joblib")
    args = parser.parse_args()
    train(args.data_dir, args.output)
