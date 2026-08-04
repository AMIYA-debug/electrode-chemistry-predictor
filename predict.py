from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
FEATURES_PATH = MODELS_DIR / "feature_names.json"

CHEMISTRY_DISPLAY_ORDER = ["C", "Si", "Mn", "Cr", "Ni", "Mo", "Ti", "Al", "Nb", "S", "P"]


def load_feature_names() -> list[str]:
    
    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(
            f"Feature definition is missing: {FEATURES_PATH}. Run the notebook export cell first."
        )
    try:
        feature_names = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Feature definition is not valid JSON: {FEATURES_PATH}") from exc

    if not isinstance(feature_names, list) or not feature_names or not all(isinstance(name, str) for name in feature_names):
        raise ValueError("feature_names.json must be a non-empty JSON list of feature-name strings.")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("feature_names.json contains duplicate feature names.")
    return feature_names


def collect_formulation(feature_names: list[str]) -> pd.DataFrame:
    
    values: dict[str, float] = {}
    print("Enter the 4.8 mm electrode formulation values used in training:")

    for feature_name in feature_names:
        while True:
            raw_value = input(f"Enter {feature_name}: ").strip()
            if not raw_value:
                print("A value is required; blank values are not allowed.")
                continue
            try:
                value = float(raw_value)
            except ValueError:
                print("Please enter a numeric value, for example 0, 2.5, or 18.")
                continue
            if not math.isfinite(value):
                print("Please enter a finite numeric value (not NaN or infinity).")
                continue
            values[feature_name] = value
            break

    formulation = pd.DataFrame([[values[name] for name in feature_names]], columns=feature_names)
    if list(formulation.columns) != feature_names or formulation.isna().any().any():
        raise ValueError("Input feature order or values do not match the saved training definition.")
    return formulation


def find_saved_targets() -> list[Path]:
    
    if not MODELS_DIR.is_dir():
        raise FileNotFoundError(f"Models directory is missing: {MODELS_DIR}. Run the notebook export cell first.")
    target_by_name = {
        path.name: path
        for path in MODELS_DIR.iterdir()
        if path.is_dir() and (path / "model.pkl").is_file()
    }
    if not target_by_name:
        raise FileNotFoundError(f"No target model files were found beneath: {MODELS_DIR}")

    ordered_names = [name for name in CHEMISTRY_DISPLAY_ORDER if name in target_by_name]
    additional_names = sorted(set(target_by_name) - set(CHEMISTRY_DISPLAY_ORDER))
    return [target_by_name[name] for name in ordered_names + additional_names]


def predict_chemistry(formulation: pd.DataFrame) -> pd.DataFrame:
    
    predictions: list[dict[str, float | str]] = []
    for target_dir in find_saved_targets():
        model = joblib.load(target_dir / "model.pkl")
        scaler_path = target_dir / "scaler.pkl"

        model_input = formulation
        if scaler_path.is_file():
            scaler = joblib.load(scaler_path)
            if getattr(scaler, "n_features_in_", formulation.shape[1]) != formulation.shape[1]:
                raise ValueError(
                    f"Saved scaler for {target_dir.name} expects {scaler.n_features_in_} features, "
                    f"but feature_names.json defines {formulation.shape[1]}."
                )
            model_input = scaler.transform(formulation)

        if getattr(model, "n_features_in_", model_input.shape[1]) != model_input.shape[1]:
            raise ValueError(
                f"Saved model for {target_dir.name} expects {model.n_features_in_} features, "
                f"but the supplied formulation has {model_input.shape[1]}."
            )

        value = float(np.asarray(model.predict(model_input)).ravel()[0])
        predictions.append({"Element": target_dir.name, "Predicted value": value})

    return pd.DataFrame(predictions)


def main() -> None:
    feature_names = load_feature_names()
    formulation = collect_formulation(feature_names)
    predictions = predict_chemistry(formulation)

    print("\n" + "=" * 40)
    print("Predicted Weld Chemistry (4.8 mm)")
    print("=" * 40)
    print(predictions.to_string(index=False, formatters={"Predicted value": "{:.6f}".format}))


if __name__ == "__main__":
    main()
