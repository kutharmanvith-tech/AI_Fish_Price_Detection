from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from forecasting import prepare_data, train_model


ROOT = Path(__file__).parent


def train_from_csv(csv_path: Path = ROOT / "data" / "fish_prices.csv") -> dict[str, float | int]:
    data = prepare_data(pd.read_csv(csv_path))
    report = train_model(data)
    model_path = ROOT / "models" / "coastal_karnataka_price_model.joblib"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(report.model, model_path)
    metrics = {"mae": round(report.mae, 2), "r2": round(report.r2, 4), "train_rows": report.train_rows, "test_rows": report.test_rows}
    (ROOT / "models" / "metrics.json").write_text(__import__("json").dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Coastal Karnataka fish price model")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "fish_prices.csv")
    args = parser.parse_args()
    print(train_from_csv(args.csv))