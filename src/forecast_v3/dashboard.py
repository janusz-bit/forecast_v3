"""Read results saved by the modeling notebook."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

DASHBOARD_FILES = (
    "data/processed/daily_sales.csv",
    "outputs/protocol.json",
    "outputs/predictions.csv",
    "outputs/metrics.csv",
)


def load_dashboard_data(
    project_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load history and results for both scenarios saved by the notebook."""
    daily_path, protocol_path, predictions_path, metrics_path = (
        project_root / name for name in DASHBOARD_FILES
    )
    daily = pd.read_csv(daily_path, parse_dates=["Date"]).set_index("Date")
    predictions = pd.read_csv(predictions_path, parse_dates=["Date"]).set_index("Date")
    metrics = pd.read_csv(metrics_path).set_index("model")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    return daily, predictions, metrics, protocol
