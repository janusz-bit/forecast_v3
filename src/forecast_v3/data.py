"""Read the source CSV for the notebook."""

from pathlib import Path

import pandas as pd


def load_sales_data(path: str | Path) -> pd.DataFrame:
    """Load sales and parse the Date column as dates."""
    return pd.read_csv(path, parse_dates=["Date"])
