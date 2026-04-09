from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
DATA_DIR = CLEAN_DIR / "data"
OUTPUT_DIR = CLEAN_DIR / "output"
RUONIA_1M_PATH = DATA_DIR / "ruonia_1M.csv"
RUONIA_MONTHLY_PATH = OUTPUT_DIR / "ruonia_1m_monthly.csv"
def load_ruonia_1m(path: str | Path = RUONIA_1M_PATH) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.rename(columns={"Дата": "date", "1 месяц": "ruonia_1m_pct"})
    if "date" not in raw or "ruonia_1m_pct" not in raw:
        raise ValueError("RUONIA 1M file must contain 'Дата' and '1 месяц' columns")

    raw["date"] = pd.to_datetime(raw["date"], format="%m/%d/%y")
    raw["ruonia_1m_pct"] = pd.to_numeric(raw["ruonia_1m_pct"], errors="coerce")
    raw = raw.dropna(subset=["ruonia_1m_pct"]).sort_values("date").reset_index(drop=True)
    return raw[["date", "ruonia_1m_pct"]]


def build_ruonia_monthly(raw_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_df is None:
        raw_df = load_ruonia_1m()

    monthly = (
        raw_df.assign(month_end=raw_df["date"].dt.to_period("M").dt.to_timestamp("M"))
        .groupby("month_end", as_index=False)["ruonia_1m_pct"]
        .mean()
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    monthly["month"] = monthly["month_end"].dt.to_period("M").astype(str)
    monthly["short_rate_monthly_cc"] = np.log1p(monthly["ruonia_1m_pct"] / 100.0) / 12.0
    return monthly[["month", "month_end", "ruonia_1m_pct", "short_rate_monthly_cc"]]


def save_ruonia_monthly(monthly_df: pd.DataFrame, path: str | Path = RUONIA_MONTHLY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    monthly_df.to_csv(path, index=False)


def load_ruonia_monthly(path: str | Path = RUONIA_MONTHLY_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["month_end"])


def run_ruonia_monthly(
    raw_df: pd.DataFrame | None = None,
    output_path: str | Path = RUONIA_MONTHLY_PATH,
) -> pd.DataFrame:
    monthly_df = build_ruonia_monthly(raw_df=raw_df)
    save_ruonia_monthly(monthly_df=monthly_df, path=output_path)
    return monthly_df


def main() -> None:
    monthly_df = run_ruonia_monthly()
    print(f"saved {RUONIA_MONTHLY_PATH}: {monthly_df.shape}")


if __name__ == "__main__":
    main()
