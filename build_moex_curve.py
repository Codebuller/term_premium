from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


K_FACTOR = 1.6
A2 = 0.6
CLEAN_DIR = Path(__file__).resolve().parent
DEFAULT_DYNAMIC_PATH = CLEAN_DIR/ "data" / "dynamic.csv"
DEFAULT_DAILY_OUTPUT = CLEAN_DIR / "output" / "moex_curve_daily.csv"
DEFAULT_MONTHLY_OUTPUT = CLEAN_DIR / "output" / "moex_curve_monthly.csv"
DEFAULT_MONTHS = tuple(range(1, 181, 1))


@dataclass(frozen=True)
class CurveParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float
    g: np.ndarray


def fixed_nodes() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros(9, dtype=float)
    b = np.zeros(9, dtype=float)
    a[0] = 0.0
    a[1] = A2
    b[0] = A2
    for i in range(1, 9):
        b[i] = b[i - 1] * K_FACTOR
    for i in range(2, 9):
        a[i] = a[i - 1] + A2 * (K_FACTOR ** (i - 1))
    return a, b


def load_dynamic_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", skiprows=1, decimal=",")
    df["tradedate"] = pd.to_datetime(df["tradedate"], format="%d.%m.%Y")
    df["datetime"] = pd.to_datetime(
        df["tradedate"].dt.strftime("%Y-%m-%d") + " " + df["tradetime"]
    )
    return df.sort_values(["tradedate", "tradetime"]).reset_index(drop=True)


def _as_params(row: pd.Series) -> CurveParams:
    return CurveParams(
        beta0=float(row["B1"]),
        beta1=float(row["B2"]),
        beta2=float(row["B3"]),
        tau=float(row["T1"]),
        g=row[[f"G{i}" for i in range(1, 10)]].to_numpy(dtype=float),
    )


def zero_curve_percent(
    maturity_years: Sequence[float] | np.ndarray,
    params: CurveParams,
) -> np.ndarray:
    t = np.asarray(maturity_years, dtype=float)
    x = t / params.tau
    load_1 = np.where(x == 0.0, 1.0, (1.0 - np.exp(-x)) / x)
    load_2 = load_1 - np.exp(-x)
    nelson_siegel = params.beta0 + params.beta1 * load_1 + params.beta2 * load_2

    a, b = fixed_nodes()
    basis = np.exp(-((t[:, None] - a[None, :]) / b[None, :]) ** 2)
    continuous_bp = nelson_siegel + basis @ params.g
    continuous_rate = continuous_bp / 10000.0
    return (np.exp(continuous_rate) - 1.0) * 100.0


def tenor_columns(months: Iterable[int]) -> list[str]:
    return [f"M{month:03d}" for month in months]


def compute_daily_curve(dynamic_df: pd.DataFrame, months: Sequence[int]) -> pd.DataFrame:
    maturity_years = np.asarray(months, dtype=float) / 12.0
    columns = tenor_columns(months)
    records: list[dict[str, object]] = []

    for _, row in dynamic_df.iterrows():
        values = zero_curve_percent(maturity_years, _as_params(row))
        record: dict[str, object] = {
            "tradedate": row["tradedate"],
            "tradetime": row["tradetime"],
            "datetime": row["datetime"],
        }
        record.update(dict(zip(columns, values)))
        records.append(record)

    return pd.DataFrame.from_records(records)


def build_monthly_curve(daily_curve: pd.DataFrame, months: Sequence[int]) -> pd.DataFrame:
    columns = tenor_columns(months)
    monthly = (
        daily_curve.set_index("tradedate")[columns]
        .resample("ME")
        .mean()
        .rename_axis("month_end")
        .reset_index()
    )
    monthly["month"] = monthly["month_end"].dt.to_period("M").astype(str)
    return monthly[["month", "month_end", *columns]]


def build_moex_curve(
    dynamic_path: str | Path,
    daily_output_path: str | Path,
    monthly_output_path: str | Path,
    months: Sequence[int] = DEFAULT_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dynamic = load_dynamic_csv(dynamic_path)
    daily_curve = compute_daily_curve(dynamic, months)
    monthly_curve = build_monthly_curve(daily_curve, months)

    daily_output_path = Path(daily_output_path)
    monthly_output_path = Path(monthly_output_path)
    daily_output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_output_path.parent.mkdir(parents=True, exist_ok=True)

    daily_curve.to_csv(daily_output_path, index=False)
    monthly_curve.to_csv(monthly_output_path, index=False)
    return daily_curve, monthly_curve


def parse_months(value: str) -> list[int]:
    months = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not months:
        raise argparse.ArgumentTypeError("months must not be empty")
    if min(months) <= 0:
        raise argparse.ArgumentTypeError("months must be positive integers")
    return months

def main() -> None:
    months = DEFAULT_MONTHS
    daily_curve, monthly_curve = build_moex_curve(
        dynamic_path=DEFAULT_DYNAMIC_PATH,
        daily_output_path=DEFAULT_DAILY_OUTPUT,
        monthly_output_path=DEFAULT_MONTHLY_OUTPUT,
        months=months,
    )

    print(f"saved {Path(DEFAULT_DAILY_OUTPUT)}: {daily_curve.shape}")
    print(f"saved {Path(DEFAULT_MONTHLY_OUTPUT)}: {monthly_curve.shape}")
    print(f"months: {months[0]}..{months[-1]} ({len(months)} tenors)")


if __name__ == "__main__":
    main()
