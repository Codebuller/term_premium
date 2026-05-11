from __future__ import annotations

from collections.abc import Iterable, Sequence
import re

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from run_acm_model import to_acm_curve


DEFAULT_MONTHS = tuple(range(1, 181))


def sparse_tenor_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if re.fullmatch(r"y\d+(\.\d+)?", col)],
        key=lambda col: float(col[1:]),
    )


def dense_tenor_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if re.fullmatch(r"M\d{3}", col)],
        key=lambda col: int(col[1:]),
    )


def normalize_sparse_curve_frame(curve_df: pd.DataFrame) -> pd.DataFrame:
    df = curve_df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        for candidate in ("month_end", "tradedate", "date", "datetime"):
            if candidate in df.columns:
                df["date"] = pd.to_datetime(df[candidate])
                df = df.set_index("date")
                break
        else:
            raise TypeError("`curve_df` must have a DatetimeIndex or a date-like column")

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    curve_cols = sparse_tenor_columns(df)
    if not curve_cols:
        raise ValueError("No sparse `y...` tenor columns were found")

    out = df[curve_cols].apply(pd.to_numeric, errors="coerce")
    out.index.name = "date"
    return out


def _observed_maturity_years(curve_cols: Sequence[str]) -> np.ndarray:
    return np.asarray([float(col[1:]) for col in curve_cols], dtype=float)


def _target_maturity_years(months: Sequence[int]) -> np.ndarray:
    return np.asarray(months, dtype=float) / 12.0


def _yields_pct_to_log_prices(yields_pct: np.ndarray, maturity_years: np.ndarray) -> np.ndarray:
    continuous_yields = np.log1p(yields_pct / 100.0)
    return -(continuous_yields * maturity_years)


def _log_prices_to_yields_pct(log_prices: np.ndarray, maturity_years: np.ndarray) -> np.ndarray:
    continuous_yields = -log_prices / maturity_years
    return np.expm1(continuous_yields) * 100.0


def _interpolate_log_prices_row(
    observed_years: np.ndarray,
    observed_log_prices: np.ndarray,
    target_years: np.ndarray,
    method: str,
) -> np.ndarray:
    valid = np.isfinite(observed_log_prices)
    x = observed_years[valid]
    y = observed_log_prices[valid]

    if len(x) < 2:
        raise ValueError("Each row must have at least two finite sparse maturities")

    if method == "pchip" and len(x) >= 3:
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        interpolated = interpolator(target_years)
    elif method == "linear":
        interpolated = np.interp(target_years, x, y)
    else:
        raise ValueError("`method` must be either `pchip` or `linear`")

    left_mask = target_years < x[0]
    if left_mask.any():
        # Keep the first observed zero yield flat down to the origin.
        interpolated[left_mask] = y[0] * (target_years[left_mask] / x[0])

    right_mask = target_years > x[-1]
    if right_mask.any():
        # Keep the last observed zero yield flat beyond the last sparse tenor.
        interpolated[right_mask] = y[-1] * (target_years[right_mask] / x[-1])

    return interpolated


def expand_sparse_zero_curve(
    curve_df: pd.DataFrame,
    months: Sequence[int] = DEFAULT_MONTHS,
    method: str = "pchip",
) -> pd.DataFrame:
    sparse_curve = normalize_sparse_curve_frame(curve_df)
    curve_cols = sparse_tenor_columns(sparse_curve)

    observed_years = _observed_maturity_years(curve_cols)
    target_months = list(int(month) for month in months)
    target_years = _target_maturity_years(target_months)

    observed_yields = sparse_curve[curve_cols].to_numpy(dtype=float)
    observed_log_prices = _yields_pct_to_log_prices(observed_yields, observed_years[None, :])

    interpolated_log_prices = np.empty((len(sparse_curve.index), len(target_months)), dtype=float)

    for idx, row in enumerate(observed_log_prices):
        interpolated_log_prices[idx, :] = _interpolate_log_prices_row(
            observed_years=observed_years,
            observed_log_prices=row,
            target_years=target_years,
            method=method,
        )

    interpolated_yields = _log_prices_to_yields_pct(interpolated_log_prices, target_years[None, :])
    dense_columns = [f"M{month:03d}" for month in target_months]
    dense_curve = pd.DataFrame(interpolated_yields, index=sparse_curve.index, columns=dense_columns)
    dense_curve.index.name = "date"

    # Preserve exact knot values wherever the sparse tenor lands on an integer month.
    for curve_col, maturity_years in zip(curve_cols, observed_years):
        month = int(round(maturity_years * 12))
        dense_col = f"M{month:03d}"
        if dense_col in dense_curve.columns:
            dense_curve[dense_col] = sparse_curve[curve_col].to_numpy(dtype=float)

    return dense_curve.reset_index()


def expand_sparse_zero_curve_acm(
    curve_df: pd.DataFrame,
    months: Sequence[int] = DEFAULT_MONTHS,
    method: str = "pchip",
) -> pd.DataFrame:
    dense_cols = dense_tenor_columns(curve_df if isinstance(curve_df, pd.DataFrame) else pd.DataFrame())

    if dense_cols:
        return to_acm_curve(curve_df)

    dense_curve = expand_sparse_zero_curve(curve_df=curve_df, months=months, method=method)
    return to_acm_curve(dense_curve)
