from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from acm import NominalACM
from build_moex_curve import DEFAULT_MONTHS, build_moex_curve
from ruonia import build_ruonia_monthly, load_ruonia_1m


CLEAN_DIR = Path(__file__).resolve().parent
DEFAULT_DYNAMIC_PATH = CLEAN_DIR / "data" / "dynamic.csv"
DEFAULT_DAILY_OUTPUT = CLEAN_DIR / "output" / "moex_curve_daily.csv"
DEFAULT_MONTHLY_OUTPUT = CLEAN_DIR / "output" / "moex_curve_monthly.csv"

SHORT_RATE_PROXY_CURVE_1M = "curve_1m"
SHORT_RATE_PROXY_RUONIA_1M = "ruonia_1m"


@dataclass(frozen=True)
class ACMRunConfig:
    selected_maturities: list[int] | None
    date_from: pd.Timestamp | None
    date_to: pd.Timestamp | None
    short_rate_proxy_name: str
    n_factors: int
    months: list[int]


@dataclass(frozen=True)
class ACMRunResult:
    config: ACMRunConfig
    all_monthly_curve: pd.DataFrame
    monthly_curve: pd.DataFrame
    yield_curve: pd.DataFrame
    short_rate_proxy: pd.Series
    model: NominalACM
    term_premium_frame: pd.DataFrame
    summary: pd.DataFrame


def tenor_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in df.columns if re.fullmatch(r"M\d{3}", col)],
        key=lambda col: int(col[1:]),
    )


def to_acm_curve(curve_df: pd.DataFrame) -> pd.DataFrame:
    df = curve_df.copy()
    df["date"] = pd.to_datetime(df["month_end"])
    df = df.set_index("date").sort_index()

    curve_cols = tenor_columns(df)
    if not curve_cols:
        raise ValueError("No tenor columns found in the input curve")

    yield_curve = df[curve_cols].copy()
    yield_curve.columns = [int(col[1:]) for col in curve_cols]
    yield_curve = np.log1p(yield_curve / 100.0)
    yield_curve.index.name = "date"
    yield_curve.columns.name = "maturity_months"
    return yield_curve


def build_term_premium_frame(yield_curve: pd.DataFrame, acm: NominalACM) -> pd.DataFrame:
    months = sorted(set(yield_curve.columns) & set(acm.miy.columns) & set(acm.rny.columns) & set(acm.tp.columns))
    arrays: dict[str, object] = {"date": yield_curve.index}

    for month in months:
        arrays[f"obs_M{month:03d}"] = yield_curve[month].to_numpy(dtype=float) * 100.0
        arrays[f"fit_M{month:03d}"] = acm.miy[month].to_numpy(dtype=float) * 100.0
        arrays[f"rn_M{month:03d}"] = acm.rny[month].to_numpy(dtype=float) * 100.0
        arrays[f"tp_M{month:03d}"] = acm.tp[month].to_numpy(dtype=float) * 100.0

    return pd.DataFrame(arrays)


def slice_monthly_curve(
    monthly_curve: pd.DataFrame,
    date_from: str | pd.Timestamp | None = None,
    date_to: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = monthly_curve.copy()
    df["month_end"] = pd.to_datetime(df["month_end"])

    start = None if date_from is None else pd.to_datetime(date_from)
    end = None if date_to is None else pd.to_datetime(date_to)

    if start is not None:
        df = df.loc[df["month_end"] >= start]
    if end is not None:
        df = df.loc[df["month_end"] <= end]

    df = df.sort_values("month_end").reset_index(drop=True)
    if df.empty:
        raise ValueError("The requested date range produced an empty monthly curve")

    df["month"] = df["month_end"].dt.to_period("M").astype(str)
    return df


def normalize_selected_maturities(
    selected_maturities: Sequence[int] | str | None,
    available_maturities: Sequence[int],
) -> list[int] | None:
    if selected_maturities is None:
        return None

    if isinstance(selected_maturities, str):
        values = [int(part.strip()) for part in selected_maturities.split(",") if part.strip()]
    else:
        values = [int(value) for value in selected_maturities]

    if not values:
        raise ValueError("`selected_maturities` must not be empty")

    available = set(int(value) for value in available_maturities)
    missing = [value for value in values if value not in available]
    if missing:
        raise ValueError(
            "`selected_maturities` contains maturities that are not available "
            f"in the curve: {missing[:10]}"
        )

    return values


def resolve_short_rate_proxy(
    short_rate_proxy: str | pd.Series | pd.DataFrame | None,
    date_from: str | pd.Timestamp | None = None,
    date_to: str | pd.Timestamp | None = None,
) -> tuple[str, pd.Series | pd.DataFrame | None]:
    if short_rate_proxy is None:
        return SHORT_RATE_PROXY_CURVE_1M, None

    if isinstance(short_rate_proxy, str):
        proxy_name = short_rate_proxy.lower()

        if proxy_name == SHORT_RATE_PROXY_CURVE_1M:
            return SHORT_RATE_PROXY_CURVE_1M, None

        if proxy_name in {SHORT_RATE_PROXY_RUONIA_1M, "ruonia"}:
            proxy = build_ruonia_monthly(load_ruonia_1m()).set_index("month_end")["short_rate_monthly_cc"]
            return SHORT_RATE_PROXY_RUONIA_1M, slice_proxy(proxy, date_from=date_from, date_to=date_to)

        raise ValueError(
            "`short_rate_proxy` string must be one of: "
            f"`{SHORT_RATE_PROXY_CURVE_1M}`, `{SHORT_RATE_PROXY_RUONIA_1M}`"
        )

    if isinstance(short_rate_proxy, (pd.Series, pd.DataFrame)):
        return "custom", slice_proxy(short_rate_proxy, date_from=date_from, date_to=date_to)

    raise TypeError("`short_rate_proxy` must be None, a string, a Series, or a DataFrame")


def slice_proxy(
    proxy: pd.Series | pd.DataFrame,
    date_from: str | pd.Timestamp | None = None,
    date_to: str | pd.Timestamp | None = None,
) -> pd.Series | pd.DataFrame:
    obj = proxy.copy()

    if not isinstance(obj.index, pd.DatetimeIndex):
        raise TypeError("`short_rate_proxy` must have a DatetimeIndex")

    start = None if date_from is None else pd.to_datetime(date_from)
    end = None if date_to is None else pd.to_datetime(date_to)

    obj = obj.sort_index()
    if start is not None:
        obj = obj.loc[obj.index >= start]
    if end is not None:
        obj = obj.loc[obj.index <= end]

    if len(obj) == 0:
        raise ValueError("The requested date range produced an empty short-rate proxy")

    return obj


def build_summary(
    yield_curve: pd.DataFrame,
    selected_maturities: list[int] | None,
    proxy_name: str,
    n_factors: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "metric": [
                "sample_start",
                "sample_end",
                "n_monthly_obs",
                "n_curve_maturities",
                "n_selected_maturities",
                "n_factors",
                "short_rate_proxy",
            ],
            "value": [
                yield_curve.index.min().date(),
                yield_curve.index.max().date(),
                len(yield_curve),
                len(yield_curve.columns),
                len(selected_maturities) if selected_maturities is not None else len(yield_curve.columns),
                n_factors,
                proxy_name,
            ],
        }
    )


def run_acm_model(
    selected_maturities: Sequence[int] | str | None = None,
    date_from: str | pd.Timestamp | None = None,
    date_to: str | pd.Timestamp | None = None,
    short_rate_proxy: str | pd.Series | pd.DataFrame | None = None,
    n_factors: int = 5,
    months: Sequence[int] = DEFAULT_MONTHS,
    dynamic_path: str | Path = DEFAULT_DYNAMIC_PATH,
    daily_output_path: str | Path = DEFAULT_DAILY_OUTPUT,
    monthly_output_path: str | Path = DEFAULT_MONTHLY_OUTPUT,
) -> ACMRunResult:
    """
    Run the project ACM model on the MOEX monthly curve with optional
    date-range filtering and short-rate proxy override.

    Parameters
    ----------
    selected_maturities
        Maturities to use in the ACM excess-return regression.
    date_from, date_to
        Inclusive date boundaries for the monthly estimation sample.
    short_rate_proxy
        Either:
        - None / "curve_1m": use the first curve maturity
        - "ruonia_1m" or "ruonia": use RUONIA 1M proxy
        - pandas.Series / single-column pandas.DataFrame with DatetimeIndex
    n_factors
        Number of PCA factors.
    months
        Full curve tenors to build before filtering by date.
    """

    months = [int(month) for month in months]
    _, all_monthly_curve = build_moex_curve(
        dynamic_path=dynamic_path,
        daily_output_path=daily_output_path,
        monthly_output_path=monthly_output_path,
        months=months,
    )

    monthly_curve = slice_monthly_curve(
        monthly_curve=all_monthly_curve,
        date_from=date_from,
        date_to=date_to,
    )
    yield_curve = to_acm_curve(monthly_curve)

    normalized_selected_maturities = normalize_selected_maturities(
        selected_maturities=selected_maturities,
        available_maturities=yield_curve.columns,
    )
    proxy_name, resolved_proxy = resolve_short_rate_proxy(
        short_rate_proxy=short_rate_proxy,
        date_from=date_from,
        date_to=date_to,
    )

    model = NominalACM(
        curve=yield_curve,
        n_factors=n_factors,
        selected_maturities=normalized_selected_maturities,
        short_rate_proxy=resolved_proxy,
    )
    term_premium_frame = build_term_premium_frame(yield_curve=yield_curve, acm=model)
    summary = build_summary(
        yield_curve=yield_curve,
        selected_maturities=normalized_selected_maturities,
        proxy_name=proxy_name,
        n_factors=n_factors,
    )

    config = ACMRunConfig(
        selected_maturities=normalized_selected_maturities,
        date_from=None if date_from is None else pd.to_datetime(date_from),
        date_to=None if date_to is None else pd.to_datetime(date_to),
        short_rate_proxy_name=proxy_name,
        n_factors=n_factors,
        months=months,
    )

    return ACMRunResult(
        config=config,
        all_monthly_curve=all_monthly_curve,
        monthly_curve=monthly_curve,
        yield_curve=yield_curve,
        short_rate_proxy=model.short_rate_proxy,
        model=model,
        term_premium_frame=term_premium_frame,
        summary=summary,
    )

