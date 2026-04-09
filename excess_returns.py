from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ruonia import load_ruonia_monthly


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
CURVE_PATH = OUTPUT_DIR / "moex_curve_monthly.csv"
EXCESS_RETURNS_PATH = OUTPUT_DIR / "moex_curve_excess_returns.csv"
EXCESS_RETURNS_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_excess_returns_summary.csv"
DATE_COLUMNS = ["month_end"]
HOLDING_PERIOD_MONTHS = 1
USE_ALL_PRICING_TENORS = True
PRICING_MONTHS = list(range(12, 181, 12))
INTERPOLATE_MISSING_TENORS = True


def load_curve(path: str | Path = CURVE_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def curve_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [column for column in df.columns if column.startswith("M")],
        key=lambda column: int(column[1:]),
    )


def selected_pricing_months(
    curve_df: pd.DataFrame,
    pricing_months: list[int] | None = None,
    use_all_pricing_tenors: bool = USE_ALL_PRICING_TENORS,
    holding_period_months: int = HOLDING_PERIOD_MONTHS,
) -> list[int]:
    available_months = [int(column[1:]) for column in curve_columns(curve_df)]
    if use_all_pricing_tenors:
        selected = available_months
    else:
        configured = pricing_months if pricing_months is not None else PRICING_MONTHS
        selected = [month for month in configured if month in available_months]

    selected = sorted(month for month in selected if month > holding_period_months)
    if not selected:
        raise ValueError("No pricing tenors remain after applying the holding period filter")
    return selected


def effective_yields_to_cc(curve_df: pd.DataFrame, months: list[int]) -> np.ndarray:
    columns = [f"M{month:03d}" for month in months]
    yields_eff = curve_df[columns].to_numpy(dtype=float) / 100.0
    return np.log1p(yields_eff)


def build_log_price_matrix(yields_cc: np.ndarray, months: list[int]) -> np.ndarray:
    maturities_years = np.asarray(months, dtype=float) / 12.0
    return -yields_cc * maturities_years[None, :]


def interpolate_yields_cc(
    yields_cc: np.ndarray,
    source_months: list[int],
    target_months: list[int],
) -> np.ndarray:
    if source_months == target_months:
        return yields_cc

    interpolated = np.empty((yields_cc.shape[0], len(target_months)), dtype=float)
    for idx in range(yields_cc.shape[0]):
        interpolated[idx] = np.interp(target_months, source_months, yields_cc[idx])
    return interpolated


def build_funding_leg(short_rate_monthly_cc: np.ndarray, holding_period_months: int) -> np.ndarray:
    if holding_period_months == 1:
        return short_rate_monthly_cc[:-1]

    return np.array(
        [
            short_rate_monthly_cc[t : t + holding_period_months].sum()
            for t in range(len(short_rate_monthly_cc) - holding_period_months)
        ]
    )


def compute_excess_returns(
    curve_df: pd.DataFrame | None = None,
    yields_cc: np.ndarray | None = None,
    short_rate_monthly_cc: np.ndarray | None = None,
    short_rate_df: pd.DataFrame | None = None,
    available_months: list[int] | None = None,
    pricing_months: list[int] | None = None,
    holding_period_months: int = HOLDING_PERIOD_MONTHS,
    use_all_pricing_tenors: bool = USE_ALL_PRICING_TENORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if curve_df is None and yields_cc is None:
        curve_df = load_curve()

    if curve_df is not None:
        if short_rate_df is None:
            short_rate_df = load_ruonia_monthly()
        curve_df = curve_df.merge(
            short_rate_df[["month", "month_end", "short_rate_monthly_cc"]],
            on=["month", "month_end"],
            how="inner",
        )
        if curve_df.empty:
            raise ValueError("Curve and RUONIA 1M monthly data have no overlapping observations")
        source_months = [int(column[1:]) for column in curve_columns(curve_df)]
        pricing_months = selected_pricing_months(
            curve_df=curve_df,
            pricing_months=pricing_months,
            use_all_pricing_tenors=use_all_pricing_tenors,
            holding_period_months=holding_period_months,
        )
        if yields_cc is None:
            yields_cc = effective_yields_to_cc(curve_df=curve_df, months=source_months)
        if available_months is None:
            if INTERPOLATE_MISSING_TENORS:
                available_months = list(range(min(source_months), max(source_months) + 1))
            else:
                available_months = source_months
        yields_cc = interpolate_yields_cc(
            yields_cc=yields_cc,
            source_months=source_months,
            target_months=available_months,
        )
        if short_rate_monthly_cc is None:
            short_rate_monthly_cc = curve_df["short_rate_monthly_cc"].to_numpy(dtype=float)
    else:
        if available_months is None:
            raise ValueError("available_months is required when curve_df is not provided")
        if pricing_months is None:
            raise ValueError("pricing_months is required when curve_df is not provided")
        if short_rate_monthly_cc is None:
            raise ValueError("short_rate_monthly_cc is required when curve_df is not provided")

    if yields_cc is None or short_rate_monthly_cc is None or available_months is None or pricing_months is None:
        raise ValueError("Missing inputs for excess return calculation")

    if len(yields_cc) != len(short_rate_monthly_cc):
        raise ValueError("yields_cc and short_rate_monthly_cc must have the same number of observations")
    if len(yields_cc) <= holding_period_months:
        raise ValueError("Not enough observations for the requested holding period")
    if yields_cc.shape[1] != len(available_months):
        raise ValueError("The number of yield columns must match available_months")

    month_to_index = {month: idx for idx, month in enumerate(available_months)}
    start_maturities = np.asarray(pricing_months, dtype=int)
    valid_start_maturities = [
        month
        for month in start_maturities
        if month - holding_period_months in month_to_index
    ]
    if not valid_start_maturities:
        raise ValueError("No pricing tenors have matching end-of-holding maturities in available_months")

    start_maturities = np.asarray(valid_start_maturities, dtype=int)
    end_maturities = start_maturities - holding_period_months
    start_idx = np.asarray([month_to_index[month] for month in start_maturities], dtype=int)
    end_idx = np.asarray([month_to_index[month] for month in end_maturities], dtype=int)
    log_prices = build_log_price_matrix(yields_cc=yields_cc, months=available_months)
    funding_leg = build_funding_leg(
        short_rate_monthly_cc=short_rate_monthly_cc,
        holding_period_months=holding_period_months,
    )
    end_columns = [f"rx_M{month:03d}" for month in start_maturities]

    rx = (
        log_prices[holding_period_months:, end_idx]
        - log_prices[:-holding_period_months, start_idx]
        - funding_leg[:, None]
    )

    if curve_df is not None:
        panel_base = pd.DataFrame(
            {
                "start_month": curve_df["month"].iloc[:-holding_period_months].to_numpy(),
                "start_month_end": curve_df["month_end"].iloc[:-holding_period_months].to_numpy(),
                "end_month": curve_df["month"].iloc[holding_period_months:].to_numpy(),
                "end_month_end": curve_df["month_end"].iloc[holding_period_months:].to_numpy(),
            }
        )
    else:
        panel_base = pd.DataFrame(index=np.arange(len(rx)))

    panel = pd.concat([panel_base, pd.DataFrame(rx, columns=end_columns)], axis=1)
    summary = pd.DataFrame(
        {
            "tenor_months": start_maturities,
            "mean_bp": rx.mean(axis=0) * 10000.0,
            "std_bp": rx.std(axis=0, ddof=0) * 10000.0,
            "n_obs": len(rx),
            "holding_period_months": holding_period_months,
        }
    )
    return panel, summary


def save_excess_returns_results(
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    panel_path: str | Path = EXCESS_RETURNS_PATH,
    summary_path: str | Path = EXCESS_RETURNS_SUMMARY_PATH,
) -> None:
    panel_path = Path(panel_path)
    summary_path = Path(summary_path)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(panel_path, index=False)
    summary.to_csv(summary_path, index=False)


def run_excess_returns(
    curve_df: pd.DataFrame | None = None,
    yields_cc: np.ndarray | None = None,
    short_rate_monthly_cc: np.ndarray | None = None,
    short_rate_df: pd.DataFrame | None = None,
    available_months: list[int] | None = None,
    pricing_months: list[int] | None = None,
    holding_period_months: int = HOLDING_PERIOD_MONTHS,
    use_all_pricing_tenors: bool = USE_ALL_PRICING_TENORS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel, summary = compute_excess_returns(
        curve_df=curve_df,
        yields_cc=yields_cc,
        short_rate_monthly_cc=short_rate_monthly_cc,
        short_rate_df=short_rate_df,
        available_months=available_months,
        pricing_months=pricing_months,
        holding_period_months=holding_period_months,
        use_all_pricing_tenors=use_all_pricing_tenors,
    )
    save_excess_returns_results(panel=panel, summary=summary)
    return panel, summary


def main() -> None:
    panel, summary = run_excess_returns()
    print(f"saved {EXCESS_RETURNS_PATH}: {panel.shape}")
    print(f"saved {EXCESS_RETURNS_SUMMARY_PATH}: {summary.shape}")


if __name__ == "__main__":
    main()
