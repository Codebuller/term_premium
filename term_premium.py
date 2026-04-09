from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
CURVE_PATH = OUTPUT_DIR / "moex_curve_monthly.csv"
RISK_NEUTRAL_PATH = OUTPUT_DIR / "moex_curve_risk_neutral.csv"
TERM_PREMIUM_PATH = OUTPUT_DIR / "moex_curve_term_premium.csv"
TERM_PREMIUM_SELECTED_PATH = OUTPUT_DIR / "moex_curve_term_premium_selected.csv"
TERM_PREMIUM_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_term_premium_summary.csv"
DATE_COLUMNS = ["month_end"]
SELECTED_MONTHS = [12, 24, 36, 60, 120, 180]


def load_curve(path: str | Path = CURVE_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def load_risk_neutral_curve(path: str | Path = RISK_NEUTRAL_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def curve_months(df: pd.DataFrame) -> list[int]:
    return sorted(int(column[1:]) for column in df.columns if column.startswith("M"))


def build_term_premium(
    curve_df: pd.DataFrame | None = None,
    risk_neutral_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if curve_df is None:
        curve_df = load_curve()
    if risk_neutral_df is None:
        risk_neutral_df = load_risk_neutral_curve()

    months = curve_months(curve_df)
    merged = curve_df[["month", "month_end", *[f"M{month:03d}" for month in months]]].merge(
        risk_neutral_df,
        on=["month", "month_end"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Curve and risk-neutral outputs have no overlapping observations")

    output_arrays: dict[str, np.ndarray] = {
        "month": merged["month"].to_numpy(),
        "month_end": merged["month_end"].to_numpy(),
    }
    latest_row: dict[str, object] = {"latest_month": str(merged["month"].iloc[-1])}

    for month in months:
        observed_cc_pct = np.log1p(merged[f"M{month:03d}"].to_numpy(dtype=float) / 100.0) * 100.0
        fit_cc_pct = merged[f"fit_M{month:03d}"].to_numpy(dtype=float)
        rn_cc_pct = merged[f"rn_M{month:03d}"].to_numpy(dtype=float)
        tp_cc_pct = fit_cc_pct - rn_cc_pct

        output_arrays[f"obs_M{month:03d}"] = observed_cc_pct
        output_arrays[f"fit_M{month:03d}"] = fit_cc_pct
        output_arrays[f"rn_M{month:03d}"] = rn_cc_pct
        output_arrays[f"tp_M{month:03d}"] = tp_cc_pct

        if month in SELECTED_MONTHS:
            latest_row[f"tp_M{month:03d}_last_pct"] = float(tp_cc_pct[-1])

    output = pd.DataFrame(output_arrays)
    selected_months = [month for month in SELECTED_MONTHS if month in months]
    selected_arrays: dict[str, np.ndarray] = {
        "month": output["month"].to_numpy(),
        "month_end": output["month_end"].to_numpy(),
    }
    for month in selected_months:
        selected_arrays[f"obs_M{month:03d}"] = output[f"obs_M{month:03d}"].to_numpy()
        selected_arrays[f"rn_M{month:03d}"] = output[f"rn_M{month:03d}"].to_numpy()
        selected_arrays[f"tp_M{month:03d}"] = output[f"tp_M{month:03d}"].to_numpy()

    selected = pd.DataFrame(selected_arrays)
    summary = pd.DataFrame([latest_row])
    return output, selected, summary


def save_term_premium_results(
    output_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str | Path = TERM_PREMIUM_PATH,
    selected_path: str | Path = TERM_PREMIUM_SELECTED_PATH,
    summary_path: str | Path = TERM_PREMIUM_SUMMARY_PATH,
) -> None:
    output_path = Path(output_path)
    selected_path = Path(selected_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    selected_df.to_csv(selected_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def run_term_premium(
    curve_df: pd.DataFrame | None = None,
    risk_neutral_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_df, selected_df, summary_df = build_term_premium(
        curve_df=curve_df,
        risk_neutral_df=risk_neutral_df,
    )
    save_term_premium_results(
        output_df=output_df,
        selected_df=selected_df,
        summary_df=summary_df,
    )
    return output_df, selected_df, summary_df


def main() -> None:
    output_df, selected_df, summary_df = run_term_premium()
    print(f"saved {TERM_PREMIUM_PATH}: {output_df.shape}")
    print(f"saved {TERM_PREMIUM_SELECTED_PATH}: {selected_df.shape}")
    print(f"saved {TERM_PREMIUM_SUMMARY_PATH}: {summary_df.shape}")


if __name__ == "__main__":
    main()
