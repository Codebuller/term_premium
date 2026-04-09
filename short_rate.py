from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ruonia import load_ruonia_monthly


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
FACTORS_PATH = OUTPUT_DIR / "moex_curve_pca_scores.csv"
SHORT_RATE_PARAMS_PATH = OUTPUT_DIR / "moex_curve_short_rate_params.csv"
SHORT_RATE_FITTED_PATH = OUTPUT_DIR / "moex_curve_short_rate_fitted.csv"
SHORT_RATE_RESIDUALS_PATH = OUTPUT_DIR / "moex_curve_short_rate_residuals.csv"
SHORT_RATE_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_short_rate_summary.csv"
FACTOR_COLUMNS = ["PC1", "PC2", "PC3"]


def load_factors(path: str | Path = FACTORS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["month_end"])


def selected_factor_columns(df: pd.DataFrame) -> list[str]:
    available = set(df.columns)
    selected = [column for column in FACTOR_COLUMNS if column in available]
    if not selected:
        raise ValueError("No configured factor columns found in the factor file")
    return selected


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    residuals = y - fitted
    return coef, fitted, residuals


def fit_short_rate(
    factors_df: pd.DataFrame | None = None,
    short_rate_df: pd.DataFrame | None = None,
    short_rate_monthly_cc: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if factors_df is None:
        factors_df = load_factors()
    if short_rate_df is None:
        short_rate_df = load_ruonia_monthly()

    factor_columns = selected_factor_columns(factors_df)
    merged = short_rate_df[["month", "month_end", "ruonia_1m_pct", "short_rate_monthly_cc"]].merge(
        factors_df[["month", "month_end", *factor_columns]],
        on=["month", "month_end"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Curve and factor data have no overlapping observations")

    if short_rate_monthly_cc is None:
        short_rate_monthly_cc = merged["short_rate_monthly_cc"].to_numpy(dtype=float)

    rhs = np.column_stack([np.ones(len(merged)), merged[factor_columns].to_numpy(dtype=float)])
    coef, fitted, residuals = ols(short_rate_monthly_cc, rhs)

    params_rows = [{"parameter": "delta0", "value": float(coef[0])}]
    params_rows.extend(
        {"parameter": f"delta1_{column}", "value": float(value)}
        for column, value in zip(factor_columns, coef[1:])
    )
    params_rows.append({"parameter": "sigma2", "value": float(np.var(residuals, ddof=0))})
    params_df = pd.DataFrame(params_rows)

    fitted_df = merged[["month", "month_end"]].copy()
    fitted_df["ruonia_1m_pct"] = merged["ruonia_1m_pct"].to_numpy(dtype=float)
    fitted_df["short_rate_monthly_cc"] = short_rate_monthly_cc
    fitted_df["fitted_short_rate_monthly_cc"] = fitted

    residuals_df = merged[["month", "month_end"]].copy()
    residuals_df["ruonia_1m_pct"] = merged["ruonia_1m_pct"].to_numpy(dtype=float)
    residuals_df["resid_short_rate_monthly_cc"] = residuals

    summary_df = pd.DataFrame(
        [
            {
                "n_obs": len(merged),
                "sigma2": float(np.var(residuals, ddof=0)),
                "rmse_bp": float(np.sqrt(np.mean(residuals**2)) * 10000.0),
                "mean_short_rate_bp": float(short_rate_monthly_cc.mean() * 10000.0),
            }
        ]
    )
    return params_df, fitted_df, residuals_df, summary_df


def save_short_rate_results(
    params_df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    params_path: str | Path = SHORT_RATE_PARAMS_PATH,
    fitted_path: str | Path = SHORT_RATE_FITTED_PATH,
    residuals_path: str | Path = SHORT_RATE_RESIDUALS_PATH,
    summary_path: str | Path = SHORT_RATE_SUMMARY_PATH,
) -> None:
    params_path = Path(params_path)
    fitted_path = Path(fitted_path)
    residuals_path = Path(residuals_path)
    summary_path = Path(summary_path)

    params_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_path.parent.mkdir(parents=True, exist_ok=True)
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    params_df.to_csv(params_path, index=False)
    fitted_df.to_csv(fitted_path, index=False)
    residuals_df.to_csv(residuals_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def run_short_rate(
    factors_df: pd.DataFrame | None = None,
    short_rate_df: pd.DataFrame | None = None,
    short_rate_monthly_cc: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params_df, fitted_df, residuals_df, summary_df = fit_short_rate(
        factors_df=factors_df,
        short_rate_df=short_rate_df,
        short_rate_monthly_cc=short_rate_monthly_cc,
    )
    save_short_rate_results(
        params_df=params_df,
        fitted_df=fitted_df,
        residuals_df=residuals_df,
        summary_df=summary_df,
    )
    return params_df, fitted_df, residuals_df, summary_df


def main() -> None:
    params_df, fitted_df, residuals_df, summary_df = run_short_rate()
    print(f"saved {SHORT_RATE_PARAMS_PATH}: {params_df.shape}")
    print(f"saved {SHORT_RATE_FITTED_PATH}: {fitted_df.shape}")
    print(f"saved {SHORT_RATE_RESIDUALS_PATH}: {residuals_df.shape}")
    print(f"saved {SHORT_RATE_SUMMARY_PATH}: {summary_df.shape}")


if __name__ == "__main__":
    main()
