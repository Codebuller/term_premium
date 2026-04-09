from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
CURVE_PATH = OUTPUT_DIR / "moex_curve_monthly.csv"
FACTORS_PATH = OUTPUT_DIR / "moex_curve_pca_scores.csv"
VAR_PARAMS_PATH = OUTPUT_DIR / "moex_curve_var_params.csv"
VAR_RESIDUALS_PATH = OUTPUT_DIR / "moex_curve_var_residuals.csv"
SHORT_RATE_PARAMS_PATH = OUTPUT_DIR / "moex_curve_short_rate_params.csv"
LAMBDA_PARAMS_PATH = OUTPUT_DIR / "moex_curve_lambda_params.csv"
RISK_NEUTRAL_PATH = OUTPUT_DIR / "moex_curve_risk_neutral.csv"
RISK_NEUTRAL_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_risk_neutral_summary.csv"
DATE_COLUMNS = ["month_end"]
FACTOR_COLUMNS = ["PC1", "PC2", "PC3"]
SUMMARY_MONTHS = [12, 24, 36, 60, 120, 180]


def load_curve(path: str | Path = CURVE_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def load_factors(path: str | Path = FACTORS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def load_var_params(path: str | Path = VAR_PARAMS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_var_residuals(path: str | Path = VAR_RESIDUALS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def load_short_rate_params(path: str | Path = SHORT_RATE_PARAMS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_lambda_params(path: str | Path = LAMBDA_PARAMS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def curve_months(df: pd.DataFrame) -> list[int]:
    return sorted(int(column[1:]) for column in df.columns if column.startswith("M"))


def selected_factor_columns(df: pd.DataFrame) -> list[str]:
    available = set(df.columns)
    selected = [column for column in FACTOR_COLUMNS if column in available]
    if not selected:
        raise ValueError("No configured factor columns found")
    return selected


def parse_var_parameters(var_params_df: pd.DataFrame, factor_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array(
        [
            float(
                var_params_df.loc[
                    (var_params_df["equation"] == column) & (var_params_df["term"] == "const"),
                    "value",
                ].iloc[0]
            )
            for column in factor_columns
        ]
    )
    phi = np.array(
        [
            [
                float(
                    var_params_df.loc[
                        (var_params_df["equation"] == row_column)
                        & (var_params_df["term"] == f"lag1_{col_column}"),
                        "value",
                    ].iloc[0]
                )
                for col_column in factor_columns
            ]
            for row_column in factor_columns
        ]
    )
    return mu, phi


def parse_short_rate_parameters(short_rate_params_df: pd.DataFrame, factor_columns: list[str]) -> tuple[float, np.ndarray, float]:
    delta0 = float(short_rate_params_df.loc[short_rate_params_df["parameter"] == "delta0", "value"].iloc[0])
    delta1 = np.array(
        [
            float(short_rate_params_df.loc[short_rate_params_df["parameter"] == f"delta1_{column}", "value"].iloc[0])
            for column in factor_columns
        ]
    )
    sigma2 = float(short_rate_params_df.loc[short_rate_params_df["parameter"] == "sigma2", "value"].iloc[0])
    return delta0, delta1, sigma2


def parse_lambda_parameters(lambda_params_df: pd.DataFrame, factor_columns: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
    lambda0 = np.array(
        [
            float(lambda_params_df.loc[lambda_params_df["parameter"] == f"lambda0_{column}", "value"].iloc[0])
            for column in factor_columns
        ]
    )
    lambda1 = np.array(
        [
            [
                float(
                    lambda_params_df.loc[
                        lambda_params_df["parameter"] == f"lambda1_{row_column}_{col_column}",
                        "value",
                    ].iloc[0]
                )
                for col_column in factor_columns
            ]
            for row_column in factor_columns
        ]
    )
    omega_row = lambda_params_df.loc[lambda_params_df["parameter"] == "omega_scalar", "value"]
    if omega_row.empty:
        raise ValueError("omega_scalar not found in lambda parameters")
    omega_scalar = float(omega_row.iloc[0])
    return lambda0, lambda1, omega_scalar


def affine_recursion(
    mu: np.ndarray,
    phi: np.ndarray,
    sigma: np.ndarray,
    delta0: float,
    delta1: np.ndarray,
    omega_scalar: float,
    max_months: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_factors = len(delta1)
    a = np.zeros(max_months + 1)
    b = np.zeros((max_months + 1, n_factors))
    for month in range(1, max_months + 1):
        prev_b = b[month - 1]
        a[month] = a[month - 1] + prev_b @ mu + 0.5 * (prev_b @ sigma @ prev_b + omega_scalar) - delta0
        b[month] = prev_b @ phi - delta1
    return a, b


def build_risk_neutral_curve(
    curve_df: pd.DataFrame | None = None,
    factors_df: pd.DataFrame | None = None,
    var_params_df: pd.DataFrame | None = None,
    var_residuals_df: pd.DataFrame | None = None,
    short_rate_params_df: pd.DataFrame | None = None,
    lambda_params_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if curve_df is None:
        curve_df = load_curve()
    if factors_df is None:
        factors_df = load_factors()
    if var_params_df is None:
        var_params_df = load_var_params()
    if var_residuals_df is None:
        var_residuals_df = load_var_residuals()
    if short_rate_params_df is None:
        short_rate_params_df = load_short_rate_params()
    if lambda_params_df is None:
        lambda_params_df = load_lambda_params()

    factor_columns = selected_factor_columns(factors_df)
    months = curve_months(curve_df)
    max_months = max(months)
    y_columns = [f"M{month:03d}" for month in months]
    merged = curve_df[["month", "month_end", *y_columns]].merge(
        factors_df[["month", "month_end", *factor_columns]],
        on=["month", "month_end"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("Curve and factors have no overlapping observations")

    yields_cc_pct = np.log1p(merged[y_columns].to_numpy(dtype=float) / 100.0) * 100.0
    state = merged[factor_columns].to_numpy(dtype=float)
    mu, phi = parse_var_parameters(var_params_df=var_params_df, factor_columns=factor_columns)
    sigma = np.cov(
        var_residuals_df[[f"eps_{column}" for column in factor_columns]].to_numpy(dtype=float),
        rowvar=False,
        bias=False,
    )
    delta0, delta1, _ = parse_short_rate_parameters(
        short_rate_params_df=short_rate_params_df,
        factor_columns=factor_columns,
    )
    lambda0, lambda1, omega_scalar = parse_lambda_parameters(
        lambda_params_df=lambda_params_df,
        factor_columns=factor_columns,
    )

    a_p, b_p = affine_recursion(
        mu - lambda0,
        phi - lambda1,
        sigma,
        delta0,
        delta1,
        omega_scalar,
        max_months=max_months,
    )
    a_q, b_q = affine_recursion(
        mu,
        phi,
        sigma,
        delta0,
        delta1,
        omega_scalar,
        max_months=max_months,
    )

    fit_arrays: dict[str, np.ndarray] = {}
    rn_arrays: dict[str, np.ndarray] = {}
    for month in months:
        tau_years = month / 12.0
        fit_arrays[f"fit_M{month:03d}"] = (-(a_p[month] + state @ b_p[month]) / tau_years) * 100.0
        rn_arrays[f"rn_M{month:03d}"] = (-(a_q[month] + state @ b_q[month]) / tau_years) * 100.0

    output_arrays: dict[str, np.ndarray] = {
        "month": merged["month"].to_numpy(),
        "month_end": merged["month_end"].to_numpy(),
    }
    output_arrays.update(fit_arrays)
    output_arrays.update(rn_arrays)
    output = pd.DataFrame(output_arrays)

    summary_months = [month for month in SUMMARY_MONTHS if month in months]
    summary_idx = [months.index(month) for month in summary_months]
    fit_matrix = np.column_stack([fit_arrays[f"fit_M{month:03d}"] for month in months])
    summary_df = pd.DataFrame(
        [
            {
                "n_obs": len(output),
                "n_tenors": len(months),
                "fit_rmse_full_grid_bp": float(np.sqrt(np.mean((fit_matrix - yields_cc_pct) ** 2)) * 100.0),
                "fit_rmse_selected_bp": float(
                    np.sqrt(np.mean((fit_matrix[:, summary_idx] - yields_cc_pct[:, summary_idx]) ** 2)) * 100.0
                ),
                "physical_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(phi)))),
                "risk_neutral_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(phi - lambda1)))),
            }
        ]
    )
    return output, summary_df


def save_risk_neutral_results(
    output_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: str | Path = RISK_NEUTRAL_PATH,
    summary_path: str | Path = RISK_NEUTRAL_SUMMARY_PATH,
) -> None:
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def run_risk_neutral_curve(
    curve_df: pd.DataFrame | None = None,
    factors_df: pd.DataFrame | None = None,
    var_params_df: pd.DataFrame | None = None,
    var_residuals_df: pd.DataFrame | None = None,
    short_rate_params_df: pd.DataFrame | None = None,
    lambda_params_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_df, summary_df = build_risk_neutral_curve(
        curve_df=curve_df,
        factors_df=factors_df,
        var_params_df=var_params_df,
        var_residuals_df=var_residuals_df,
        short_rate_params_df=short_rate_params_df,
        lambda_params_df=lambda_params_df,
    )
    save_risk_neutral_results(output_df=output_df, summary_df=summary_df)
    return output_df, summary_df


def main() -> None:
    output_df, summary_df = run_risk_neutral_curve()
    print(f"saved {RISK_NEUTRAL_PATH}: {output_df.shape}")
    print(f"saved {RISK_NEUTRAL_SUMMARY_PATH}: {summary_df.shape}")


if __name__ == "__main__":
    main()
