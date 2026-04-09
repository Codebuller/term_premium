from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
FACTORS_PATH = OUTPUT_DIR / "moex_curve_pca_scores.csv"
VAR_RESIDUALS_PATH = OUTPUT_DIR / "moex_curve_var_residuals.csv"
EXCESS_RETURNS_PATH = OUTPUT_DIR / "moex_curve_excess_returns.csv"
SHORT_RATE_PARAMS_PATH = OUTPUT_DIR / "moex_curve_short_rate_params.csv"
LAMBDA_PARAMS_PATH = OUTPUT_DIR / "moex_curve_lambda_params.csv"
RX_REGRESSION_PARAMS_PATH = OUTPUT_DIR / "moex_curve_rx_regression_params.csv"
RX_FITTED_PATH = OUTPUT_DIR / "moex_curve_rx_fitted.csv"
RX_RESIDUALS_PATH = OUTPUT_DIR / "moex_curve_rx_residuals.csv"
LAMBDA_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_lambda_summary.csv"
DATE_COLUMNS = ["month_end", "start_month_end", "end_month_end"]
FACTOR_COLUMNS = ["PC1", "PC2", "PC3"]


def load_factors(path: str | Path = FACTORS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["month_end"])


def load_var_residuals(path: str | Path = VAR_RESIDUALS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["month_end"])


def load_excess_returns(path: str | Path = EXCESS_RETURNS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["start_month_end", "end_month_end"])


def load_short_rate_params(path: str | Path = SHORT_RATE_PARAMS_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def selected_factor_columns(df: pd.DataFrame) -> list[str]:
    available = set(df.columns)
    selected = [column for column in FACTOR_COLUMNS if column in available]
    if not selected:
        raise ValueError("No configured factor columns found")
    return selected


def innovation_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("eps_")]
    if not columns:
        raise ValueError("No innovation columns found in VAR residuals")
    return columns


def rx_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith("rx_M")]
    if not columns:
        raise ValueError("No excess return columns found")
    return columns


def extract_sigma2(short_rate_params_df: pd.DataFrame) -> float:
    row = short_rate_params_df.loc[short_rate_params_df["parameter"] == "sigma2", "value"]
    if row.empty:
        raise ValueError("sigma2 not found in short-rate params")
    return float(row.iloc[0])


def align_stage2_inputs(
    factors_df: pd.DataFrame,
    var_residuals_df: pd.DataFrame,
    excess_returns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    factor_cols = selected_factor_columns(factors_df)
    innovation_cols = innovation_columns(var_residuals_df)
    excess_return_cols = rx_columns(excess_returns_df)

    state_t_df = factors_df[["month", "month_end", *factor_cols]].rename(
        columns={
            "month": "start_month",
            "month_end": "start_month_end",
        }
    )
    innovations_df = var_residuals_df[["month", "month_end", *innovation_cols]].rename(
        columns={
            "month": "end_month",
            "month_end": "end_month_end",
        }
    )

    aligned = excess_returns_df.merge(
        state_t_df,
        on=["start_month", "start_month_end"],
        how="inner",
    ).merge(
        innovations_df,
        on=["end_month", "end_month_end"],
        how="inner",
    )
    if aligned.empty:
        raise ValueError("No overlapping observations across factors, VAR residuals, and excess returns")
    return aligned, factor_cols, innovation_cols, excess_return_cols


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    residuals = y - fitted
    return coef, fitted, residuals


def fit_market_price_of_risk(
    factors_df: pd.DataFrame | None = None,
    var_residuals_df: pd.DataFrame | None = None,
    excess_returns_df: pd.DataFrame | None = None,
    sigma2: float | None = None,
    state_t: np.ndarray | None = None,
    innovations: np.ndarray | None = None,
    excess_returns: np.ndarray | None = None,
    factor_columns: list[str] | None = None,
    rx_tenors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aligned: pd.DataFrame | None = None
    if state_t is None or innovations is None or excess_returns is None:
        if factors_df is None:
            factors_df = load_factors()
        if var_residuals_df is None:
            var_residuals_df = load_var_residuals()
        if excess_returns_df is None:
            excess_returns_df = load_excess_returns()
        aligned, factor_columns, innovation_cols, rx_columns_selected = align_stage2_inputs(
            factors_df=factors_df,
            var_residuals_df=var_residuals_df,
            excess_returns_df=excess_returns_df,
        )
        state_t = aligned[factor_columns].to_numpy(dtype=float)
        innovations = aligned[innovation_cols].to_numpy(dtype=float)
        excess_returns = aligned[rx_columns_selected].to_numpy(dtype=float)
        rx_tenors = [column.replace("rx_", "") for column in rx_columns_selected]
    else:
        if factor_columns is None:
            factor_columns = FACTOR_COLUMNS
        if rx_tenors is None:
            rx_tenors = [f"M{i + 1:03d}" for i in range(excess_returns.shape[1])]

    if sigma2 is None:
        sigma2 = extract_sigma2(load_short_rate_params())

    sigma = np.cov(innovations, rowvar=False, bias=False)
    rhs = np.column_stack([np.ones(len(excess_returns)), state_t, innovations])
    coef, fitted, residuals = ols(excess_returns, rhs)

    n_factors = state_t.shape[1]
    a = coef[0]
    c = coef[1 : 1 + n_factors].T
    beta = coef[1 + n_factors :].T
    beta_star = np.vstack([np.kron(beta_row, beta_row) for beta_row in beta])

    omega_scalar = float(np.var(residuals.reshape(-1, 1), ddof=0))
    omega_diag = np.full((beta.shape[0], 1), omega_scalar)
    s0 = sigma.reshape((-1, 1))

    factors = np.column_stack([np.ones(len(state_t)), state_t])
    innovations_projection = innovations @ np.linalg.pinv(innovations.T @ innovations) @ innovations.T
    factors_ortho = factors - innovations_projection @ factors

    adjustment = beta_star @ s0 + omega_diag
    rx_adjusted = excess_returns + 0.5 * adjustment.reshape(1, -1)
    y = (np.linalg.pinv(factors_ortho.T @ factors_ortho) @ factors_ortho.T @ rx_adjusted).T
    lambda_matrix = np.linalg.pinv(beta.T @ beta) @ beta.T @ y
    lambda0 = lambda_matrix[:, 0]
    lambda1 = lambda_matrix[:, 1:]

    lambda_rows = [
        *(
            {"parameter": f"lambda0_{column}", "value": float(value)}
            for column, value in zip(factor_columns, lambda0)
        ),
        {"parameter": "omega_scalar", "value": omega_scalar},
        *(
            {
                "parameter": f"lambda1_{row_column}_{col_column}",
                "value": float(lambda1[row_idx, col_idx]),
            }
            for row_idx, row_column in enumerate(factor_columns)
            for col_idx, col_column in enumerate(factor_columns)
        ),
    ]
    lambda_params_df = pd.DataFrame(lambda_rows)

    regression_rows: list[dict[str, object]] = []
    rhs_terms = ["const", *[f"state_{column}" for column in factor_columns], *[f"eps_{column}" for column in factor_columns]]
    for eq_idx, tenor in enumerate(rx_tenors):
        for term_idx, term in enumerate(rhs_terms):
            regression_rows.append(
                {
                    "equation": f"rx_{tenor}",
                    "term": term,
                    "value": float(coef[term_idx, eq_idx]),
                }
            )
    regression_params_df = pd.DataFrame(regression_rows)

    if aligned is None:
        fitted_df = pd.DataFrame(fitted, columns=[f"rxhat_{tenor}" for tenor in rx_tenors])
        residuals_df = pd.DataFrame(residuals, columns=[f"rxerr_{tenor}" for tenor in rx_tenors])
    else:
        fitted_df = pd.concat(
            [
                aligned[["end_month", "end_month_end"]].reset_index(drop=True),
                pd.DataFrame(fitted, columns=[f"rxhat_{tenor}" for tenor in rx_tenors]),
            ],
            axis=1,
        )
        residuals_df = pd.concat(
            [
                aligned[["end_month", "end_month_end"]].reset_index(drop=True),
                pd.DataFrame(residuals, columns=[f"rxerr_{tenor}" for tenor in rx_tenors]),
            ],
            axis=1,
        )

    summary_df = pd.DataFrame(
        [
            {
                "n_obs": len(excess_returns),
                "n_equations": excess_returns.shape[1],
                "sigma2": float(sigma2),
                "omega_scalar": omega_scalar,
                "rmse_bp": float(np.sqrt(np.mean(residuals**2)) * 10000.0),
                "lambda0_norm": float(np.linalg.norm(lambda0)),
                "lambda1_norm": float(np.linalg.norm(lambda1)),
            }
        ]
    )
    return lambda_params_df, regression_params_df, fitted_df, residuals_df, summary_df


def save_market_price_of_risk_results(
    lambda_params_df: pd.DataFrame,
    regression_params_df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    lambda_params_path: str | Path = LAMBDA_PARAMS_PATH,
    regression_params_path: str | Path = RX_REGRESSION_PARAMS_PATH,
    fitted_path: str | Path = RX_FITTED_PATH,
    residuals_path: str | Path = RX_RESIDUALS_PATH,
    summary_path: str | Path = LAMBDA_SUMMARY_PATH,
) -> None:
    lambda_params_path = Path(lambda_params_path)
    regression_params_path = Path(regression_params_path)
    fitted_path = Path(fitted_path)
    residuals_path = Path(residuals_path)
    summary_path = Path(summary_path)

    lambda_params_path.parent.mkdir(parents=True, exist_ok=True)
    regression_params_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_path.parent.mkdir(parents=True, exist_ok=True)
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    lambda_params_df.to_csv(lambda_params_path, index=False)
    regression_params_df.to_csv(regression_params_path, index=False)
    fitted_df.to_csv(fitted_path, index=False)
    residuals_df.to_csv(residuals_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def run_market_price_of_risk(
    factors_df: pd.DataFrame | None = None,
    var_residuals_df: pd.DataFrame | None = None,
    excess_returns_df: pd.DataFrame | None = None,
    sigma2: float | None = None,
    state_t: np.ndarray | None = None,
    innovations: np.ndarray | None = None,
    excess_returns: np.ndarray | None = None,
    factor_columns: list[str] | None = None,
    rx_tenors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs = fit_market_price_of_risk(
        factors_df=factors_df,
        var_residuals_df=var_residuals_df,
        excess_returns_df=excess_returns_df,
        sigma2=sigma2,
        state_t=state_t,
        innovations=innovations,
        excess_returns=excess_returns,
        factor_columns=factor_columns,
        rx_tenors=rx_tenors,
    )
    save_market_price_of_risk_results(*outputs)
    return outputs


def main() -> None:
    lambda_params_df, regression_params_df, fitted_df, residuals_df, summary_df = run_market_price_of_risk()
    print(f"saved {LAMBDA_PARAMS_PATH}: {lambda_params_df.shape}")
    print(f"saved {RX_REGRESSION_PARAMS_PATH}: {regression_params_df.shape}")
    print(f"saved {RX_FITTED_PATH}: {fitted_df.shape}")
    print(f"saved {RX_RESIDUALS_PATH}: {residuals_df.shape}")
    print(f"saved {LAMBDA_SUMMARY_PATH}: {summary_df.shape}")


if __name__ == "__main__":
    main()
