from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
FACTORS_PATH = OUTPUT_DIR / "moex_curve_pca_scores.csv"
VAR_PARAMS_PATH = OUTPUT_DIR / "moex_curve_var_params.csv"
VAR_RESIDUALS_PATH = OUTPUT_DIR / "moex_curve_var_residuals.csv"
VAR_FITTED_PATH = OUTPUT_DIR / "moex_curve_var_fitted.csv"
VAR_SUMMARY_PATH = OUTPUT_DIR / "moex_curve_var_summary.csv"
DATE_COLUMNS = ["month_end"]
ID_COLUMNS = ["month", "month_end"]
FACTOR_COLUMNS = ["PC1", "PC2", "PC3"]


def load_factors(path: str | Path = FACTORS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def selected_factor_columns(df: pd.DataFrame) -> list[str]:
    available = set(df.columns)
    selected = [column for column in FACTOR_COLUMNS if column in available]
    if not selected:
        raise ValueError("No configured factor columns found in the factor file")
    return selected


def ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ coef
    resid = y - fitted
    return coef, fitted, resid


def fit_var1(factors_df: pd.DataFrame) -> dict[str, object]:
    factor_columns = selected_factor_columns(factors_df)
    x = factors_df[factor_columns].to_numpy(dtype=float)

    rhs = np.column_stack([np.ones(len(x) - 1), x[:-1]])
    coef, fitted_next, residuals = ols(x[1:], rhs)

    intercept = coef[0]
    phi = coef[1:].T
    sigma = np.cov(residuals, rowvar=False, bias=False)
    eigvals = np.linalg.eigvals(phi)

    residual_norm = np.linalg.norm(residuals, axis=1)
    fitted_df = factors_df.iloc[1:][ID_COLUMNS].copy()
    residuals_df = factors_df.iloc[1:][ID_COLUMNS].copy()

    for idx, column in enumerate(factor_columns):
        fitted_df[column] = fitted_next[:, idx]
        residuals_df[f"eps_{column}"] = residuals[:, idx]

    params_rows: list[dict[str, object]] = []
    for idx, target in enumerate(factor_columns):
        params_rows.append(
            {
                "equation": target,
                "term": "const",
                "value": intercept[idx],
            }
        )
        for source_idx, source in enumerate(factor_columns):
            params_rows.append(
                {
                    "equation": target,
                    "term": f"lag1_{source}",
                    "value": phi[idx, source_idx],
                }
            )
    params_df = pd.DataFrame(params_rows)

    summary_df = pd.DataFrame(
        [
            {
                "n_obs": len(factors_df),
                "n_equations": len(factor_columns),
                "spectral_radius": float(np.max(np.abs(eigvals))),
                "residual_mean_norm": float(residual_norm.mean()),
                "residual_std_norm": float(residual_norm.std(ddof=0)),
                "sigma_trace": float(np.trace(sigma)),
            }
        ]
    )

    return {
        "params": params_df,
        "residuals": residuals_df,
        "fitted": fitted_df,
        "summary": summary_df,
    }


def save_var_results(
    params_df: pd.DataFrame,
    residuals_df: pd.DataFrame,
    fitted_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    params_path: str | Path = VAR_PARAMS_PATH,
    residuals_path: str | Path = VAR_RESIDUALS_PATH,
    fitted_path: str | Path = VAR_FITTED_PATH,
    summary_path: str | Path = VAR_SUMMARY_PATH,
) -> None:
    params_path = Path(params_path)
    residuals_path = Path(residuals_path)
    fitted_path = Path(fitted_path)
    summary_path = Path(summary_path)

    params_path.parent.mkdir(parents=True, exist_ok=True)
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    fitted_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    params_df.to_csv(params_path, index=False)
    residuals_df.to_csv(residuals_path, index=False)
    fitted_df.to_csv(fitted_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def run_var(factors_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if factors_df is None:
        factors_df = load_factors()

    result = fit_var1(factors_df=factors_df)
    params_df = result["params"]  # type: ignore[assignment]
    residuals_df = result["residuals"]  # type: ignore[assignment]
    fitted_df = result["fitted"]  # type: ignore[assignment]
    summary_df = result["summary"]  # type: ignore[assignment]

    save_var_results(
        params_df=params_df,
        residuals_df=residuals_df,
        fitted_df=fitted_df,
        summary_df=summary_df,
    )
    return params_df, residuals_df, fitted_df, summary_df


def main() -> None:
    params_df, residuals_df, fitted_df, summary_df = run_var()
    print(f"saved {VAR_PARAMS_PATH}: {params_df.shape}")
    print(f"saved {VAR_RESIDUALS_PATH}: {residuals_df.shape}")
    print(f"saved {VAR_FITTED_PATH}: {fitted_df.shape}")
    print(f"saved {VAR_SUMMARY_PATH}: {summary_df.shape}")


if __name__ == "__main__":
    main()
