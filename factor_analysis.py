from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
CURVE_PATH = CLEAN_DIR / "output" / "moex_curve_monthly.csv"
OUTPUT_DIR = CLEAN_DIR / "output"
SCORES_PATH = OUTPUT_DIR / "moex_curve_pca_scores.csv"
LOADINGS_PATH = OUTPUT_DIR / "moex_curve_pca_loadings.csv"
SUMMARY_PATH = OUTPUT_DIR / "moex_curve_pca_summary.csv"
N_COMPONENTS = 3
DATE_COLUMNS = ["month_end"]
USE_ALL_TENORS = False
PCA_MONTHS = list(range(12, 181, 12))
USE_CONTINUOUS_YIELDS = True


def load_curve(path: str | Path = CURVE_PATH) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=DATE_COLUMNS)


def curve_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("M")]


def selected_curve_columns(df: pd.DataFrame) -> list[str]:
    all_columns = set(curve_columns(df))
    requested = [f"M{month:03d}" for month in PCA_MONTHS]
    selected = [column for column in requested if column in all_columns]

    if USE_ALL_TENORS:
        return sorted(all_columns)
    if not selected:
        raise ValueError("No configured tenor columns found in the curve file")
    return selected


def build_pca_input(curve_df: pd.DataFrame, y_columns: list[str]) -> np.ndarray:
    y = curve_df[y_columns].to_numpy(dtype=float) / 100.0
    if USE_CONTINUOUS_YIELDS:
        return np.log1p(y) * 100.0
    return y * 100.0


def run_pca(curve_df: pd.DataFrame, n_components: int = N_COMPONENTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_columns = selected_curve_columns(curve_df)
    y = build_pca_input(curve_df=curve_df, y_columns=y_columns)
    y_centered = y - y.mean(axis=0, keepdims=True)

    u, s, vh = np.linalg.svd(y_centered, full_matrices=False)
    n_components = min(n_components, len(s), len(y_columns))

    explained_variance = (s ** 2) / max(len(curve_df) - 1, 1)
    explained_ratio = explained_variance / explained_variance.sum()

    factor_names = [f"PC{i}" for i in range(1, n_components + 1)]
    scores = pd.DataFrame(u[:, :n_components] * s[:n_components], columns=factor_names)
    scores.insert(0, "month_end", curve_df["month_end"].to_numpy())
    scores.insert(0, "month", curve_df["month"].to_numpy())

    loadings = pd.DataFrame(vh[:n_components].T, index=y_columns, columns=factor_names).reset_index()
    loadings = loadings.rename(columns={"index": "tenor"})

    summary = pd.DataFrame(
        {
            "component": factor_names,
            "explained_variance": explained_variance[:n_components],
            "explained_variance_ratio": explained_ratio[:n_components],
            "cumulative_explained_variance_ratio": np.cumsum(explained_ratio[:n_components]),
            "n_tenors": len(y_columns),
            "yield_space": "continuous_pct" if USE_CONTINUOUS_YIELDS else "effective_pct",
        }
    )
    return scores, loadings, summary


def save_pca_results(
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    summary: pd.DataFrame,
    scores_path: str | Path = SCORES_PATH,
    loadings_path: str | Path = LOADINGS_PATH,
    summary_path: str | Path = SUMMARY_PATH,
) -> None:
    scores_path = Path(scores_path)
    loadings_path = Path(loadings_path)
    summary_path = Path(summary_path)

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    loadings_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    scores.to_csv(scores_path, index=False)
    loadings.to_csv(loadings_path, index=False)
    summary.to_csv(summary_path, index=False)


def run_factor_analysis(curve_df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if curve_df is None:
        curve_df = load_curve()
    scores, loadings, summary = run_pca(curve_df=curve_df)
    save_pca_results(scores=scores, loadings=loadings, summary=summary)
    return scores, loadings, summary


def main() -> None:
    scores, loadings, summary = run_factor_analysis()
    print(f"saved {SCORES_PATH}: {scores.shape}")
    print(f"saved {LOADINGS_PATH}: {loadings.shape}")
    print(f"saved {SUMMARY_PATH}: {summary.shape}")


if __name__ == "__main__":
    main()
