from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL


MONTH_MAP = {
    "январь": 1,
    "февраль": 2,
    "март": 3,
    "апрель": 4,
    "май": 5,
    "июнь": 6,
    "июль": 7,
    "август": 8,
    "сентябрь": 9,
    "октябрь": 10,
    "ноябрь": 11,
    "декабрь": 12,
}


def load_cpi_expectations(
    path: str | Path = "data/raw/cbr_infl_exp_26_03.xlsx",
    sheet_name: str = "Данные за все годы",
    row_index: int = 70,
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    dates = pd.to_datetime(raw.iloc[1, 1:], errors="coerce")
    values = pd.to_numeric(raw.iloc[row_index, 1:], errors="coerce")
    frame = pd.DataFrame({"date": dates, "cpi_exp_pct": values}).dropna()
    frame["month_end"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    frame = frame.groupby("month_end", as_index=False)["cpi_exp_pct"].last()
    return frame


def load_rosstat_cpi_monthly(
    path: str | Path = "data/raw/rosstat_ipc_monthly_1991_2026.xlsx",
    sheet_name: str = "01",
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    years = [int(year) for year in raw.iloc[3, 1:] if pd.notna(year)]
    month_names = raw.iloc[5:17, 0].tolist()

    records: list[dict[str, float | pd.Timestamp]] = []
    for row_idx, month_name in enumerate(month_names, start=5):
        month = MONTH_MAP[str(month_name).strip().lower()]
        for col_idx, year in enumerate(years, start=1):
            value = pd.to_numeric(raw.iat[row_idx, col_idx], errors="coerce")
            if pd.notna(value):
                records.append(
                    {
                        "month_end": pd.Timestamp(year=year, month=month, day=1).to_period("M").to_timestamp("M"),
                        "cpi_mm_pct_raw": float(value) - 100.0,
                    }
                )

    frame = pd.DataFrame(records).sort_values("month_end").reset_index(drop=True)
    return frame


def add_stl_seasonal_adjustment(
    cpi_monthly_df: pd.DataFrame,
    period: int = 12,
) -> pd.DataFrame:
    frame = cpi_monthly_df.sort_values("month_end").reset_index(drop=True).copy()
    stl = STL(frame["cpi_mm_pct_raw"], period=period, robust=True).fit()
    frame["cpi_mmsa_pct"] = frame["cpi_mm_pct_raw"] - stl.seasonal
    return frame


def load_usd_monthly_volatility(
    json_paths: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    if json_paths is None:
        json_paths = sorted(Path("data/raw").glob("moex_usd_tom_*.json"))

    parts: list[pd.DataFrame] = []
    for path in json_paths:
        payload = json.loads(Path(path).read_text())
        data = payload.get("candles", {}).get("data", [])
        columns = payload.get("candles", {}).get("columns", [])
        if not data or not columns:
            continue
        parts.append(pd.DataFrame(data, columns=columns))

    if not parts:
        raise ValueError("No MOEX USD candle data found")

    frame = pd.concat(parts, ignore_index=True)
    frame = frame.drop_duplicates(subset=["begin"]).sort_values("begin").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["begin"])
    frame["month_end"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    return frame.groupby("month_end", as_index=False)["close"].std().rename(columns={"close": "usd_vol"})


def build_cbr_observed_factors(
    cpi_path: str | Path = "data/raw/rosstat_ipc_monthly_1991_2026.xlsx",
    cpi_exp_path: str | Path = "data/raw/cbr_infl_exp_26_03.xlsx",
    usd_json_paths: Iterable[str | Path] | None = None,
) -> pd.DataFrame:
    cpi = add_stl_seasonal_adjustment(load_rosstat_cpi_monthly(cpi_path))
    cpi_exp = load_cpi_expectations(cpi_exp_path)
    usd_vol = load_usd_monthly_volatility(usd_json_paths)

    factors = (
        cpi.merge(cpi_exp, on="month_end", how="outer")
        .merge(usd_vol, on="month_end", how="outer")
        .sort_values("month_end")
        .reset_index(drop=True)
    )
    factors["cpi_mmsa_lag1"] = factors["cpi_mmsa_pct"].shift(1)
    factors["usd_vol_lag1"] = factors["usd_vol"].shift(1)
    return factors


def run_tp_determinant_regressions(
    term_premium_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    tenors: Sequence[int] = (24, 60, 120),
    cov_type: str = "HC1",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = term_premium_df.copy()
    if "date" not in frame.columns:
        raise ValueError("term_premium_df must contain a 'date' column")

    frame["month_end"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    merged = frame.merge(
        factors_df[["month_end", "cpi_mmsa_lag1", "cpi_exp_pct", "usd_vol_lag1"]],
        on="month_end",
        how="inner",
    ).dropna()

    rows: list[dict[str, float | int | str]] = []
    for tenor in tenors:
        y = merged[f"tp_M{tenor:03d}"].astype(float)
        x = sm.add_constant(merged[["cpi_mmsa_lag1", "cpi_exp_pct", "usd_vol_lag1"]].astype(float))
        result = sm.OLS(y, x).fit(cov_type=cov_type)

        for parameter in result.params.index:
            rows.append(
                {
                    "tenor_months": tenor,
                    "parameter": parameter,
                    "coef": float(result.params[parameter]),
                    "std_err": float(result.bse[parameter]),
                    "t": float(result.tvalues[parameter]),
                    "p": float(result.pvalues[parameter]),
                    "r2": float(result.rsquared),
                    "n_obs": int(result.nobs),
                }
            )

    return pd.DataFrame(rows), merged


def prepare_determinant_sample(
    term_premium_df: pd.DataFrame,
    factors_df: pd.DataFrame,
) -> pd.DataFrame:
    frame = term_premium_df.copy()
    if "date" not in frame.columns:
        raise ValueError("term_premium_df must contain a 'date' column")

    frame["month_end"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp("M")
    return frame.merge(
        factors_df[["month_end", "cpi_exp_pct", "cpi_mmsa_lag1", "usd_vol_lag1"]],
        on="month_end",
        how="inner",
    ).dropna()


def plot_tp_determinant_regressions(
    determinant_results: pd.DataFrame,
    model_samples: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
):
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    factor_labels = {
        "cpi_mmsa_lag1": "CPI_mmSA(t-1)",
        "cpi_exp_pct": "CPI_exp",
        "usd_vol_lag1": "USD_vol(t-1)",
    }
    factor_colors = {
        "cpi_mmsa_lag1": "#0b6e4f",
        "cpi_exp_pct": "#d95d39",
        "usd_vol_lag1": "#4169e1",
    }
    model_titles = {
        "library": "Library baseline",
        "hybrid_ruonia": "Hybrid RUONIA",
    }
    tenor_labels = {24: "2Y", 60: "5Y", 120: "10Y"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#f5f1e8")
    for row_idx, model_name in enumerate(("library", "hybrid_ruonia")):
        coef_ax = axes[row_idx, 0]
        ts_ax = axes[row_idx, 1]
        sample_df = model_samples[model_name]
        title = model_titles.get(model_name, model_name)

        subset = determinant_results[
            (determinant_results["model"] == model_name)
            & (determinant_results["parameter"] != "const")
        ].copy()

        for parameter, label in factor_labels.items():
            tmp = subset[subset["parameter"] == parameter].sort_values("tenor_months")
            coef_ax.plot(
                tmp["tenor_months"],
                tmp["coef"],
                marker="o",
                lw=2,
                color=factor_colors[parameter],
                label=label,
            )
            for _, obs in tmp.iterrows():
                if obs["p"] < 0.10:
                    coef_ax.text(
                        obs["tenor_months"],
                        obs["coef"],
                        "*",
                        fontsize=14,
                        ha="center",
                        va="bottom",
                    )

        coef_ax.axhline(0.0, color="#7a8890", lw=0.9)
        coef_ax.set_xticks([24, 60, 120])
        coef_ax.set_xticklabels([tenor_labels[x] for x in [24, 60, 120]])
        coef_ax.set_title(f"{title}: coefficients by tenor", loc="left")
        coef_ax.set_ylabel("Coefficient")
        coef_ax.grid(True, axis="y", alpha=0.3)
        coef_ax.spines["top"].set_visible(False)
        coef_ax.spines["right"].set_visible(False)

        std_frame = sample_df[["month_end", "tp_M120", "cpi_exp_pct", "cpi_mmsa_lag1", "usd_vol_lag1"]].copy()
        for column in ["tp_M120", "cpi_exp_pct", "cpi_mmsa_lag1", "usd_vol_lag1"]:
            std = std_frame[column].std(ddof=0)
            if std == 0 or pd.isna(std):
                std_frame[column] = 0.0
            else:
                std_frame[column] = (std_frame[column] - std_frame[column].mean()) / std

        ts_ax.plot(
            std_frame["month_end"],
            std_frame["tp_M120"],
            color="#10212b",
            lw=2.2,
            label="TP 10Y (z-score)",
        )
        ts_ax.plot(
            std_frame["month_end"],
            std_frame["cpi_exp_pct"],
            color=factor_colors["cpi_exp_pct"],
            lw=1.8,
            label="CPI_exp",
        )
        ts_ax.plot(
            std_frame["month_end"],
            std_frame["cpi_mmsa_lag1"],
            color=factor_colors["cpi_mmsa_lag1"],
            lw=1.8,
            label="CPI_mmSA(t-1)",
        )
        ts_ax.plot(
            std_frame["month_end"],
            std_frame["usd_vol_lag1"],
            color=factor_colors["usd_vol_lag1"],
            lw=1.5,
            label="USD_vol(t-1)",
        )
        ts_ax.axhline(0.0, color="#7a8890", lw=0.9)
        ts_ax.set_title(f"{title}: standardized TP10Y and factors", loc="left")
        ts_ax.grid(True, axis="y", alpha=0.3)
        ts_ax.spines["top"].set_visible(False)
        ts_ax.spines["right"].set_visible(False)
        ts_ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ts_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles2, labels2 = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle("CBR determinant regressions: library baseline vs hybrid RUONIA", fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.95))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, axes


def compute_determinant_r2_summary(
    term_premium_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    tenors: Sequence[int] = (24, 60, 120),
) -> pd.DataFrame:
    merged = prepare_determinant_sample(term_premium_df=term_premium_df, factors_df=factors_df)
    factor_specs = {
        "cpi_mmsa_lag1": ["cpi_mmsa_lag1"],
        "cpi_exp_pct": ["cpi_exp_pct"],
        "usd_vol_lag1": ["usd_vol_lag1"],
        "full_model": ["cpi_mmsa_lag1", "cpi_exp_pct", "usd_vol_lag1"],
    }

    rows: list[dict[str, float | int | str]] = []
    for tenor in tenors:
        y = merged[f"tp_M{tenor:03d}"].astype(float)
        for spec_name, columns in factor_specs.items():
            x = sm.add_constant(merged[columns].astype(float))
            result = sm.OLS(y, x).fit()
            rows.append(
                {
                    "tenor_months": tenor,
                    "spec": spec_name,
                    "r2": float(result.rsquared),
                    "adj_r2": float(result.rsquared_adj),
                    "n_obs": int(result.nobs),
                }
            )

    return pd.DataFrame(rows)


def plot_determinant_r2_summary(
    r2_summary: pd.DataFrame,
    output_path: str | Path | None = None,
):
    import matplotlib.pyplot as plt

    spec_labels = {
        "cpi_mmsa_lag1": "CPI_mmSA only",
        "cpi_exp_pct": "CPI_exp only",
        "usd_vol_lag1": "USD_vol only",
        "full_model": "Full model",
    }
    spec_colors = {
        "cpi_mmsa_lag1": "#0b6e4f",
        "cpi_exp_pct": "#d95d39",
        "usd_vol_lag1": "#4169e1",
        "full_model": "#10212b",
    }
    model_titles = {
        "library": "Library baseline",
        "hybrid_ruonia": "Hybrid RUONIA",
    }
    tenor_labels = {24: "2Y", 60: "5Y", 120: "10Y"}

    models = [model for model in ("library", "hybrid_ruonia") if model in r2_summary["model"].unique()]
    if not models:
        raise ValueError("Expected at least one of: 'library', 'hybrid_ruonia' in r2_summary['model']")

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 4.8), sharey=True, facecolor="#f5f1e8")
    if len(models) == 1:
        axes = [axes]

    bar_specs = ["cpi_mmsa_lag1", "cpi_exp_pct", "usd_vol_lag1", "full_model"]
    width = 0.18
    x = np.arange(3)

    for ax, model_name in zip(axes, models):
        subset = r2_summary[r2_summary["model"] == model_name].copy()
        for idx, spec in enumerate(bar_specs):
            tmp = subset[subset["spec"] == spec].sort_values("tenor_months")
            xpos = x + (idx - 1.5) * width
            values = [tmp.loc[tmp["tenor_months"] == tenor, "r2"].iloc[0] if tenor in tmp["tenor_months"].values else np.nan for tenor in (24, 60, 120)]
            ax.bar(xpos, values, width=width, color=spec_colors[spec], label=spec_labels[spec], alpha=0.95)

        ax.set_xticks(x)
        ax.set_xticklabels([tenor_labels[t] for t in (24, 60, 120)])
        ax.set_title(f"{model_titles.get(model_name, model_name)}: R² by specification", loc="left")
        ax.set_ylabel("R²")
        ax.set_ylim(0, max(0.55, r2_summary["r2"].max() * 1.15))
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Determinant regressions: explanatory power by tenor", fontsize=16, y=1.05)
    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.96))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig, axes


if __name__ == "__main__":
    output_dir = Path("output")
    observed_factors = pd.read_csv(output_dir / "cbr_observed_factors.csv", parse_dates=["month_end"])
    library_tp = pd.read_csv(output_dir / "lib_term_premium_curve_short.csv", parse_dates=["date"])
    hybrid_tp = pd.read_csv(output_dir / "lib_term_premium_ruonia_short.csv", parse_dates=["date"])

    library_det_results = pd.read_csv(output_dir / "lib_ruonia_determinants_library.csv")
    hybrid_det_results = pd.read_csv(output_dir / "lib_ruonia_determinants_hybrid.csv")

    if "model" not in library_det_results.columns:
        library_det_results.insert(0, "model", "library")
    if "model" not in hybrid_det_results.columns:
        hybrid_det_results.insert(0, "model", "hybrid_ruonia")

    determinant_results = pd.concat([library_det_results, hybrid_det_results], ignore_index=True)
    model_samples = {
        "library": prepare_determinant_sample(library_tp, observed_factors),
        "hybrid_ruonia": prepare_determinant_sample(hybrid_tp, observed_factors),
    }

    output_path = output_dir / "lib_ruonia_determinants.png"
    plot_tp_determinant_regressions(determinant_results, model_samples, output_path=output_path)
    print(f"Saved determinant visualization to {output_path}")

    library_r2 = compute_determinant_r2_summary(library_tp, observed_factors)
    library_r2.insert(0, "model", "library")
    hybrid_r2 = compute_determinant_r2_summary(hybrid_tp, observed_factors)
    hybrid_r2.insert(0, "model", "hybrid_ruonia")
    r2_summary = pd.concat([library_r2, hybrid_r2], ignore_index=True)
    r2_summary_path = output_dir / "lib_ruonia_determinants_r2.csv"
    r2_summary.to_csv(r2_summary_path, index=False)
    r2_plot_path = output_dir / "lib_ruonia_determinants_r2.png"
    plot_determinant_r2_summary(r2_summary, output_path=r2_plot_path)
    print(f"Saved determinant R2 summary to {r2_summary_path}")
    print(f"Saved determinant R2 visualization to {r2_plot_path}")
