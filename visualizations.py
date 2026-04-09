from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
DEFAULT_TENOR_LABELS = {
    24: "2 года",
    60: "5 лет",
    120: "10 лет",
}
DEFAULT_COLORS = {
    "obs": "#10212b",
    "rn": "#0b6e4f",
    "tp": "#d95d39",
    "bg": "#f5f1e8",
    "grid": "#d8e0e6",
}


def _ensure_matplotlib(backend: str | None = None):
    if backend is not None:
        matplotlib_cache = CLEAN_DIR / ".cache" / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        import matplotlib

        if matplotlib.get_backend().lower() != backend.lower():
            matplotlib.use(backend)

    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    return plt, mdates


def _detect_date_column(df: pd.DataFrame) -> str | None:
    for candidate in ("month_end", "date"):
        if candidate in df.columns:
            return candidate
    return None


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    date_column = _detect_date_column(df)
    if date_column is not None:
        out = df.copy()
        out[date_column] = pd.to_datetime(out[date_column])
        return out.rename(columns={date_column: "date"}).sort_values("date").reset_index(drop=True)

    if isinstance(df.index, pd.DatetimeIndex):
        out = df.copy().reset_index()
        out.columns = ["date", *out.columns[1:]]
        out["date"] = pd.to_datetime(out["date"])
        return out.sort_values("date").reset_index(drop=True)

    raise ValueError("Could not find a datetime axis. Expected 'month_end', 'date', or DatetimeIndex.")


def _available_tp_months(df: pd.DataFrame) -> list[int]:
    tp_months = sorted(
        int(column[4:])
        for column in df.columns
        if isinstance(column, str) and column.startswith("tp_M")
    )
    if tp_months:
        return tp_months

    numeric_months: list[int] = []
    for column in df.columns:
        if isinstance(column, int):
            numeric_months.append(int(column))
        elif isinstance(column, str) and column.isdigit():
            numeric_months.append(int(column))
    return sorted(numeric_months)


def _normalize_tp_version_frame(
    tp_like_df: pd.DataFrame,
    tenors: Sequence[int] | None = None,
) -> pd.DataFrame:
    frame = _normalize_date_index(tp_like_df)
    available_months = _available_tp_months(frame)
    if not available_months:
        raise ValueError("No term premium columns found")

    selected_months = sorted(set(tenors or available_months))
    selected_months = [month for month in selected_months if month in available_months]
    if not selected_months:
        raise ValueError("Requested tenors are not present in the input data")

    output = frame[["date"]].copy()

    if any(isinstance(column, str) and column.startswith("tp_M") for column in frame.columns):
        for month in selected_months:
            output[f"tp_M{month:03d}"] = frame[f"tp_M{month:03d}"].to_numpy(dtype=float)
        return output

    sample = frame[[month if month in frame.columns else str(month) for month in selected_months]].to_numpy(dtype=float)
    scale = 100.0 if np.nanmax(np.abs(sample)) < 1.0 else 1.0
    for month in selected_months:
        column = month if month in frame.columns else str(month)
        output[f"tp_M{month:03d}"] = frame[column].to_numpy(dtype=float) * scale
    return output


def _normalize_decomposition_frame(
    term_premium_df: pd.DataFrame,
    tenors: Sequence[int] | None = None,
) -> pd.DataFrame:
    frame = _normalize_date_index(term_premium_df)
    available_months = sorted(
        int(column[5:])
        for column in frame.columns
        if isinstance(column, str) and column.startswith("obs_M")
    )
    if not available_months:
        raise ValueError("Expected decomposition columns like 'obs_M060', 'rn_M060', 'tp_M060'")

    selected_months = sorted(set(tenors or available_months))
    selected_months = [
        month
        for month in selected_months
        if all(f"{prefix}_M{month:03d}" in frame.columns for prefix in ("obs", "rn", "tp"))
    ]
    if not selected_months:
        raise ValueError("Requested tenors are not present in the decomposition dataframe")

    columns = ["date"]
    for month in selected_months:
        columns.extend([f"obs_M{month:03d}", f"rn_M{month:03d}", f"tp_M{month:03d}"])
    return frame[columns].copy()


def plot_term_premium_decomposition(
    term_premium_df: pd.DataFrame,
    tenors: Mapping[int, str] | Sequence[int] | None = None,
    title: str = "Декомпозиция доходности: term premium + ожидаемая доходность",
):
    plt, mdates = _ensure_matplotlib()

    if tenors is None:
        tenor_labels = DEFAULT_TENOR_LABELS.copy()
    elif isinstance(tenors, Mapping):
        tenor_labels = dict(tenors)
    else:
        tenor_labels = {month: f"{month // 12} лет" if month % 12 == 0 else f"{month} мес." for month in tenors}

    frame = _normalize_decomposition_frame(term_premium_df=term_premium_df, tenors=tenor_labels.keys())
    available_labels = {
        month: label
        for month, label in tenor_labels.items()
        if all(f"{prefix}_M{month:03d}" in frame.columns for prefix in ("obs", "rn", "tp"))
    }
    if not available_labels:
        raise ValueError("No configured tenors are available for the term premium plot")

    fig, axes = plt.subplots(
        nrows=len(available_labels),
        ncols=1,
        figsize=(15, 3.7 * len(available_labels)),
        sharex=True,
        facecolor=DEFAULT_COLORS["bg"],
    )
    if len(available_labels) == 1:
        axes = [axes]

    dates = frame["date"]
    for ax, (month, label) in zip(axes, available_labels.items()):
        obs = frame[f"obs_M{month:03d}"]
        rn = frame[f"rn_M{month:03d}"]
        tp = frame[f"tp_M{month:03d}"]

        ax.set_facecolor("white")
        ax.fill_between(dates, 0.0, tp, color=DEFAULT_COLORS["tp"], alpha=0.30, label="Term premium", interpolate=True)
        ax.fill_between(dates, tp, obs, color=DEFAULT_COLORS["rn"], alpha=0.22, label="Ожидаемая доходность", interpolate=True)
        ax.plot(dates, obs, color=DEFAULT_COLORS["obs"], lw=2.4, label="Доходность")
        ax.plot(dates, rn, color=DEFAULT_COLORS["rn"], lw=1.6, alpha=0.95)
        ax.plot(dates, tp, color=DEFAULT_COLORS["tp"], lw=1.0, alpha=0.9)

        ax.axhline(0.0, color="#7a8890", lw=0.9, alpha=0.8)
        ax.set_title(label, loc="left", fontsize=13, color=DEFAULT_COLORS["obs"])
        ax.set_ylabel("%, cc")
        ax.grid(True, axis="y", color=DEFAULT_COLORS["grid"], alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Дата")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout(rect=(0.03, 0.04, 0.97, 0.95))
    return fig, axes


def save_term_premium_decomposition(
    term_premium_df: pd.DataFrame,
    output_path: str | Path,
    tenors: Mapping[int, str] | Sequence[int] | None = None,
    title: str = "Декомпозиция доходности: term premium + ожидаемая доходность",
) -> Path:
    plt, _ = _ensure_matplotlib(backend="Agg")
    fig, _ = plot_term_premium_decomposition(term_premium_df=term_premium_df, tenors=tenors, title=title)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output


def plot_term_premium_comparison(
    versions: Mapping[str, pd.DataFrame],
    baseline: str | None = None,
    tenors: Sequence[int] = (24, 60, 120),
    title: str = "Сравнение term premium по версиям реализации",
):
    if not versions:
        raise ValueError("versions must not be empty")

    plt, mdates = _ensure_matplotlib()

    names = list(versions.keys())
    baseline_name = baseline or names[0]
    if baseline_name not in versions:
        raise ValueError(f"baseline '{baseline_name}' not found in versions")

    normalized = {
        name: _normalize_tp_version_frame(tp_like_df=df, tenors=tenors).set_index("date")
        for name, df in versions.items()
    }

    common_dates = None
    for frame in normalized.values():
        common_dates = frame.index if common_dates is None else common_dates.intersection(frame.index)
    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No overlapping dates across versions")
    common_dates = common_dates.sort_values()

    normalized = {name: frame.loc[common_dates].reset_index() for name, frame in normalized.items()}
    valid_tenors = [
        month
        for month in tenors
        if all(f"tp_M{month:03d}" in frame.columns for frame in normalized.values())
    ]
    if not valid_tenors:
        raise ValueError("No overlapping requested tenors across versions")

    palette = ["#10212b", "#0b6e4f", "#d95d39", "#3e5c76", "#7c6a0a", "#8a3b12"]
    colors = {name: palette[idx % len(palette)] for idx, name in enumerate(names)}

    fig, axes = plt.subplots(
        nrows=len(valid_tenors),
        ncols=2,
        figsize=(16, 4.2 * len(valid_tenors)),
        sharex="col",
        gridspec_kw={"width_ratios": [1.45, 1.0]},
    )
    if len(valid_tenors) == 1:
        axes = np.array([axes])

    for row_idx, month in enumerate(valid_tenors):
        level_ax = axes[row_idx, 0]
        diff_ax = axes[row_idx, 1]
        baseline_frame = normalized[baseline_name]
        baseline_series = baseline_frame[f"tp_M{month:03d}"]

        for name in names:
            frame = normalized[name]
            series = frame[f"tp_M{month:03d}"]
            level_ax.plot(frame["date"], series, color=colors[name], lw=1.8, label=name)
            if name != baseline_name:
                diff_ax.plot(frame["date"], series - baseline_series, color=colors[name], lw=1.6, label=f"{name} - {baseline_name}")

        level_ax.set_title(f"{month} мес.: уровни", loc="left")
        level_ax.axhline(0.0, color="#7a8890", lw=0.8, alpha=0.8)
        level_ax.grid(True, axis="y", alpha=0.35)
        level_ax.spines["top"].set_visible(False)
        level_ax.spines["right"].set_visible(False)
        level_ax.set_ylabel("%, cc")

        diff_ax.set_title(f"{month} мес.: отклонение от {baseline_name}", loc="left")
        diff_ax.axhline(0.0, color="#7a8890", lw=0.8, alpha=0.8)
        diff_ax.grid(True, axis="y", alpha=0.35)
        diff_ax.spines["top"].set_visible(False)
        diff_ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Дата")

    level_handles, level_labels = axes[0, 0].get_legend_handles_labels()
    diff_handles, diff_labels = axes[0, 1].get_legend_handles_labels()
    if level_handles:
        fig.legend(level_handles, level_labels, loc="upper center", ncol=min(len(level_handles), 4), frameon=False, bbox_to_anchor=(0.36, 0.985))
    if diff_handles:
        fig.legend(diff_handles, diff_labels, loc="upper center", ncol=min(len(diff_handles), 3), frameon=False, bbox_to_anchor=(0.80, 0.985))
    fig.suptitle(title, fontsize=17, y=0.995)
    fig.tight_layout(rect=(0.03, 0.04, 0.98, 0.95))
    return fig, axes
