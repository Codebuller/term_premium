from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import clear_output, display

from build_moex_curve import DEFAULT_MONTHS, build_moex_curve
from run_acm_model import (
    DEFAULT_DAILY_OUTPUT,
    DEFAULT_DYNAMIC_PATH,
    DEFAULT_MONTHLY_OUTPUT,
    SHORT_RATE_PROXY_CURVE_1M,
    SHORT_RATE_PROXY_RUONIA_1M,
    run_acm_model,
)


DEFAULT_VARIANTS: dict[str, dict[str, Any]] = {
    "base_ruonia": {
        "selected_maturities": [12, 24, 60, 120],
        "date_from": "2018-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_RUONIA_1M,
        "n_factors": 3,
    },
}


class TPSensitivityManager:
    def __init__(self, variants: dict[str, dict[str, Any]] | None = None):
        base = DEFAULT_VARIANTS if variants is None else variants
        self.variants: dict[str, dict[str, Any]] = {
            label: dict(config) for label, config in base.items()
        }
        self.cache: dict[tuple[Any, ...], Any] = {}

    def add_or_update(
        self,
        label: str,
        selected_maturities: Sequence[int] | str | None = None,
        date_from: str | pd.Timestamp | None = None,
        date_to: str | pd.Timestamp | None = None,
        short_rate_proxy: str = SHORT_RATE_PROXY_RUONIA_1M,
        n_factors: int = 3,
    ) -> None:
        self.variants[label] = {
            "selected_maturities": parse_selected_maturities(selected_maturities),
            "date_from": None if date_from is None else str(pd.to_datetime(date_from).date()),
            "date_to": None if date_to is None else str(pd.to_datetime(date_to).date()),
            "short_rate_proxy": short_rate_proxy,
            "n_factors": int(n_factors),
        }

    def remove(self, label: str) -> None:
        self.variants.pop(label, None)

    def clear(self) -> None:
        self.variants.clear()

    def compute(
        self,
        tenor: int = 120,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return compute_tp_lines(
            variants=self.variants,
            tenor=int(tenor),
            cache=self.cache,
        )

    def plot(
        self,
        tenor: int = 120,
        diff_to: str | None = None,
        max_abs_for_plot: float = 100.0,
        figsize: tuple[int, int] = (13, 5),
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        plot_df, summary_df = self.compute(tenor=tenor)

        if plot_df.empty:
            print("No variants available")
            return plot_df, summary_df

        columns_to_plot: list[str] = []
        for column in plot_df.columns:
            series = plot_df[column]
            finite = pd.Series(series).replace([float("inf"), float("-inf")], pd.NA).dropna()
            if finite.empty:
                continue
            if finite.abs().max() > max_abs_for_plot:
                continue
            columns_to_plot.append(column)

        if not columns_to_plot:
            print("No stable series available for plotting")
            display(summary_df)
            return plot_df, summary_df

        chart_df = plot_df[columns_to_plot].copy()

        title = f"Term premium sensitivity, tenor M{int(tenor):03d}"
        ylabel = "%"
        if diff_to is not None:
            if diff_to not in chart_df.columns:
                raise ValueError(f"`diff_to` baseline '{diff_to}' is not available")
            chart_df = chart_df.sub(chart_df[diff_to], axis=0).drop(columns=[diff_to])
            title = f"Sensitivity vs {diff_to}, tenor M{int(tenor):03d}"
            ylabel = "difference, p.p."

        ax = chart_df.plot(figsize=figsize, lw=2)
        if diff_to is not None:
            ax.axhline(0.0, color="black", lw=1, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(frameon=False, ncol=2)
        plt.show()

        display(summary_df)
        return plot_df, summary_df


def parse_selected_maturities(value: str | Sequence[int] | None) -> list[int] | None:
    if value is None:
        return None

    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return None
        return [int(part) for part in parts]

    values = [int(item) for item in value]
    return values or None


def format_selected_maturities(value: Sequence[int] | None) -> str:
    if not value:
        return ""
    return ",".join(str(int(item)) for item in value)


def get_month_end_options(months: Sequence[int] = DEFAULT_MONTHS) -> list[str]:
    _, monthly_curve = build_moex_curve(
        dynamic_path=DEFAULT_DYNAMIC_PATH,
        daily_output_path=DEFAULT_DAILY_OUTPUT,
        monthly_output_path=DEFAULT_MONTHLY_OUTPUT,
        months=months,
    )
    month_end = pd.to_datetime(monthly_curve["month_end"]).sort_values().dt.strftime("%Y-%m-%d")
    return month_end.tolist()


def compute_tp_lines(
    variants: dict[str, dict[str, Any]],
    tenor: int,
    cache: dict[tuple[Any, ...], Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = {} if cache is None else cache

    frames: list[pd.DataFrame] = []
    rows: list[dict[str, Any]] = []

    for label, config in variants.items():
        normalized_selected = parse_selected_maturities(config.get("selected_maturities"))
        cache_key = (
            tuple(normalized_selected) if normalized_selected is not None else None,
            config.get("date_from"),
            config.get("date_to"),
            config.get("short_rate_proxy"),
            int(config.get("n_factors", 3)),
            tenor,
        )

        try:
            result = cache.get(cache_key)
            if result is None:
                result = run_acm_model(
                    selected_maturities=normalized_selected,
                    date_from=config.get("date_from"),
                    date_to=config.get("date_to"),
                    short_rate_proxy=config.get("short_rate_proxy"),
                    n_factors=int(config.get("n_factors", 3)),
                )
                cache[cache_key] = result

            tp_col = f"tp_M{int(tenor):03d}"
            frame = (
                result.term_premium_frame[["date", tp_col]]
                .rename(columns={tp_col: label})
                .set_index("date")
            )
            frames.append(frame)
            rows.append(
                {
                    "variant": label,
                    "status": "ok",
                    "sample_start": result.monthly_curve["month_end"].min(),
                    "sample_end": result.monthly_curve["month_end"].max(),
                    "n_obs": len(result.monthly_curve),
                    "short_rate_proxy": result.config.short_rate_proxy_name,
                    "n_factors": result.config.n_factors,
                    "selected_maturities": format_selected_maturities(result.config.selected_maturities),
                    "last_tp_pct": float(result.term_premium_frame[tp_col].iloc[-1]),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "variant": label,
                    "status": f"error: {type(exc).__name__}",
                    "sample_start": pd.NaT,
                    "sample_end": pd.NaT,
                    "n_obs": pd.NA,
                    "short_rate_proxy": config.get("short_rate_proxy"),
                    "n_factors": config.get("n_factors"),
                    "selected_maturities": format_selected_maturities(normalized_selected),
                    "last_tp_pct": pd.NA,
                    "message": str(exc),
                }
            )

    plot_df = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    summary_df = pd.DataFrame(rows)
    return plot_df, summary_df


def display_tp_sensitivity_dashboard(
    default_variants: dict[str, dict[str, Any]] | None = None,
    months: Sequence[int] = DEFAULT_MONTHS,
) -> dict[str, Any]:
    month_options = get_month_end_options(months=months)
    default_variants = DEFAULT_VARIANTS if default_variants is None else default_variants
    variants: dict[str, dict[str, Any]] = {
        label: dict(config) for label, config in default_variants.items()
    }
    cache: dict[tuple[Any, ...], Any] = {}

    variant_selector = widgets.Dropdown(
        description="Variant",
        options=["<new>", *variants.keys()],
        value="<new>",
        layout=widgets.Layout(width="260px"),
    )
    label_text = widgets.Text(
        value="variant_1",
        description="Label",
        layout=widgets.Layout(width="260px"),
    )
    tenor_dropdown = widgets.Dropdown(
        description="TP tenor",
        options=[24, 60, 120, 180],
        value=120,
        layout=widgets.Layout(width="180px"),
    )
    selected_text = widgets.Text(
        value="12,24,60,120",
        description="Maturities",
        layout=widgets.Layout(width="320px"),
    )
    date_from_dropdown = widgets.Dropdown(
        description="From",
        options=month_options,
        value=month_options[max(0, len(month_options) - 84)] if month_options else None,
        layout=widgets.Layout(width="220px"),
    )
    date_to_dropdown = widgets.Dropdown(
        description="To",
        options=month_options,
        value=month_options[-1] if month_options else None,
        layout=widgets.Layout(width="220px"),
    )
    short_rate_dropdown = widgets.Dropdown(
        description="Proxy",
        options=[
            ("curve_1m", SHORT_RATE_PROXY_CURVE_1M),
            ("ruonia_1m", SHORT_RATE_PROXY_RUONIA_1M),
        ],
        value=SHORT_RATE_PROXY_RUONIA_1M,
        layout=widgets.Layout(width="200px"),
    )
    n_factors_dropdown = widgets.Dropdown(
        description="Factors",
        options=[2, 3, 4, 5],
        value=3,
        layout=widgets.Layout(width="160px"),
    )

    add_update_button = widgets.Button(
        description="Add / Update line",
        button_style="success",
        layout=widgets.Layout(width="160px"),
    )
    remove_button = widgets.Button(
        description="Remove line",
        button_style="warning",
        layout=widgets.Layout(width="120px"),
    )
    clear_button = widgets.Button(
        description="Clear all",
        button_style="danger",
        layout=widgets.Layout(width="100px"),
    )
    refresh_button = widgets.Button(
        description="Recompute",
        layout=widgets.Layout(width="110px"),
    )

    chart_output = widgets.Output()
    message_output = widgets.Output()

    def next_variant_name() -> str:
        return f"variant_{len(variants) + 1}"

    def current_config() -> dict[str, Any]:
        return {
            "selected_maturities": parse_selected_maturities(selected_text.value),
            "date_from": date_from_dropdown.value,
            "date_to": date_to_dropdown.value,
            "short_rate_proxy": short_rate_dropdown.value,
            "n_factors": int(n_factors_dropdown.value),
        }

    def apply_config(label: str) -> None:
        if label not in variants:
            return
        config = variants[label]
        label_text.value = label
        selected_text.value = format_selected_maturities(config.get("selected_maturities"))
        date_from_dropdown.value = config.get("date_from")
        date_to_dropdown.value = config.get("date_to")
        short_rate_dropdown.value = config.get("short_rate_proxy")
        n_factors_dropdown.value = int(config.get("n_factors", 3))

    def sync_selector() -> None:
        current = variant_selector.value if variant_selector.value in variants or variant_selector.value == "<new>" else "<new>"
        variant_selector.options = ["<new>", *variants.keys()]
        variant_selector.value = current if current in variant_selector.options else "<new>"

    def write_message(text: str) -> None:
        with message_output:
            clear_output(wait=True)
            print(text)

    def redraw(*_args: Any) -> None:
        plot_df, summary_df = compute_tp_lines(
            variants=variants,
            tenor=int(tenor_dropdown.value),
            cache=cache,
        )

        with chart_output:
            clear_output(wait=True)

            if plot_df.empty:
                print("No variants on the chart yet. Configure one and click 'Add / Update line'.")
                return

            fig, ax = plt.subplots(figsize=(13, 5))
            for column in plot_df.columns:
                ax.plot(plot_df.index, plot_df[column], lw=2, label=column)

            ax.set_title(f"Term premium sensitivity, tenor M{int(tenor_dropdown.value):03d}")
            ax.set_ylabel("%")
            ax.set_xlabel("")
            ax.grid(True, axis="y", alpha=0.3)
            ax.legend(frameon=False, ncol=2)
            plt.show()

            display(summary_df)

    def on_variant_change(change: dict[str, Any]) -> None:
        if change.get("name") != "value":
            return
        label = change.get("new")
        if label and label != "<new>":
            apply_config(label)
        else:
            label_text.value = next_variant_name()

    def on_add_update(_button: widgets.Button) -> None:
        label = label_text.value.strip() or next_variant_name()
        variants[label] = current_config()
        sync_selector()
        variant_selector.value = label
        write_message(f"Saved line: {label}")
        redraw()

    def on_remove(_button: widgets.Button) -> None:
        label = variant_selector.value
        if label in variants:
            variants.pop(label, None)
            sync_selector()
            label_text.value = next_variant_name()
            write_message(f"Removed line: {label}")
            redraw()

    def on_clear(_button: widgets.Button) -> None:
        variants.clear()
        sync_selector()
        label_text.value = next_variant_name()
        write_message("Cleared all lines")
        redraw()

    variant_selector.observe(on_variant_change, names="value")
    add_update_button.on_click(on_add_update)
    remove_button.on_click(on_remove)
    clear_button.on_click(on_clear)
    refresh_button.on_click(lambda _button: redraw())
    tenor_dropdown.observe(redraw, names="value")

    controls = widgets.VBox(
        [
            widgets.HBox([variant_selector, label_text, tenor_dropdown]),
            widgets.HBox([selected_text, short_rate_dropdown, n_factors_dropdown]),
            widgets.HBox([date_from_dropdown, date_to_dropdown]),
            widgets.HBox([add_update_button, remove_button, clear_button, refresh_button]),
            message_output,
        ]
    )

    display(controls)
    display(chart_output)

    if variants:
        first_label = next(iter(variants))
        variant_selector.value = first_label
        redraw()
    else:
        redraw()

    return {
        "variants": variants,
        "cache": cache,
        "controls": controls,
        "output": chart_output,
    }
