from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "term_rate_mpl_cache"),
)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox
import pandas as pd

from acm_interactive import (
    TPSensitivityManager,
    format_selected_maturities,
    parse_selected_maturities,
)
from run_acm_model import (
    SHORT_RATE_PROXY_CURVE_1M,
    SHORT_RATE_PROXY_RUONIA_1M,
)


MATURITY_PRESETS: dict[str, dict[str, Any]] = {
    "6..120 x6": {
        "values": [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
        "expr": "[6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]",
    },
    "6..180 x6": {
        "values": list(range(6, 181, 6)),
        "expr": "tuple(range(6, 181, 6))",
    },
    "1..180": {
        "values": list(range(1, 181, 1)),
        "expr": "tuple(range(1, 181, 1))",
    },
    "1..120": {
        "values": list(range(1, 121, 1)),
        "expr": "tuple(range(1, 121, 1))",
    },
    "1..180 x12": {
        "values": list(range(1, 181, 12)),
        "expr": "tuple(range(1, 181, 12))",
    },
    "1..120 x12": {
        "values": list(range(1, 121, 12)),
        "expr": "tuple(range(1, 121, 12))",
    },
}


DEMO_VARIANTS: dict[str, dict[str, Any]] = {
    "base_ruonia": {
        "selected_maturities": [12, 24, 60, 120],
        "date_from": "2018-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_RUONIA_1M,
        "n_factors": 3,
    },
    "curve1m_proxy": {
        "selected_maturities": [12, 24, 60, 120],
        "date_from": "2018-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_CURVE_1M,
        "n_factors": 3,
    },
    "short_sample": {
        "selected_maturities": [12, 24, 60, 120],
        "date_from": "2020-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_RUONIA_1M,
        "n_factors": 3,
    },
    "longer_sample": {
        "selected_maturities": [12, 24, 60, 120],
        "date_from": "2016-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_RUONIA_1M,
        "n_factors": 3,
    },
    "broader_maturities": {
        "selected_maturities": [12, 24, 36, 60, 84, 120, 180],
        "date_from": "2018-01-31",
        "date_to": "2024-12-31",
        "short_rate_proxy": SHORT_RATE_PROXY_RUONIA_1M,
        "n_factors": 3,
    },
}


class ACMInteractiveApp:
    def __init__(self) -> None:
        self.manager = TPSensitivityManager(variants=DEMO_VARIANTS)
        self.fig = plt.figure(figsize=(17.5, 10.2))
        self.fig.canvas.manager.set_window_title("ACM Term Premium Sensitivity")
        self.selected_preset_name: str | None = None
        self.preset_buttons: dict[str, Button] = {}

        self.ax_plot = self.fig.add_axes([0.42, 0.10, 0.55, 0.82])
        self.ax_summary = self.fig.add_axes([0.03, 0.02, 0.36, 0.16])
        self.ax_status = self.fig.add_axes([0.03, 0.19, 0.36, 0.06])
        self.ax_summary.axis("off")
        self.ax_status.axis("off")

        self.label_box = TextBox(
            self.fig.add_axes([0.05, 0.93, 0.31, 0.04]),
            "Label",
            initial="base_ruonia",
        )
        self.selected_box = TextBox(
            self.fig.add_axes([0.05, 0.87, 0.31, 0.04]),
            "Maturities",
            initial="12,24,60,120",
        )
        self.ax_preset_title = self.fig.add_axes([0.05, 0.83, 0.31, 0.025])
        self.ax_preset_title.axis("off")
        self.ax_preset_title.text(
            0.0,
            0.5,
            "Preset maturities",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )
        preset_positions = [
            ("6..120 x6", [0.05, 0.78, 0.145, 0.04]),
            ("6..180 x6", [0.215, 0.78, 0.145, 0.04]),
            ("1..180", [0.05, 0.73, 0.145, 0.04]),
            ("1..120", [0.215, 0.73, 0.145, 0.04]),
            ("1..180 x12", [0.05, 0.68, 0.145, 0.04]),
            ("1..120 x12", [0.215, 0.68, 0.145, 0.04]),
        ]
        for preset_name, coords in preset_positions:
            button = Button(
                self.fig.add_axes(coords),
                preset_name,
                color="#f2f2f2",
                hovercolor="#d9ecff",
            )
            button.on_clicked(lambda _event, name=preset_name: self.on_select_preset(name))
            self.preset_buttons[preset_name] = button
        self.from_box = TextBox(
            self.fig.add_axes([0.05, 0.61, 0.15, 0.04]),
            "From",
            initial="2018-01-31",
        )
        self.to_box = TextBox(
            self.fig.add_axes([0.21, 0.61, 0.15, 0.04]),
            "To",
            initial="2024-12-31",
        )
        self.baseline_box = TextBox(
            self.fig.add_axes([0.05, 0.55, 0.31, 0.04]),
            "Baseline",
            initial="base_ruonia",
        )

        self.proxy_radio = RadioButtons(
            self.fig.add_axes([0.05, 0.30, 0.14, 0.18]),
            labels=["ruonia_1m", "curve_1m"],
            active=0,
        )
        self.factor_radio = RadioButtons(
            self.fig.add_axes([0.21, 0.30, 0.08, 0.18]),
            labels=["2", "3", "4", "5"],
            active=1,
        )
        self.tenor_radio = RadioButtons(
            self.fig.add_axes([0.31, 0.28, 0.06, 0.20]),
            labels=["24", "60", "120", "180"],
            active=2,
        )
        self.mode_radio = RadioButtons(
            self.fig.add_axes([0.05, 0.22, 0.14, 0.07]),
            labels=["level", "diff"],
            active=0,
        )

        self.load_button = Button(
            self.fig.add_axes([0.05, 0.48, 0.14, 0.05]),
            "Load",
        )
        self.update_button = Button(
            self.fig.add_axes([0.22, 0.48, 0.14, 0.05]),
            "Add / Update",
        )
        self.remove_button = Button(
            self.fig.add_axes([0.05, 0.42, 0.14, 0.05]),
            "Remove",
        )
        self.clear_button = Button(
            self.fig.add_axes([0.22, 0.42, 0.14, 0.05]),
            "Clear all",
        )
        self.recompute_button = Button(
            self.fig.add_axes([0.22, 0.22, 0.14, 0.05]),
            "Recompute",
        )

        self.load_button.on_clicked(self.on_load)
        self.update_button.on_clicked(self.on_add_or_update)
        self.remove_button.on_clicked(self.on_remove)
        self.clear_button.on_clicked(self.on_clear)
        self.recompute_button.on_clicked(self.on_recompute)
        self.selected_box.on_submit(self.on_selected_text_submit)
        self.tenor_radio.on_clicked(lambda _label: self.redraw())
        self.mode_radio.on_clicked(lambda _label: self.redraw())

        self.refresh_preset_button_styles()
        self.write_status(
            "Loaded demo variants. Use preset buttons for maturities or load an existing label."
        )
        self.redraw()

    def current_label(self) -> str:
        return self.label_box.text.strip()

    def current_config(self) -> dict[str, Any]:
        selected_text = self.selected_box.text.strip()
        selected_maturities = None if not selected_text else selected_text
        return {
            "selected_maturities": selected_maturities,
            "date_from": self.from_box.text.strip() or None,
            "date_to": self.to_box.text.strip() or None,
            "short_rate_proxy": self.proxy_radio.value_selected,
            "n_factors": int(self.factor_radio.value_selected),
        }

    def identify_preset(self, selected_maturities: Any) -> str | None:
        normalized_values = parse_selected_maturities(selected_maturities)
        normalized = format_selected_maturities(normalized_values)
        if not normalized:
            return None
        for preset_name, preset in MATURITY_PRESETS.items():
            if normalized == format_selected_maturities(preset["values"]):
                return preset_name
        return None

    def refresh_preset_button_styles(self) -> None:
        for preset_name, button in self.preset_buttons.items():
            facecolor = "#cfe8ff" if preset_name == self.selected_preset_name else "#f2f2f2"
            button.ax.set_facecolor(facecolor)
        self.fig.canvas.draw_idle()

    def on_select_preset(self, preset_name: str) -> None:
        preset = MATURITY_PRESETS[preset_name]
        self.selected_preset_name = preset_name
        self.selected_box.set_val(format_selected_maturities(preset["values"]))
        self.refresh_preset_button_styles()
        self.write_status(f"Selected preset: {preset['expr']}")

    def on_selected_text_submit(self, text: str) -> None:
        self.selected_preset_name = self.identify_preset(text)
        self.refresh_preset_button_styles()

    def apply_config_to_controls(self, label: str, config: dict[str, Any]) -> None:
        self.label_box.set_val(label)
        self.selected_box.set_val(format_selected_maturities(config.get("selected_maturities")))
        self.selected_preset_name = self.identify_preset(config.get("selected_maturities"))
        self.refresh_preset_button_styles()
        self.from_box.set_val(config.get("date_from") or "")
        self.to_box.set_val(config.get("date_to") or "")
        self.baseline_box.set_val(self.baseline_box.text.strip() or "base_ruonia")

        proxy_target = config.get("short_rate_proxy", SHORT_RATE_PROXY_RUONIA_1M)
        for index, candidate in enumerate(self.proxy_radio.labels):
            if candidate.get_text() == proxy_target:
                self.proxy_radio.set_active(index)
                break

        factor_target = str(int(config.get("n_factors", 3)))
        for index, candidate in enumerate(self.factor_radio.labels):
            if candidate.get_text() == factor_target:
                self.factor_radio.set_active(index)
                break

    def write_status(self, text: str) -> None:
        self.ax_status.clear()
        self.ax_status.axis("off")
        self.ax_status.text(
            0.0,
            0.5,
            text,
            va="center",
            ha="left",
            fontsize=10,
            family="monospace",
        )
        self.fig.canvas.draw_idle()

    def build_summary_text(self, summary_df: pd.DataFrame) -> str:
        if summary_df.empty:
            return "No variants"

        lines = [
            "Variants on chart:",
            "",
        ]
        for _, row in summary_df.iterrows():
            variant = str(row.get("variant", ""))
            status = str(row.get("status", ""))
            proxy = str(row.get("short_rate_proxy", ""))
            factors = row.get("n_factors", "")
            last_tp = row.get("last_tp_pct", "")
            last_tp_text = "NA" if pd.isna(last_tp) else f"{float(last_tp):.3f}"
            lines.append(
                f"{variant:<18} {status:<12} tp_last={last_tp_text:<8} proxy={proxy:<10} K={factors}"
            )
        return "\n".join(lines)

    def redraw(self) -> None:
        tenor = int(self.tenor_radio.value_selected)
        diff_mode = self.mode_radio.value_selected == "diff"
        baseline = self.baseline_box.text.strip() or "base_ruonia"

        plot_df, summary_df = self.manager.compute(tenor=tenor)

        stable_columns: list[str] = []
        for column in plot_df.columns:
            finite = plot_df[column].replace([float("inf"), float("-inf")], pd.NA).dropna()
            if finite.empty:
                continue
            if finite.abs().max() > 100:
                continue
            stable_columns.append(column)

        self.ax_plot.clear()
        if not stable_columns:
            self.ax_plot.text(0.5, 0.5, "No stable series to plot", ha="center", va="center")
            self.ax_plot.set_axis_off()
        else:
            chart_df = plot_df[stable_columns].copy()

            if diff_mode:
                if baseline not in chart_df.columns:
                    baseline = stable_columns[0]
                    self.baseline_box.set_val(baseline)
                chart_df = chart_df.sub(chart_df[baseline], axis=0).drop(columns=[baseline])
                self.ax_plot.axhline(0.0, color="black", lw=1, alpha=0.7)
                title = f"Sensitivity vs {baseline}, tenor M{tenor:03d}"
                ylabel = "difference, p.p."
            else:
                title = f"Term premium sensitivity, tenor M{tenor:03d}"
                ylabel = "%"

            if chart_df.empty:
                self.ax_plot.text(
                    0.5,
                    0.5,
                    "Only baseline remains after diff transform",
                    ha="center",
                    va="center",
                )
            else:
                for column in chart_df.columns:
                    self.ax_plot.plot(chart_df.index, chart_df[column], lw=2, label=column)
                self.ax_plot.legend(frameon=False, ncol=2, fontsize=9)

            self.ax_plot.set_title(title)
            self.ax_plot.set_ylabel(ylabel)
            self.ax_plot.set_xlabel("")
            self.ax_plot.grid(True, axis="y", alpha=0.3)

        self.ax_summary.clear()
        self.ax_summary.axis("off")
        self.ax_summary.text(
            0.0,
            1.0,
            self.build_summary_text(summary_df),
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

        self.fig.canvas.draw_idle()

    def on_load(self, _event: object) -> None:
        label = self.current_label()
        if not label:
            self.write_status("Enter a label to load")
            return
        config = self.manager.variants.get(label)
        if config is None:
            self.write_status(f"Variant not found: {label}")
            return
        self.apply_config_to_controls(label, config)
        self.write_status(f"Loaded variant: {label}")
        self.redraw()

    def on_add_or_update(self, _event: object) -> None:
        label = self.current_label()
        if not label:
            self.write_status("Label must not be empty")
            return
        try:
            self.manager.add_or_update(label=label, **self.current_config())
            self.write_status(f"Saved variant: {label}")
            self.redraw()
        except Exception as exc:
            self.write_status(f"Save failed: {type(exc).__name__}: {exc}")

    def on_remove(self, _event: object) -> None:
        label = self.current_label()
        if not label:
            self.write_status("Enter a label to remove")
            return
        if label not in self.manager.variants:
            self.write_status(f"Variant not found: {label}")
            return
        self.manager.remove(label)
        self.write_status(f"Removed variant: {label}")
        self.redraw()

    def on_clear(self, _event: object) -> None:
        self.manager.clear()
        self.write_status("Cleared all variants")
        self.redraw()

    def on_recompute(self, _event: object) -> None:
        self.write_status("Recomputed chart")
        self.redraw()


def main() -> None:
    app = ACMInteractiveApp()
    plt.show()


if __name__ == "__main__":
    main()
