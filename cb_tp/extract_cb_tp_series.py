from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import find_peaks


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@dataclass(frozen=True)
class ChartSpec:
    filename: str
    tenor_years: int
    y_max: float
    y_tick: float
    start_date: str = "2015-01-31"
    end_date: str = "2024-05-31"


SPECS = (
    ChartSpec(
        filename="inputs/tp_2y.png",
        tenor_years=2,
        y_max=25.0,
        y_tick=5.0,
    ),
    ChartSpec(
        filename="inputs/tp_5y.png",
        tenor_years=5,
        y_max=20.0,
        y_tick=2.0,
    ),
    ChartSpec(
        filename="inputs/tp_10y.png",
        tenor_years=10,
        y_max=18.0,
        y_tick=2.0,
    ),
)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB")).astype(np.int16)


def build_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    red = (r > 150) & (r - g > 80) & (r - b > 80)
    blue = (b > 120) & (g > 80) & (r < 100)
    color = red | blue

    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)

    light_grey = (
        (r > 180)
        & (r < 240)
        & (diff_rg < 12)
        & (diff_rb < 12)
        & (diff_gb < 12)
    )
    dark_grey = (
        (r > 80)
        & (r < 180)
        & (diff_rg < 18)
        & (diff_rb < 18)
        & (diff_gb < 18)
    )
    return {
        "red": red,
        "blue": blue,
        "color": color,
        "light_grey": light_grey,
        "dark_grey": dark_grey,
    }


def detect_x_bounds(color_mask: np.ndarray) -> tuple[int, int]:
    height, width = color_mask.shape
    row_counts = color_mask.sum(axis=1)
    rows = np.where(
        (row_counts > 0.45 * width)
        & (np.arange(height) > 90)
        & (np.arange(height) < height - 120)
    )[0]
    if rows.size == 0:
        raise RuntimeError("Could not locate the filled plot area.")

    dense_segments = np.split(rows, np.where(np.diff(rows) != 1)[0] + 1)
    bottom_segment = dense_segments[-1]
    bottom_band = color_mask[bottom_segment[0]:bottom_segment[-1] + 1]

    col_counts = bottom_band.sum(axis=0)
    cols = np.where(col_counts > 0.30 * bottom_band.shape[0])[0]
    if cols.size == 0:
        raise RuntimeError("Could not locate x-axis bounds.")
    return int(cols[0]), int(cols[-1])


def detect_y_bounds(
    light_grey_mask: np.ndarray,
    x_left: int,
    x_right: int,
    spec: ChartSpec,
) -> tuple[int, int, list[int]]:
    grid_counts = light_grey_mask[:, x_left:x_right + 1].sum(axis=1)
    peaks, _ = find_peaks(grid_counts, distance=20, prominence=40)
    peaks = peaks[(peaks > 90) & (peaks < light_grey_mask.shape[0] - 140)]
    if peaks.size < 2:
        raise RuntimeError("Could not detect enough horizontal grid lines.")

    step_px = int(round(np.median(np.diff(peaks))))
    y_top = int(peaks[0])
    n_steps = int(round(spec.y_max / spec.y_tick))
    y_bottom = int(y_top + step_px * n_steps)
    return y_top, y_bottom, peaks.tolist()


def trim_plot_columns(
    color_crop: np.ndarray,
    x_left: int,
    x_right: int,
) -> tuple[int, int]:
    fill_rows = np.where(
        color_crop.any(axis=0),
        color_crop.argmax(axis=0),
        color_crop.shape[0] - 1,
    )
    good = fill_rows < 0.75 * color_crop.shape[0]
    window = 5

    left_shift = 0
    for idx in range(0, len(fill_rows) - window + 1):
        if good[idx:idx + window].all():
            left_shift = idx
            break

    right_shift = 0
    for idx in range(len(fill_rows) - window, -1, -1):
        if good[idx:idx + window].all():
            right_shift = len(fill_rows) - 1 - (idx + window - 1)
            break

    return x_left + left_shift, x_right - right_shift


def first_true_row(mask_2d: np.ndarray, default_row: int) -> np.ndarray:
    return np.where(mask_2d.any(axis=0), mask_2d.argmax(axis=0), default_row).astype(float)


def pixel_rows_to_values(rows: np.ndarray, y_top: int, y_bottom: int, y_max: float) -> np.ndarray:
    plot_height = y_bottom - y_top
    return (plot_height - rows) / plot_height * y_max


def monthly_index(spec: ChartSpec) -> pd.DatetimeIndex:
    return pd.date_range(spec.start_date, spec.end_date, freq="M")


def save_overlay(
    image_path: Path,
    output_path: Path,
    x_left: int,
    x_right: int,
    y_top: int,
    y_bottom: int,
    observed_rows: np.ndarray,
    tp_rows: np.ndarray,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    observed_points = [
        (int(x_left + idx), int(y_top + row))
        for idx, row in enumerate(observed_rows)
    ]
    tp_points = [
        (int(x_left + idx), int(y_top + row))
        for idx, row in enumerate(tp_rows)
    ]

    draw.line(observed_points, fill=(255, 255, 0), width=2)
    draw.line(tp_points, fill=(0, 255, 0), width=2)
    draw.rectangle((x_left, y_top, x_right, y_bottom), outline=(255, 165, 0), width=1)
    image.save(output_path)


def digitize_chart(spec: ChartSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    image_path = BASE_DIR / spec.filename
    image = load_rgb(image_path)
    masks = build_masks(image)

    x_left, x_right = detect_x_bounds(masks["color"])
    y_top, y_bottom, peaks = detect_y_bounds(
        masks["light_grey"],
        x_left=x_left,
        x_right=x_right,
        spec=spec,
    )

    initial_color_crop = masks["color"][y_top:y_bottom + 1, x_left:x_right + 1]
    x_left, x_right = trim_plot_columns(initial_color_crop, x_left, x_right)

    red_crop = masks["red"][y_top:y_bottom + 1, x_left:x_right + 1]
    color_crop = masks["color"][y_top:y_bottom + 1, x_left:x_right + 1]
    dark_grey_crop = masks["dark_grey"][y_top:y_bottom + 1, x_left:x_right + 1]

    default_row = color_crop.shape[0] - 1
    tp_rows = first_true_row(red_crop, default_row)
    fill_rows = first_true_row(color_crop, default_row)

    observed_rows = fill_rows.copy()
    for idx, fill_row in enumerate(fill_rows.astype(int)):
        lower = max(0, fill_row - 4)
        upper = min(dark_grey_crop.shape[0], fill_row + 5)
        grey_rows = np.where(dark_grey_crop[lower:upper, idx])[0]
        if grey_rows.size:
            observed_rows[idx] = lower + float(np.median(grey_rows))

    observed = pixel_rows_to_values(observed_rows, y_top=y_top, y_bottom=y_bottom, y_max=spec.y_max)
    tp = pixel_rows_to_values(tp_rows, y_top=y_top, y_bottom=y_bottom, y_max=spec.y_max)
    risk_neutral = observed - tp

    pixel_frame = pd.DataFrame(
        {
            "source_file": spec.filename,
            "tenor_years": spec.tenor_years,
            "pixel_x": np.arange(x_left, x_right + 1),
            "observed": observed,
            "tp": tp,
            "risk_neutral": risk_neutral,
        }
    )

    dates = monthly_index(spec)
    sample_positions = np.linspace(0, len(pixel_frame) - 1, len(dates))
    monthly_frame = pd.DataFrame(
        {
            "date": dates,
            "tenor_years": spec.tenor_years,
            "source_file": spec.filename,
            "observed": np.interp(sample_positions, np.arange(len(pixel_frame)), pixel_frame["observed"]),
            "tp": np.interp(sample_positions, np.arange(len(pixel_frame)), pixel_frame["tp"]),
        }
    )
    monthly_frame["risk_neutral"] = monthly_frame["observed"] - monthly_frame["tp"]

    overlay_name = f"overlay_{spec.tenor_years}y.png"
    save_overlay(
        image_path=image_path,
        output_path=OUTPUT_DIR / overlay_name,
        x_left=x_left,
        x_right=x_right,
        y_top=y_top,
        y_bottom=y_bottom,
        observed_rows=observed_rows,
        tp_rows=tp_rows,
    )

    print(
        f"{spec.tenor_years}Y: "
        f"x=[{x_left}, {x_right}], y=[{y_top}, {y_bottom}], "
        f"grid peaks={peaks[:6]}"
    )
    return pixel_frame, monthly_frame


def main() -> None:
    pixel_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []

    for spec in SPECS:
        pixel_frame, monthly_frame = digitize_chart(spec)
        pixel_frames.append(pixel_frame)
        monthly_frames.append(monthly_frame)

    pixel_data = pd.concat(pixel_frames, ignore_index=True)
    monthly_data = pd.concat(monthly_frames, ignore_index=True)

    pixel_path = OUTPUT_DIR / "cb_tp_digitized_pixels.csv"
    monthly_path = OUTPUT_DIR / "cb_tp_digitized_monthly.csv"

    pixel_data.to_csv(pixel_path, index=False)
    monthly_data.to_csv(monthly_path, index=False)

    summary = (
        monthly_data.groupby("tenor_years")[["observed", "tp", "risk_neutral"]]
        .agg(["min", "max"])
        .round(3)
    )
    print()
    print(summary)
    print()
    print(f"Saved pixel-level data to {pixel_path}")
    print(f"Saved monthly data to {monthly_path}")


if __name__ == "__main__":
    main()
