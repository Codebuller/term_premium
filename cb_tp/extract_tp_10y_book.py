from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy.signal import find_peaks


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "inputs" / "tp_10y_book.png"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

X_LEFT = 153
X_RIGHT = 1711
Y_MAX = 5.3
Y_TICK = 0.5
START_DATE = "2003-01-31"
END_DATE = "2025-02-28"


def load_image() -> np.ndarray:
    return np.array(Image.open(IMAGE_PATH).convert("RGB")).astype(np.int16)


def build_masks(image: np.ndarray) -> dict[str, np.ndarray]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]

    return {
        "black": (r < 60) & (g < 60) & (b < 60),
        "grid": (
            (r > 180)
            & (r < 240)
            & (np.abs(r - g) < 12)
            & (np.abs(r - b) < 12)
            & (np.abs(g - b) < 12)
        ),
    }


def detect_y_bounds(grid_mask: np.ndarray, x_left: int, x_right: int) -> tuple[int, int]:
    counts = grid_mask[:, x_left : x_right + 1].sum(axis=1)
    peaks, _ = find_peaks(counts, distance=20, prominence=40)
    peaks = peaks[(peaks > 20) & (peaks < grid_mask.shape[0] - 20)]
    if len(peaks) < 2:
        raise RuntimeError("Could not detect enough horizontal grid lines")

    step_px = int(round(np.median(np.diff(peaks[:10]))))
    y_top = int(peaks[0])
    y_bottom = 867
    # y_bottom = int(y_top + step_px * int(round(Y_MAX / Y_TICK)))
    return y_top, y_bottom


def first_true_row(mask_2d: np.ndarray, default_row: int) -> np.ndarray:
    return np.where(mask_2d.any(axis=0), mask_2d.argmax(axis=0), default_row).astype(float)


def row_to_value(rows: np.ndarray, y_top: int, y_bottom: int) -> np.ndarray:
    plot_height = y_bottom - y_top
    return (plot_height - rows) / plot_height * Y_MAX


def save_overlay(
    image_path: Path,
    output_path: Path,
    x_left: int,
    x_right: int,
    y_top: int,
    y_bottom: int,
    tp_rows: np.ndarray,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    points = [(int(x_left + idx), int(y_top + row)) for idx, row in enumerate(tp_rows)]
    draw.line(points, fill=(0, 255, 0), width=2)
    draw.rectangle((x_left, y_top, x_right, y_bottom), outline=(255, 165, 0), width=1)
    image.save(output_path)


def main() -> None:
    image = load_image()
    masks = build_masks(image)

    y_top, y_bottom = detect_y_bounds(masks["grid"], X_LEFT, X_RIGHT)
    black_crop = masks["black"][y_top : y_bottom + 1, X_LEFT : X_RIGHT + 1].copy()
    black_crop[-8:, :] = False

    default_row = black_crop.shape[0] - 1
    tp_rows = first_true_row(black_crop, default_row)
    tp_rows = pd.Series(tp_rows).replace(default_row, np.nan).interpolate(limit_direction="both").bfill().ffill().to_numpy()
    tp = row_to_value(tp_rows, y_top=y_top, y_bottom=y_bottom)

    dates = pd.date_range(START_DATE, END_DATE, freq="M")
    sample_positions = np.linspace(0, len(tp) - 1, len(dates))
    monthly = pd.DataFrame(
        {
            "date": dates,
            "tenor_years": 10,
            "source_file": "inputs/tp_10y_book.png",
            "series_kind": "tp",
            "tp": np.interp(sample_positions, np.arange(len(tp)), tp),
        }
    )

    pixel = pd.DataFrame(
        {
            "source_file": "inputs/tp_10y_book.png",
            "tenor_years": 10,
            "series_kind": "tp",
            "pixel_x": np.arange(X_LEFT, X_RIGHT + 1),
            "tp": tp,
        }
    )

    monthly_path = OUTPUT_DIR / "tp_10y_book_digitized_monthly.csv"
    pixel_path = OUTPUT_DIR / "tp_10y_book_digitized_pixels.csv"
    overlay_path = OUTPUT_DIR / "overlay_tp_10y_book.png"

    monthly.to_csv(monthly_path, index=False)
    pixel.to_csv(pixel_path, index=False)
    save_overlay(IMAGE_PATH, overlay_path, X_LEFT, X_RIGHT, y_top, y_bottom, tp_rows)

    print(f"Saved monthly data to {monthly_path}")
    print(f"Saved pixel data to {pixel_path}")
    print(f"Saved overlay to {overlay_path}")


if __name__ == "__main__":
    main()
