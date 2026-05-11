from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import date, datetime

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "full-yield-curve"
DEFAULT_START_DATE = "01.01.2023"
DEFAULT_BATCH = relativedelta(years=5)

def build_cbr_url(start_date: str, end_date: str, posted: bool = True) -> str:
    posted_flag = "True" if posted else "False"
    return (
        "https://www.cbr.ru/hd_base/zcyc_params/"
        f"?UniDbQuery.Posted={posted_flag}"
        f"&UniDbQuery.From={start_date}"
        f"&UniDbQuery.To={end_date}"
    )


def _flatten_column_name(column: object) -> str:
    if not isinstance(column, tuple):
        return str(column).strip()
    parts = [str(part).strip() for part in column if str(part).strip() and str(part) != "nan"]
    return "__".join(parts)


def fetch_full_yield_curve(start_date: date, end_date: date, posted: bool = True) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    current_start = start_date

    while current_start <= end_date:
        batch_end_date = min(current_start + DEFAULT_BATCH - relativedelta(days=1), end_date)

        url = build_cbr_url(
            start_date=current_start.strftime("%d.%m.%Y"),
            end_date=batch_end_date.strftime("%d.%m.%Y"),
            posted=posted,
        )

        table = pd.read_html(url, decimal=",", thousands=" ")[0]
        table.columns = [_flatten_column_name(column) for column in table.columns]

        rename_map = {"Дата__Дата": "date", "Дата": "date"}

        for column in table.columns:
            if column in rename_map:
                continue
            if "__" not in column:
                continue

            _, maturity = column.split("__", 1)
            maturity = maturity.replace(",", ".")
            rename_map[column] = f"y{maturity}"

        wide = table.rename(columns=rename_map).copy()

        if "date" not in wide.columns:
            raise ValueError("Could not find date column in CBR zcyc_params table")

        wide["date"] = pd.to_datetime(wide["date"], format="%d.%m.%Y", errors="coerce")

        maturity_columns = [column for column in wide.columns if column.startswith("y")]

        for column in maturity_columns:
            wide[column] = pd.to_numeric(
                wide[column].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )

        wide = wide[["date", *maturity_columns]]
        frames.append(wide)

        current_start = batch_end_date + relativedelta(days=1)

    if not frames:
        return pd.DataFrame()

    wide_full = (
        pd.concat(frames, ignore_index=True)
        .dropna(subset=["date"])
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    maturity_columns = [column for column in wide_full.columns if column.startswith("y")]
    return wide_full[["date", *maturity_columns]]


def build_long_curve(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars="date",
        var_name="maturity_years",
        value_name="yield_pct",
    )
    long["maturity_years"] = pd.to_numeric(
        long["maturity_years"].str.removeprefix("y"),
        errors="coerce",
    )
    long = long.sort_values(["date", "maturity_years"]).reset_index(drop=True)
    return long


def save_curve_files(
    wide: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_path = output_dir / "yield_curve_wide.csv"
    long_path = output_dir / "yield_curve_long.csv"

    wide.to_csv(wide_path, index=False)
    build_long_curve(wide).to_csv(long_path, index=False)
    return wide_path, long_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the Bank of Russia zero-coupon yield curve history.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help="Start date in DD.MM.YYYY format. Default: 01.01.2023",
    )
    parser.add_argument(
        "--end-date",
        default=pd.Timestamp.today().strftime("%d.%m.%Y"),
        help="End date in DD.MM.YYYY format. Default: today",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where CSV files will be saved.",
    )
    parser.add_argument(
        "--include-unposted",
        action="store_true",
        help="Include unpublished observations if the website returns them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wide = fetch_full_yield_curve(
        start_date=datetime.strptime(args.start_date, "%d.%m.%Y"),
        end_date=datetime.strptime(args.end_date, "%d.%m.%Y"),
        posted=not args.include_unposted,
    )
    wide_path, long_path = save_curve_files(wide=wide, output_dir=args.output_dir)
    print(
        f"Saved {len(wide)} dates to {wide_path} and {len(wide) * (len(wide.columns) - 1)} rows to {long_path}"
    )


if __name__ == "__main__":
    main()
