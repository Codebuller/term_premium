from __future__ import annotations

from pathlib import Path
import sys
from urllib.request import urlretrieve

import numpy as np
import pandas as pd


CLEAN_DIR = Path(__file__).resolve().parent
DATA_DIR = CLEAN_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
WORKSPACE_BUNDLED_SITE_PACKAGES = Path(
    "/Users/codebuller/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages"
)

if WORKSPACE_BUNDLED_SITE_PACKAGES.exists():
    bundled_path = str(WORKSPACE_BUNDLED_SITE_PACKAGES)
    if bundled_path not in sys.path:
        sys.path.insert(0, bundled_path)

DEFAULT_CBR_MACRO_SURVEY_URL = "https://cbr.ru/Content/Document/File/144490/full.xlsx"
DEFAULT_CBR_MACRO_SURVEY_PATH = RAW_DIR / "cbr_macro_survey.xlsx"

KEY_RATE_SHEET = "3"
NEUTRAL_KEY_RATE_SHEET = "16"


def download_cbr_macro_survey(
    url: str = DEFAULT_CBR_MACRO_SURVEY_URL,
    output_path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
) -> Path:
    """
    Download the current official Bank of Russia macro survey workbook.
    """

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, path)
    return path


def ensure_cbr_macro_survey(
    path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
    url: str = DEFAULT_CBR_MACRO_SURVEY_URL,
) -> Path:
    path = Path(path)
    if not path.exists():
        return download_cbr_macro_survey(url=url, output_path=path)
    return path


def _clean_numeric(value) -> float | None:
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if value in {"", "-"}:
            return np.nan
    return pd.to_numeric(value, errors="coerce")


def _canonical_statistic(value: object) -> str:
    text = str(value).strip().casefold()
    if text in {"медиана", "median"}:
        return "Median"
    if text in {"среднее", "average"}:
        return "Average"
    if text in {"макс", "max"}:
        return "Max"
    if text in {"мин", "min"}:
        return "Min"
    if text in {"90-й процентиль", "90th percentile"}:
        return "90th percentile"
    if text in {"3-й квартиль", "3rd quartile"}:
        return "3rd quartile"
    if text in {"1-й квартиль", "1st quartile"}:
        return "1st quartile"
    if text in {"10-й процентиль", "10th percentile"}:
        return "10th percentile"
    return str(value).strip()


def _monthly_end_index(series_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    start = series_index.min().to_period("M").to_timestamp("M")
    end = series_index.max().to_period("M").to_timestamp("M")
    return pd.date_range(start=start, end=end, freq="M")


def _month_end_ffill(series: pd.Series) -> pd.Series:
    series = series.dropna().sort_index()
    if series.empty:
        return series
    month_ends = _monthly_end_index(series.index)
    monthly = series.reindex(month_ends, method="ffill")
    monthly.name = series.name
    return monthly


def percent_to_monthly_log_rate(series: pd.Series, name: str | None = None) -> pd.Series:
    """
    Convert an annualized percent-rate series to a monthly log short-rate series.
    """

    out = np.log1p(pd.to_numeric(series, errors="coerce") / 100.0) / 12.0
    if name is not None:
        out.name = name
    else:
        out.name = series.name
    return out


def parse_key_rate_panel(
    path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
    sheet_name: str = KEY_RATE_SHEET,
) -> pd.DataFrame:
    """
    Parse the key-rate sheet into a long panel with survey date and forecast period.

    Returns columns:
    - statistic
    - survey_date
    - survey_label
    - forecast_period
    - forecast_period_label
    - value
    """

    workbook_path = ensure_cbr_macro_survey(path)
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)

    survey_dates = pd.to_datetime(raw.iloc[5, 4:], errors="coerce")
    survey_labels = raw.iloc[6, 4:].astype(str).replace("nan", "")
    survey_columns = [
        (col_idx, survey_date, survey_label)
        for col_idx, survey_date, survey_label in zip(range(4, raw.shape[1]), survey_dates, survey_labels)
        if pd.notna(survey_date)
    ]

    records: list[dict[str, object]] = []
    current_statistic: str | None = None

    for row_idx in range(7, raw.shape[0]):
        stat_label = raw.iat[row_idx, 1]
        if pd.notna(stat_label):
            current_statistic = _canonical_statistic(stat_label)

        forecast_period = pd.to_datetime(raw.iat[row_idx, 3], errors="coerce")
        if pd.isna(forecast_period):
            continue

        forecast_period_label = raw.iat[row_idx, 2]
        for col_idx, survey_date, survey_label in survey_columns:
            value = _clean_numeric(raw.iat[row_idx, col_idx])
            if pd.isna(value):
                continue
            records.append(
                {
                    "statistic": current_statistic,
                    "survey_date": survey_date,
                    "survey_label": survey_label,
                    "forecast_period": forecast_period,
                    "forecast_period_label": forecast_period_label,
                    "value": float(value),
                }
            )

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        raise ValueError("Could not parse any key-rate survey records from the workbook")

    return panel.sort_values(["survey_date", "forecast_period", "statistic"]).reset_index(drop=True)


def build_cbr_key_rate_current_year_monthly(
    path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
    statistic: str = "Median",
) -> pd.Series:
    """
    Build a monthly month-end series of the current-year median key-rate survey.

    The survey workbook reports forecasts by calendar-year average. For Kim-Wright
    style anchoring we use the forecast for the survey year itself and carry the
    latest survey release forward to month-end.
    """

    panel = parse_key_rate_panel(path=path, sheet_name=KEY_RATE_SHEET)
    data = panel.copy()
    data = data[data["statistic"].astype(str).str.casefold() == _canonical_statistic(statistic).casefold()]
    data = data[data["forecast_period"].dt.year == data["survey_date"].dt.year]
    if data.empty:
        raise ValueError("No current-year key-rate forecast rows were found")

    series = (
        data.sort_values(["survey_date", "forecast_period"])
        .drop_duplicates(subset=["survey_date"], keep="last")
        .set_index("survey_date")["value"]
        .sort_index()
    )
    series.name = "cbr_key_rate_median"
    return _month_end_ffill(series)


def build_cbr_neutral_key_rate_monthly(
    path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
    statistic: str = "Median",
) -> pd.Series:
    """
    Build a monthly month-end series of the neutral key-rate survey.
    """
    panel = parse_neutral_key_rate_panel(path=path)
    panel = panel[
        panel["statistic"].astype(str).str.casefold() == _canonical_statistic(statistic).casefold()
    ].copy()
    if panel.empty:
        raise ValueError("No neutral key-rate survey rows were found")

    series = (
        panel.sort_values("survey_date")
        .drop_duplicates(subset=["survey_date"], keep="last")
        .set_index("survey_date")["value"]
        .sort_index()
    )
    series.name = "cbr_neutral_key_rate_median"
    return _month_end_ffill(series)


def parse_neutral_key_rate_panel(
    path: str | Path = DEFAULT_CBR_MACRO_SURVEY_PATH,
    sheet_name: str = NEUTRAL_KEY_RATE_SHEET,
) -> pd.DataFrame:
    """
    Parse the neutral key-rate sheet into a long panel.
    """

    workbook_path = ensure_cbr_macro_survey(path)
    raw = pd.read_excel(workbook_path, sheet_name=sheet_name, header=None)

    survey_dates = pd.to_datetime(raw.iloc[5, 3:], errors="coerce")
    survey_labels = raw.iloc[6, 3:].astype(str).replace("nan", "")
    survey_columns = [
        (col_idx, survey_date, survey_label)
        for col_idx, survey_date, survey_label in zip(range(3, raw.shape[1]), survey_dates, survey_labels)
        if pd.notna(survey_date)
    ]

    records: list[dict[str, object]] = []
    current_statistic: str | None = None
    for row_idx in range(7, raw.shape[0]):
        stat_label = raw.iat[row_idx, 1]
        if pd.notna(stat_label):
            current_statistic = _canonical_statistic(stat_label)

        for col_idx, survey_date, survey_label in survey_columns:
            value = _clean_numeric(raw.iat[row_idx, col_idx])
            if pd.isna(value):
                continue
            records.append(
                {
                    "statistic": current_statistic,
                    "survey_date": survey_date,
                    "survey_label": survey_label,
                    "value": float(value),
                }
            )

    panel = pd.DataFrame.from_records(records)
    if panel.empty:
        raise ValueError("Could not parse any neutral key-rate survey records from the workbook")

    return panel.sort_values(["survey_date", "statistic"]).reset_index(drop=True)
