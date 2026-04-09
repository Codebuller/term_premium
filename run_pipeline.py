from __future__ import annotations

from pathlib import Path

from build_moex_curve import (
    DEFAULT_DAILY_OUTPUT,
    DEFAULT_DYNAMIC_PATH,
    DEFAULT_MONTHLY_OUTPUT,
    DEFAULT_MONTHS,
    build_moex_curve,
)
from excess_returns import run_excess_returns
from factor_analysis import run_factor_analysis
from market_price_of_risk import run_market_price_of_risk
from risk_neutral_curve import run_risk_neutral_curve
from ruonia import run_ruonia_monthly
from short_rate import run_short_rate
from term_premium import run_term_premium
from var import run_var
from visualizations import save_term_premium_decomposition


CLEAN_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CLEAN_DIR / "output"
TERM_PREMIUM_REPORT_PATH = OUTPUT_DIR / "moex_curve_term_premium_report.png"


def run_pipeline() -> dict[str, object]:
    daily_curve, monthly_curve = build_moex_curve(
        dynamic_path=DEFAULT_DYNAMIC_PATH,
        daily_output_path=DEFAULT_DAILY_OUTPUT,
        monthly_output_path=DEFAULT_MONTHLY_OUTPUT,
        months=DEFAULT_MONTHS,
    )
    ruonia_monthly = run_ruonia_monthly()
    factor_scores, factor_loadings, factor_summary = run_factor_analysis(curve_df=monthly_curve)
    var_params, var_residuals, var_fitted, var_summary = run_var(factors_df=factor_scores)
    excess_returns_panel, excess_returns_summary = run_excess_returns(
        curve_df=monthly_curve,
        short_rate_df=ruonia_monthly,
    )
    short_rate_params, short_rate_fitted, short_rate_residuals, short_rate_summary = run_short_rate(
        factors_df=factor_scores,
        short_rate_df=ruonia_monthly,
    )
    (
        lambda_params,
        rx_regression_params,
        rx_fitted,
        rx_residuals,
        lambda_summary,
    ) = run_market_price_of_risk(
        factors_df=factor_scores,
        var_residuals_df=var_residuals,
        excess_returns_df=excess_returns_panel,
    )
    risk_neutral_df, risk_neutral_summary = run_risk_neutral_curve(
        curve_df=monthly_curve,
        factors_df=factor_scores,
        var_params_df=var_params,
        var_residuals_df=var_residuals,
        short_rate_params_df=short_rate_params,
        lambda_params_df=lambda_params,
    )
    term_premium_df, term_premium_selected_df, term_premium_summary = run_term_premium(
        curve_df=monthly_curve,
        risk_neutral_df=risk_neutral_df,
    )
    term_premium_report = save_term_premium_decomposition(
        term_premium_df=term_premium_df,
        output_path=TERM_PREMIUM_REPORT_PATH,
    )

    return {
        "daily_curve": daily_curve.shape,
        "monthly_curve": monthly_curve.shape,
        "ruonia_monthly": ruonia_monthly.shape,
        "factor_scores": factor_scores.shape,
        "factor_loadings": factor_loadings.shape,
        "factor_summary": factor_summary.shape,
        "var_params": var_params.shape,
        "var_residuals": var_residuals.shape,
        "var_fitted": var_fitted.shape,
        "var_summary": var_summary.shape,
        "excess_returns_panel": excess_returns_panel.shape,
        "excess_returns_summary": excess_returns_summary.shape,
        "short_rate_params": short_rate_params.shape,
        "short_rate_fitted": short_rate_fitted.shape,
        "short_rate_residuals": short_rate_residuals.shape,
        "short_rate_summary": short_rate_summary.shape,
        "lambda_params": lambda_params.shape,
        "rx_regression_params": rx_regression_params.shape,
        "rx_fitted": rx_fitted.shape,
        "rx_residuals": rx_residuals.shape,
        "lambda_summary": lambda_summary.shape,
        "risk_neutral": risk_neutral_df.shape,
        "risk_neutral_summary": risk_neutral_summary.shape,
        "term_premium": term_premium_df.shape,
        "term_premium_selected": term_premium_selected_df.shape,
        "term_premium_summary": term_premium_summary.shape,
        "term_premium_report": str(term_premium_report),
    }


def main() -> None:
    outputs = run_pipeline()
    for name, value in outputs.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
