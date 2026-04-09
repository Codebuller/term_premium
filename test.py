"""
This file replicates the term premium estimates from the original paper. The
file `us_data.xlsx` contains data from the authors' original matlab replication
files. The output of this script matches the one from the original.

For the updated US term premium estimates, visit the NY FED website.
"""
from pyacm import NominalACM
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


ylds_d = pd.read_excel("data/us_data.xlsx", index_col=0, sheet_name="daily")
ylds_d.index = pd.to_datetime(ylds_d.index)
ylds_d = ylds_d / 100

acm = NominalACM(
    curve=ylds_d,
    n_factors=5,
    selected_maturities=[6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
)


def compare_series(ours: pd.Series, official: pd.Series, label: str) -> dict:
    joined = pd.concat(
        [ours.rename("ours"), official.rename("official")],
        axis=1,
        join="inner",
    ).dropna()
    diff = joined["ours"] - joined["official"]
    return {
        "series": label,
        "n_obs": len(joined),
        "mean_diff_bp": float(diff.mean() * 10000),
        "mae_bp": float(diff.abs().mean() * 10000),
        "rmse_bp": float(np.sqrt((diff**2).mean()) * 10000),
        "max_abs_diff_bp": float(diff.abs().max() * 10000),
        "corr": float(joined["ours"].corr(joined["official"])),
    }


fed_daily = pd.read_excel("data/ACMTermPremium (1).xls", sheet_name="ACM Daily")
fed_daily["DATE"] = pd.to_datetime(fed_daily["DATE"])
fed_daily = fed_daily.set_index("DATE").sort_index() / 100

comparison_rows = []
for years in range(1, 11):
    month = years * 12
    comparison_rows.append(compare_series(acm.miy[month], fed_daily[f"ACMY{years:02d}"], f"yield_{years}Y"))
    comparison_rows.append(compare_series(acm.rny[month], fed_daily[f"ACMRNY{years:02d}"], f"rn_{years}Y"))
    comparison_rows.append(compare_series(acm.tp[month], fed_daily[f"ACMTP{years:02d}"], f"tp_{years}Y"))

comparison_df = pd.DataFrame(comparison_rows)
print("\nComparison with NY Fed ACMTermPremium (daily sheet)")
print(comparison_df.to_string(index=False))

tp10_compare = pd.concat(
    [
        acm.tp[120].rename("ours_tp_10Y"),
        fed_daily["ACMTP10"].rename("fed_tp_10Y"),
    ],
    axis=1,
    join="inner",
).dropna()


# =================
# ===== Chart =====
# =================
size = 7
fig = plt.figure(figsize=(size * (24 / 7.3), size))

ax = plt.subplot2grid((1, 3), (0, 0))
ax.plot(ylds_d[120], label="Actual Yield", lw=1)
ax.plot(acm.miy[120], label="Fitted Yield", lw=1, ls='--')
ax.set_title("10-Year Model Fit")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.tick_params(rotation=90, axis="x")
ax.legend(loc="upper right")

ax = plt.subplot2grid((1, 3), (0, 1))
ax.plot(ylds_d[120], label="Yield", lw=1)
ax.plot(acm.rny[120], label="Risk Neutral Yield", lw=1)
ax.plot(acm.tp[120], label="Term Premium", lw=1)
ax.set_title("10-Year Yield Decomposition")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.tick_params(rotation=90, axis="x")
ax.legend(loc="upper right")

ax = plt.subplot2grid((1, 3), (0, 2))
ax.plot(tp10_compare["ours_tp_10Y"], label="Our TP", lw=1)
ax.plot(tp10_compare["fed_tp_10Y"], label="Fed TP", lw=1, ls="--")
ax.set_title("10-Year TP: Our Model vs Fed")
ax.xaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
ax.tick_params(rotation=90, axis="x")
ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
