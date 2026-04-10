# Term Rate / ACM Sandbox

This repository is a small workspace for building a MOEX zero-coupon curve, running an ACM term premium model, and comparing model configurations.

## What is here

- `acm.py` — core `NominalACM` implementation.
- `build_moex_curve.py` — builds daily and monthly MOEX zero-coupon curves from `data/dynamic.csv`.
- `ruonia.py` — loads RUONIA 1M data and converts it to a monthly short-rate proxy.
- `run_acm_model.py` — main entry point for running the model with a selected sample, maturity set, factor count, and short-rate proxy.
- `acm_interactive.py` — helper utilities for sensitivity tests.
- `acm_interactive_app.py` — desktop-style interactive plot for comparing term premium series across configurations.
- `acm_walkthrough.ipynb` — notebook that maps the formulas to the code in `acm.py`.

## Expected data

The project expects these local inputs:

- `data/dynamic.csv` — MOEX curve parameters.
- `data/ruonia_1M.csv` — RUONIA 1M series.

Generated files are written to `output/`.

## Quick start

Use the project virtual environment if it already exists:

```bash
.venv/bin/python build_moex_curve.py
```

Run the ACM model from Python:

```python
from run_acm_model import run_acm_model

result = run_acm_model(
    selected_maturities=None,
    date_from="2014-01-31",
    date_to="2024-05-31",
    short_rate_proxy="ruonia_1m",
    n_factors=3,
)

print(result.summary)
print(result.term_premium_frame.tail())
```

Open the interactive comparison app:

```bash
.venv/bin/python acm_interactive_app.py
```

## Notes

- The model uses monthly estimation, even if the input curve is available at a higher frequency.
- If `short_rate_proxy` is not passed, the first curve maturity is used as the default short-rate proxy.
- In the current implementation, `tp` in `acm.py` is defined as `fitted yield - risk-neutral yield`.

## Main use cases

1. Build a monthly MOEX curve from raw parameters.
2. Run ACM on a chosen sample and maturity set.
3. Swap the short-rate proxy between curve 1M and RUONIA 1M.
4. Compare several term premium configurations side by side.
5. Read the notebook to trace each formula back to the implementation.
