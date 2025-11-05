# python-load-calc
Shaft Optimization in a Belt Driven Fan System using Python

This repository contains a small Python toolset to analyze shaft/beam loads, find bearing positions,
and visualize the free-body diagram, shear, and bending moment. It is still under progress.

The primary scripts are:

- `loadcalc.py` — main analysis and plotting (reads `beam_loads.xlsx`).
- `list_forces.py` — lists all point loads and sign convention for quick review.

Key points
- Units: USCS (inches, lbf, lbf·in, psi).
- Sign convention: positive = upward, negative = downward.

Prerequisites
- Python 3.10+ recommended
- Required packages: numpy, pandas, matplotlib, openpyxl

Quick setup (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Running the tools
- Run the main analysis and show plots:
```powershell
python loadcalc.py
```

- List configured point loads and sign conventions:
```powershell
python list_forces.py
```

Input data (`beam_loads.xlsx`)
- `loads` sheet (required):
  - `position_in` — location along shaft measured from left end (in)
  - `load_lbf` — point load in lbf (use negative values for downward loads)

- `params` sheet (recommended): a single-row table with any of these columns:
  - `L_in` — total shaft/beam length in inches (required)
  - `shaft_weight_lbf` — shaft weight (scalar). The scripts will append this as a downward point load at `L_in/2` if present.
  - `min_RA_in` or `min_RA_from_center_in` — minimum allowed position for Bearing A (in)
  - `min_sep_in` — minimum separation between Bearing A and B (in)
  - `shaft_dia_in` — shaft diameter (in) for solid-shaft stress check

Notes and conventions
- Bearings are placed on the right half of the shaft by default (per project requirements).
- The optimizer uses a brute-force grid search; for larger problems this can be replaced with a numeric optimizer.
- Plots: Free Body Diagram, Shear diagram, and Bending Moment diagram are opened interactively.
