import pandas as pd
import numpy as np

"""
list_forces.py

Reads `beam_loads.xlsx` and prints all point loads with clear sign convention.
- Loads sheet expected columns: position_in, load_lbf
- Params sheet may include L_in, shaft_weight_lbf, min_RA_in, min_RA_from_center_in, min_sep_in, shaft_dia_in

Convention used in this workspace:
- Positive load value = upward force (lbf)
- Negative load value = downward force (lbf)

The script also prints a simple end-bearing reaction estimate (supports at x=0 and x=L):
  RB = sum(P*(L-x)) / L
  RA = sum(P) - RB

Run: python list_forces.py
"""

def simple_end_reactions(positions, loads, L):
    # RB from moments about A (left support at 0)
    total_moment_about_A = sum([P * (L - x) for P, x in zip(loads, positions)])
    RB = total_moment_about_A / L
    RA = sum(loads) - RB
    return RA, RB


def label_loads(load_positions, load_values, L):
    labels = []
    # Common convention: if there are exactly 2 loads, label them Sheave+Belt and Fan
    if len(load_positions) == 2:
        labels = ['Sheave+Belt', 'Fan']
    else:
        # if a load is very near the center, call it Shaft weight (if nearly equal to shaft_weight param later)
        for p in load_positions:
            if abs(p - L/2.0) < 1e-3:
                labels.append('Shaft weight (center)')
            else:
                labels.append('Load')
    return labels


def main():
    try:
        df = pd.read_excel('beam_loads.xlsx', sheet_name='loads')
    except Exception as e:
        print('Error: could not read beam_loads.xlsx loads sheet:', e)
        return

    try:
        params = pd.read_excel('beam_loads.xlsx', sheet_name='params')
    except Exception:
        params = pd.DataFrame()

    # Read positions and loads (USCS)
    if 'position_in' not in df.columns or 'load_lbf' not in df.columns:
        print("Expected 'position_in' and 'load_lbf' columns in 'loads' sheet.")
        print("Got columns:", list(df.columns))
        return

    positions = df['position_in'].astype(float).tolist()
    loads = df['load_lbf'].astype(float).tolist()

    L = float(params.loc[0, 'L_in']) if ('L_in' in params.columns and len(params) > 0) else max(positions) * 2.0

    print('\nForces found in beam_loads.xlsx (USCS):')
    print('Convention: positive = upward, negative = downward (lbf)')
    print('Beam length L =', L, 'in')

    # If shaft_weight_lbf present in params and not already in loads, append it to a
    # temporary view so the printed list shows what will be used in calculations.
    if 'shaft_weight_lbf' in params.columns and len(params) > 0:
        sw = float(params.loc[0, 'shaft_weight_lbf'])
        center = L/2.0
        found_center = any(abs(p - center) < 1e-3 for p in positions)
        if not found_center:
            # append for display only
            positions_display = positions + [center]
            loads_display = loads + [-abs(sw)]
            appended_sw = True
        else:
            positions_display = positions
            loads_display = loads
            appended_sw = False
    else:
        positions_display = positions
        loads_display = loads
        appended_sw = False

    print('\nIndex  Position (in)   Load (lbf)   Direction   Label')
    print('-----  -------------  -----------  ----------  ----------------')

    labels = label_loads(positions_display, loads_display, L)

    for i, (p, F, lab) in enumerate(zip(positions_display, loads_display, labels), start=1):
        direction = 'upward' if F > 0 else ('downward' if F < 0 else 'zero')
        print(f'{i:3d}    {p:12.3f}   {F:10.2f}   {direction:9s}  {lab}')

    if appended_sw:
        print(f"\nNote: shaft_weight_lbf = {sw} lbf from params was appended as { -abs(sw) } lbf at center for display and calculations.")

    total_load = sum(loads)
    print('\nTotal vertical load (sum of loads) =', total_load, 'lbf')

    # Simple end-support reaction estimate (A at x=0, B at x=L)
    RA, RB = simple_end_reactions(positions, loads, L)
    print('\nEstimate for simply-supported ends at x=0 and x=L:')
    print(f'  Reaction A (at x=0): {RA:.2f} lbf')
    print(f'  Reaction B (at x=L): {RB:.2f} lbf')

    # List sign summary and totals in SI if user wants (convert)
    toN = 4.44822162
    tom = 0.0254
    print('\n(Optional conversions)')
    print(f'  Total vertical load = {total_load:.2f} lbf = {total_load * toN:.1f} N')

if __name__ == "__main__":
    main()
