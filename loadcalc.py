import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants for USCS units
E_STEEL = 30e6  # Steel elastic modulus (psi)
ALLOW_STRESS = 24000  # Allowable stress (psi)

# ----------------------------
# 1 Bearing Reactions
# ----------------------------
def bearing_reactions(load_positions, load_values, L):
    """
    Calculate bearing reaction forces for a simply supported beam.
    
    Parameters:
        load_positions : list of floats (positions of loads from left bearing, inches)
        load_values : list of floats (downward loads in lbf)
        L : float (distance between bearings, inches)
    
    Returns:
        RA, RB : reaction forces at bearing A (left) and B (right) in lbf
    """
    total_moment_B = sum([P * (L - x) for P, x in zip(load_values, load_positions)])
    RB = total_moment_B / L
    RA = sum(load_values) - RB
    return RA, RB


# ----------------------------
# 2 Bending Moment & Deflection
# ----------------------------
def beam_analysis(load_positions, load_values, L, E=E_STEEL, I=0.4, n_points=100):
    """
    Compute bending moment and deflection along the beam.
    E: Modulus of elasticity (psi)
    I: Moment of inertia (in^4)
    """
    x = np.linspace(0, L, n_points)
    RA, RB = bearing_reactions(load_positions, load_values, L)
    M = np.zeros_like(x)
    y = np.zeros_like(x)

    for i, xi in enumerate(x):
        # Bending moment at each position
        M[i] = RA * xi - sum([P * (xi - a) if xi > a else 0 for P, a in zip(load_values, load_positions)])
        
    # Deflection (integrating twice — simplified by polynomial fit)
    y = np.cumsum(np.cumsum(M)) * (L / n_points)**2 / (E * I)
    
    return x, M, y, RA, RB


# ----------------------------
# 3 Beam Selection
# ----------------------------
def beam_selection(max_moment, allowable_stress=24000):  # 24 ksi typical
    """
    Selects a beam based on bending stress criteria.
    Uses a small mock database (replaceable with CSV or real data).
    For USCS units: moment in lbf·in, stress in psi
    """
    data = {
        'Section': ['W4x13', 'W6x15', 'W8x18', 'W10x22'],
        'Z (in^3)': [3.01, 5.01, 8.87, 13.3]  # Section modulus
    }
    df = pd.DataFrame(data)
    df['σ (psi)'] = max_moment / df['Z (in^3)']  # stress = M/Z
    df['Pass'] = df['σ (psi)'] < allowable_stress
    df = df[df['Pass']]
    
    if df.empty:
        return "No section meets the criteria."
    else:
        best = df.iloc[0]
        return f"Recommended Section: {best['Section']} (σ = {best['σ (psi)']:.0f} psi)"


def compute_shaft_stress(max_moment, dia_in, allowable_stress=24000):
    """
    Compute bending properties for a solid circular shaft and return stress.

    Parameters:
        max_moment: bending moment (lbf·in)
        dia_in: diameter in inches
        allowable_stress: allowable bending stress (psi)

    Returns:
        dict with I (in^4), Z (in^3), sigma (psi), pass(bool)
    """
    d = float(dia_in)
    I = np.pi * d**4 / 64.0
    Z = I / (d / 2.0)  # section modulus = I / c
    sigma = max_moment / Z
    ok = sigma < allowable_stress
    return {'I_in4': I, 'Z_in3': Z, 'sigma_psi': sigma, 'pass': ok}





# ----------------------------
# 4 Read Data and Optimize Bearing Positions
# ----------------------------
def optimize_bearing_positions(load_positions, load_values, L, n_search=50, min_RA=8.0, min_sep=None):
    """
    Find optimal bearing positions on right half of shaft to minimize max bending moment.
    Constraints:
    - Both bearings must be on right half (x ≥ L/2)
    - First bearing must be left of second bearing
    - Minimum spacing between bearings (5% of length)
    """
    # Minimum bearing spacing: use provided absolute value (in) if given, otherwise fraction of length
    if min_sep is None:
        min_spacing = 0.05 * L  # default: 5% of length
    else:
        min_spacing = float(min_sep)
    # Bearings must be right of center AND at least `min_RA` inches from left
    start_pos = max(L/2, float(min_RA))
    
    best_moment = float('inf')
    best_positions = (L/2, L)  # Default positions
    
    positions = np.linspace(start_pos, L, n_search)
    for p1 in positions:
        for p2 in positions:
            if p2 <= p1 + min_spacing:  # Skip if bearings too close or wrongly ordered
                continue
                
            # Calculate reactions with bearings at these positions
            # New rule: compute RB (reaction at bearing 2) from moments of loads to the LEFT of p1
            # RB * (p2 - p1) = sum( F * (p1 - x) ) for x < p1
            left_moment = sum([F * (p1 - x) for F, x in zip(load_values, load_positions) if x < p1])
            R2 = left_moment / (p2 - p1) if (p2 - p1) != 0 else 0.0
            # RA from vertical equilibrium (sum of loads + RA + RB = 0 -> RA = -sum(loads) - RB),
            # but here loads are signed (downward negative), and previous code used RA = sum(loads) - RB
            # keeping consistent convention: RA = sum(load_values) - R2
            R1 = sum(load_values) - R2
            
            # Calculate moments at all positions
            x = np.linspace(0, L, 200)
            M = np.zeros_like(x)
            for i, xi in enumerate(x):
                M[i] = (R1 * (xi - p1) if xi > p1 else 0) + \
                       (R2 * (xi - p2) if xi > p2 else 0) - \
                       sum([F * (xi - xf) if xi > xf else 0 
                           for F, xf in zip(load_values, load_positions)])
            
            max_moment = abs(M).max()
            if max_moment < best_moment:
                best_moment = max_moment
                best_positions = (p1, p2)
    
    return best_positions

# Load data from Excel file
try:
    # Read loads sheet
    df = pd.read_excel('beam_loads.xlsx', sheet_name='loads')
    load_positions = df['position_in'].tolist()
    load_values = df['load_lbf'].tolist()
    
    # Read params sheet for beam length
    params = pd.read_excel('beam_loads.xlsx', sheet_name='params')
    L = float(params.loc[0, 'L_in'])
    # Optional parameters for minimum RA (bearing 1) position
    # Accept either absolute min RA from left (`min_RA_in`) or a min distance from center
    if 'min_RA_from_center_in' in params.columns:
        min_RA = (L/2.0) + float(params.loc[0, 'min_RA_from_center_in'])
    elif 'min_RA_in' in params.columns:
        min_RA = float(params.loc[0, 'min_RA_in'])
    else:
        min_RA = 8.0
    shaft_weight = float(params.loc[0, 'shaft_weight_lbf']) if 'shaft_weight_lbf' in params.columns else 0.0
    min_sep_param = float(params.loc[0, 'min_sep_in']) if 'min_sep_in' in params.columns else None
    
    print(f"Loaded from Excel:")
    print(f"Positions (in): {load_positions}")
    print(f"Loads (lbf): {load_values}")
    print(f"Beam length (in): {L}")
    
    # Optionally add shaft weight at center
    if shaft_weight and shaft_weight != 0.0:
        # convention: downward loads are negative
        sw_val = -abs(shaft_weight)
        load_positions.append(L/2.0)
        load_values.append(sw_val)
        # make output explicit about sign and show updated lists so it's clear the shaft
        # weight is included in subsequent calculations
        print(f"Added shaft weight at center: {sw_val:.2f} lbf (downward)")
        print(f"Updated Positions (in): {load_positions}")
        print(f"Updated Loads (lbf): {load_values}")

    # Find optimal bearing positions (constrain first bearing >= min_RA inches)
    bearing_A_pos, bearing_B_pos = optimize_bearing_positions(load_positions, load_values, L, n_search=100, min_RA=min_RA, min_sep=min_sep_param)
    print(f"\nOptimal bearing positions:")
    print(f"Bearing A: {bearing_A_pos:.3f} in from left end")
    print(f"Bearing B: {bearing_B_pos:.3f} in from left end")
    
    # Calculate final results with optimized bearing positions
    x = np.linspace(0, L, 100)
    # For the selected bearing positions, compute RB from loads left of Bearing A
    left_moment_final = sum([F * (bearing_A_pos - x) for F, x in zip(load_values, load_positions) if x < bearing_A_pos])
    RB = left_moment_final / (bearing_B_pos - bearing_A_pos) if (bearing_B_pos - bearing_A_pos) != 0 else 0.0
    RA = sum(load_values) - RB
    
    M = np.zeros_like(x)
    y = np.zeros_like(x)
    
    for i, xi in enumerate(x):
        M[i] = (RA * (xi - bearing_A_pos) if xi > bearing_A_pos else 0) + \
               (RB * (xi - bearing_B_pos) if xi > bearing_B_pos else 0) - \
               sum([F * (xi - xf) if xi > xf else 0 
                   for F, xf in zip(load_values, load_positions)])
except Exception as e:
    print(f"Error reading Excel file: {e}")
    print("Please make sure:")
    print("1. The file 'beam_loads.xlsx' exists in the same folder as this script")
    print("2. It has a 'loads' sheet with 'position_in' and 'load_lbf' columns")
    print("3. It has a 'params' sheet with 'L_in' column")
    raise
M_max = abs(M).max()

print(f"Bearing A Reaction: {RA:.2f} lbf")
print(f"Bearing B Reaction: {RB:.2f} lbf")
print(f"Max Bending Moment: {M_max:.2f} lbf·in")
print(beam_selection(M_max))
# Compute shaft bending stress for cylindrical shaft if diameter provided
try:
    shaft_dia_in = float(params.loc[0, 'shaft_dia_in']) if 'shaft_dia_in' in params.columns else 2.44
except Exception:
    shaft_dia_in = 2.44
shaft_props = compute_shaft_stress(M_max, shaft_dia_in, allowable_stress=ALLOW_STRESS)
print(f"\nShaft (solid circular) diameter: {shaft_dia_in:.3f} in")
print(f"Section modulus Z = {shaft_props['Z_in3']:.4f} in^3, I = {shaft_props['I_in4']:.4f} in^4")
print(f"Max bending stress in shaft: {shaft_props['sigma_psi']:.1f} psi ({'PASS' if shaft_props['pass'] else 'FAIL'})")

# ----------------------------
# 5 Plot FBD and Analysis Results
# ----------------------------
def plot_fbd(load_positions, load_values, L, RA, RB, bearing_A_pos, bearing_B_pos):
    """Plot Free Body Diagram showing loads and reactions"""
    plt.figure(figsize=(12, 8))
    
    # First subplot: Free Body Diagram
    plt.subplot(3, 1, 1)
    plt.title("Free Body Diagram")
    
    # Draw beam
    plt.plot([0, L], [0, 0], 'k-', linewidth=2, label='Beam')
    
    # Plot loads (downward arrows)
    max_arrow = max(abs(min(load_values)), abs(max(load_values)))
    arrow_scale = 0.12  # Adjust this to change arrow size (smaller)
    # Create labels depending on how many loads we have
    if len(load_positions) == 2:
        labels = ['Sheave+Belt', 'Fan']
    elif len(load_positions) == 3:
        labels = ['Sheave+Belt', 'Fan', 'Shaft weight']
    else:
        labels = [f'Load {i+1}' for i in range(len(load_positions))]
    for pos, load, label in zip(load_positions, load_values, labels):
        # Draw arrow
        plt.arrow(pos, 0, 0, -arrow_scale if load < 0 else arrow_scale,
                  head_width=0.012*L, head_length=0.03*L,
                  fc='red', ec='red', width=0.003*L)
        # Add load value and label
        plt.text(pos, -0.4, f'{label}\n{abs(load)} lbf',
                 ha='center', va='center')
    
    # Draw center line
    plt.axvline(x=L/2, color='gray', linestyle='--', alpha=0.5)
    plt.text(L/2, 0.8, 'Center\nBearings only\nright of here', 
            ha='center', va='center', color='gray')
    
    # Plot reactions (upward arrows)
    reaction_positions = [bearing_A_pos, bearing_B_pos]  # Actual bearing positions
    reactions = [RA, RB]
    for pos, reaction in zip(reaction_positions, reactions):
        # Draw arrow up or down depending on sign of reaction
        dy = arrow_scale if reaction >= 0 else -arrow_scale
        head_w = 0.012 * L
        head_l = 0.03 * L
        plt.arrow(pos, 0, 0, dy,
                  head_width=head_w, head_length=head_l,
                  fc='blue', ec='blue', width=0.003*L)
        # Place text above for upward reaction, below for downward
        text_y = 0.3 if reaction >= 0 else -0.4
        plt.text(pos, text_y, f'{abs(reaction):.0f} lbf',
                 ha='center', va='center', color='blue')
    
    # Add bearing symbols
    bearing_height = 0.1
    for pos in reaction_positions:
        plt.plot([pos, pos], [-bearing_height, bearing_height], 'k-', linewidth=2)
        plt.plot([pos-0.02*L, pos+0.02*L], [-bearing_height, -bearing_height], 'k-', linewidth=2)
        plt.plot([pos-0.02*L, pos+0.02*L], [bearing_height, bearing_height], 'k-', linewidth=2)
    
    plt.grid(True)
    plt.xlabel("Length (in)")
    plt.ylabel("Load Direction")
    plt.axis([-.1*L, 1.1*L, -1, 1])  # Adjust view window
    
    # Second subplot: Shear diagram
    # Compute shear V(x): sum of reactions to left of x minus sum of loads to left of x
    V = np.zeros_like(x)
    for i, xi in enumerate(x):
        shear = 0.0
        # include reaction A if to left
        if xi >= bearing_A_pos:
            shear += RA
        if xi >= bearing_B_pos:
            shear += RB
        # subtract loads left of xi (loads are signed: downward negative)
        left_loads_idx = [j for j, lp in enumerate(load_positions) if lp <= xi]
        if left_loads_idx:
            shear -= sum([load_values[j] for j in left_loads_idx])
        V[i] = shear

    plt.subplot(3, 1, 2)
    plt.plot(x, V, label="Shear (lbf)", color='green')
    plt.axhline(0, color='k', linewidth=0.5)
    plt.title("Shear Diagram")
    plt.xlabel("Length (in)")
    plt.ylabel("Shear (lbf)")
    plt.grid(True)
    plt.legend()

    # Third subplot: Bending Moment
    plt.subplot(3, 1, 3)
    plt.plot(x, M, label="Bending Moment (lbf·in)", color='tab:blue')
    plt.title("Bending Moment Diagram")
    plt.xlabel("Length (in)")
    plt.ylabel("Moment (lbf·in)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Call the plotting function with current values
plot_fbd(load_positions, load_values, L, RA, RB, bearing_A_pos, bearing_B_pos)
