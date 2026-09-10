import numpy as np
import matplotlib.pyplot as plt

# 1. Spatial grid setup and nuclear positions
x = np.linspace(-6, 6, 1000)
nuclei = [-3, 0, 3]

# 2. Initialization of arrays for the basis functions
sto = np.zeros_like(x)
gto = np.zeros_like(x)
nao = np.zeros_like(x)

# Mathematical parameters
alpha = 0.8  # Exponent for the GTO (controls the width)
rc = 2.5     # Cutoff radius for the NAO (where it is forced to zero)

# Mathematical parameters for the Plane Wave (PW)
k_pw = 1.05   # Spatial frequency factor (k); determines the oscillation
A_pw = 0.85    # Amplitude of the delocalized wave

# 3. Calculation of the sum of functions centered on each nucleus
for xi in nuclei:
    dx = np.abs(x - xi)
    
    # STO: Exact exponential decay e^(-r)
    sto += np.exp(-dx)
    
    # GTO: Gaussian decay e^(-alpha * r^2)
    gto += np.exp(-alpha * dx**2)
    
    # NAO: STO multiplied by a smooth confining potential reaching 0 at rc
    confinement = np.where(dx < rc, (1 - dx/rc)**2, 0)
    nao += np.exp(-dx) * confinement

# 4. Plane Wave (PW) calculation: Smooth delocalized function (e^ikr -> real part cos(kr))
# Representing the real component for visualization.
pw = A_pw * (np.cos(k_pw * x))**2

# 5. Figure creation
fig, ax = plt.subplots(figsize=(10, 5))

# 6. Plotting curves with academic styling
ax.plot(x, sto, '-', color='#1f77b4', linewidth=2.5, label='STO') #(Exact Cusp & Tails)
ax.plot(x, gto, '--', color='#d62728', linewidth=2.5, label='GTO') #(Zero Slope & Fast Decay)
ax.plot(x, nao, ':', color='#2ca02c', linewidth=3.0, label='NAO') #(Exact Cusp & Cutoff)
# PW: Purple dash-dot (delocalized, oscillating, and smooth across all space)
ax.plot(x, pw, '-.', color='#9467bd', linewidth=2.5, label='PW') #(Delocalized Smooth)

# 7. Styling and nucleus markers (spheres)
for xi in nuclei:
    # Vertical reference lines to guide the eye to the peak
    ax.axvline(x=xi, color='gray', linestyle='-.', alpha=0.3)
    
    # Draw the base of the sphere on the x-axis
    ax.plot(xi, 0.2, marker='o', markersize=60, color="#000000", clip_on=False, zorder=5)
    # Draw a small white highlight to give a 3D effect to the sphere
    #ax.plot(xi - 0.1, -0.01, marker='o', markersize=4, color='white', alpha=0.6, clip_on=False, zorder=6)
    # Draw a small white highlight to give a 3D effect to the sphere
    ax.plot(xi-0.2, 0.25, marker='o', markersize=5, color='white', alpha=0.6, clip_on=False, zorder=6)

# 8. Axis formatting
ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-0.1, max(sto) + 0.1)

# Y-axis tick labels are now visible (ax.set_yticks([]) was removed)
ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold')
ax.set_xlabel('Distance (Bohr)', fontsize=14, fontweight='bold')

# Remove top and right spines for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Ensure left and bottom spines maintain the same line width
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 9. Frame-free legend
ax.legend(fontsize=12, loc='upper right', frameon=False)

# Automatically adjust subplot parameters to give specified padding
plt.tight_layout()

# 9. Save the figure with transparent background and high resolution (300 dpi)
plt.savefig('basis_sets_comparison.png', transparent=True,bbox_inches="tight", dpi=300)

# Display the figure on screen if running interactively
plt.show()