"""
Gravity-Topography Correlation Analysis (Dependency-Free Version)
-----------------------------------------------------------------
This script performs Part 6 of the geophysics project.
It uses ONLY the data already present in the Bushveld dataset,
avoiding ALL external complex library dependencies.

Requirements:
pip install numpy matplotlib verde ensaio pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import verde as vd
import ensaio
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

print("--- Starting Part 6: Gravity-Topography Correlation (No Dependencies) ---")

# 1. Fetch Gravity Data (Bushveld)
print("\n1. Fetching Gravity Data (Bushveld Complex)...")
try:
    # ensaio returns the PATH to the downloaded file, not the data itself.
    fname = ensaio.fetch_bushveld_gravity(version=1)
    print(f"   Data downloaded to: {fname}")

    # We must use pandas to READ the file into a DataFrame
    data = pd.read_csv(fname)
    print("   Gravity data loaded successfully into DataFrame.")
    print("   Data columns found:", data.columns.tolist())

except Exception as e:
    print(f"   Error loading gravity data: {e}")
    exit()

# 2. Identify Elevation Column
# DIFFERENT VERSIONS OF THE DATA HAVE DIFFERENT COLUMN NAMES.
# We must check for all common variations.
elev_col = None
possible_names = [
    'height_sea_level_m',  # Most likely for new ensaio versions
    'height',              # Older versions
    'elevation',           # Common alternative
    'h'                    # Short form
]

for col in possible_names:
    if col in data.columns:
        elev_col = col
        break

if elev_col is None:
    print("\nCRITICAL ERROR: Could not find an elevation column.")
    print("Available columns are:", data.columns.tolist())
    exit()

print(f"   -> SUCCESS: Found topography data in column '{elev_col}'")

# 3. Gridding (Optional but makes maps prettier)
# We will grid BOTH gravity and topography for nice smooth maps
print("\n2. Gridding Data for visualization...")
# Get the region from the data coordinates
region = vd.get_region((data.longitude, data.latitude))
# Use a slightly larger spacing for speed and smoothness on regional data
spacing = 1/60 # ~1 minute arc spacing (approx 2km)

# Grid Gravity
print("   Gridding Gravity...")
gridder_grav = vd.Spline()
# Note: Using longitude/latitude now as that's what the new file has
gridder_grav.fit((data.longitude, data.latitude), data.gravity_bouguer_mgal)
grid_grav = gridder_grav.grid(region=region, spacing=spacing, dims=["latitude", "longitude"])

# Grid Topography
print("   Gridding Topography...")
gridder_topo = vd.Spline()
gridder_topo.fit((data.longitude, data.latitude), data[elev_col])
grid_topo = gridder_topo.grid(region=region, spacing=spacing, dims=["latitude", "longitude"])


# 4. Visualization
print("\n3. Generating Correlation Plots...")
fig = plt.figure(figsize=(15, 10))
fig.suptitle("Part 6: Gravity vs. Topography Correlation (Bushveld Complex)", fontsize=16)

# --- Map 1: Bouguer Gravity Anomaly ---
ax1 = fig.add_subplot(221)
ax1.set_title("A. Bouguer Gravity Anomaly (mGal)")
# Use pcolormesh with the gridded data.
# We use grid_grav.longitude and grid_grav.latitude now.
tmp1 = ax1.pcolormesh(grid_grav.longitude, grid_grav.latitude, grid_grav.scalars, cmap="RdBu_r", shading='auto')
fig.colorbar(tmp1, ax=ax1, label="mGal")
ax1.set_aspect('equal') # Important for maps to not look stretched
ax1.set_ylabel("Latitude")

# --- Map 2: Topography (from survey data) ---
ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1) # share axes for easy comparison
ax2.set_title("B. Topography (Survey Elevation)")
# Use a terrain colormap for elevation
tmp2 = ax2.pcolormesh(grid_topo.longitude, grid_topo.latitude, grid_topo.scalars, cmap="terrain", shading='auto')
fig.colorbar(tmp2, ax=ax2, label="Elevation (m)")
ax2.set_aspect('equal')
# ax2.set_yticklabels([]) # Optional: Hide Y labels if sharing

# --- Plot 3: Correlation Scatter Plot ---
ax3 = fig.add_subplot(212)
ax3.set_title(f"C. Correlation Analysis: Gravity vs. {elev_col.capitalize()}")
# Plot every 10th point to avoid overcrowding the plot
ax3.scatter(data[elev_col][::10], data.gravity_bouguer_mgal[::10], alpha=0.3, s=5, c='k', label="Station Data (Sampled)")

# Calculate Trend
# Remove any NaNs before polyfit to avoid errors
mask = ~np.isnan(data[elev_col]) & ~np.isnan(data.gravity_bouguer_mgal)
z = np.polyfit(data[elev_col][mask], data.gravity_bouguer_mgal[mask], 1)
p = np.poly1d(z)
corr_coeff = np.corrcoef(data[elev_col][mask], data.gravity_bouguer_mgal[mask])[0, 1]

x_trend = np.linspace(data[elev_col].min(), data[elev_col].max(), 100)
ax3.plot(x_trend, p(x_trend), "r--", linewidth=3,
         label=f"Trend: {z[0]:.3f} mGal/m\nCorrelation (r): {corr_coeff:.3f}")

ax3.set_xlabel("Elevation (m)", fontsize=12)
ax3.set_ylabel("Bouguer Anomaly (mGal)", fontsize=12)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
print("   Displaying final Part 6 plot...")
plt.show()

print("\n--- Part 6 Analysis Finished ---")
