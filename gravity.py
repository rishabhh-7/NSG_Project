"""
Gravity Anomaly Modeling & Geophysics Sandbox
-----------------------------------------------
A project for a beginner in near-surface geophysics to model
gravity anomalies, process data, and learn interpretation.

This script uses 'harmonica', 'verde', 'scipy', 'ensaio', 'pandas'.

You will need to install these libraries first:
pip install harmonica verde numpy matplotlib scipy ensaio pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import harmonica as hm
import verde as vd
from scipy.optimize import curve_fit
import ensaio
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 0. Define Constants and Survey Parameters ---
# ** NOTE: G_CONST is not used directly, as Harmonica has it built-in **
# ** G_CONST = 6.674e-11  # (m^3)/(kg*s^2) **
# ** MILI_GAL = 1e5       # Conversion factor from m/s^2 to mGal **
# ** REMOVED MILI_GAL CONVERSION AS HARMONICA ALREADY RETURNS MGAL **
KM_TO_M = 1e3        # Conversion factor from km to m

# Define a 10km x 10km survey area
AREA = (-5000, 5000, -5000, 5000)
SPACING = 250  # 250m station spacing

# Create a 2D grid of observation points 10m above the ground
(easting_grid, northing_grid) = vd.grid_coordinates(
    region=AREA, spacing=SPACING
)
upward_grid = np.full_like(easting_grid, 10.0) # 10m flying height

print(f"Created a {easting_grid.shape[0]} x {easting_grid.shape[1]} survey grid.")

# Harmonica functions require 1D arrays for observation points
easting_1d = easting_grid.ravel()
northing_1d = northing_grid.ravel()
upward_1d = upward_grid.ravel()
coordinates_1d = (easting_1d, northing_1d, upward_1d)

# --- 1 & 2. Forward Modeling: Complex Geometries ---
print("\n--- Starting Part 1 & 2: Forward Modeling ---")

# --- Model 1: Sphere (e.g., mineral deposit, salt dome) ---
# Density contrast in kg/m^3
density_contrast_sphere = 1000.0  # kg/m^3
radius_sphere = 800.0  # meters
center_sphere = (0, 0, -1500)  # (easting, northing, upward)

# Calculate mass from density and volume
volume_sphere = (4 / 3) * np.pi * (radius_sphere**3)
mass_sphere = density_contrast_sphere * volume_sphere

# Define the point mass location as three 1D arrays (easting, northing, upward)
point_easting = np.array([center_sphere[0]])
point_northing = np.array([center_sphere[1]])
point_upward = np.array([center_sphere[2]])
points_tuple = (point_easting, point_northing, point_upward)
masses_array = np.array([mass_sphere])

print("Modeling Sphere...")
# Calculate the gravity anomaly (g_z) from the point mass
# ** FIX: Harmonica returns mGal directly, no _si suffix, no conversion **
g_z_sphere = hm.point_gravity(
    coordinates=coordinates_1d,
    points=points_tuple,
    masses=masses_array,
    field="g_z",
)

# --- Model 2: Slab/Prism (e.g., buried bedrock, basin) ---
density_contrast_prism = 300.0  # kg/m^3
prism_bounds = [-2000, 2000, -3000, 3000, -1000, -500]  # [W, E, S, N, Bottom, Top]
prisms = np.array([prism_bounds])
densities_prism = np.array([density_contrast_prism])

print("Modeling Slab (Prism)...")
# ** FIX: Harmonica returns mGal directly **
g_z_prism = hm.prism_gravity(
    coordinates=coordinates_1d,
    prisms=prisms,
    density=densities_prism,
    field="g_z",
)

# --- Model 3: Horizontal Cylinder (e.g., buried channel, pipeline) ---
n_points_cylinder = 20
cylinder_easting = np.zeros(n_points_cylinder)
cylinder_northing = np.linspace(-4000, 4000, n_points_cylinder)
cylinder_upward = np.full_like(cylinder_easting, -500)
cylinder_points = (cylinder_easting, cylinder_northing, cylinder_upward)

density_contrast_cyl = 800.0
radius_cyl = 150.0
volume_cyl_segment = np.pi * radius_cyl**2 * (8000 / n_points_cylinder)
mass_cyl_segment = density_contrast_cyl * volume_cyl_segment
cylinder_masses = np.full(n_points_cylinder, mass_cyl_segment)

print("Modeling Cylinder...")
# ** FIX: Harmonica returns mGal directly **
g_z_cyl = hm.point_gravity(
    coordinates=coordinates_1d,
    points=cylinder_points,
    masses=cylinder_masses,
    field="g_z",
)

# --- Model 4: Fault Block (two adjacent prisms) ---
density_contrast_fault = 300.0  # 0.3 g/cm^3
prism_fault_block = [0, 4000, -4000, 4000, -1000, -500] # The "up" block
fault_prisms = np.array([prism_fault_block])
fault_densities = np.array([density_contrast_fault])

print("Modeling Fault...")
# ** FIX: Harmonica returns mGal directly **
g_z_fault = hm.prism_gravity(
    coordinates=coordinates_1d,
    prisms=fault_prisms,
    density=fault_densities,
    field="g_z",
)

# --- Plotting Part 1 & 2 ---
print("Plotting 2D maps for geometries...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
fig.suptitle("Part 1 & 2: Forward Modeling of Geometries", fontsize=16)

def plot_2d_map(ax, data, title):
    """Helper function to plot a 2D anomaly map."""
    ax.set_title(title)
    # Reshape the 1D data back to 2D for plotting
    data_grid = data.reshape(easting_grid.shape)
    # Calculate a dynamic, symmetrical color limit
    max_abs_val = np.max(np.abs(data))
    limit = max_abs_val * 0.9 # Use 90% of max val for good contrast
    if limit < 0.1: # ensure a minimum limit for very small anomalies
        limit = 0.1
        
    img = ax.pcolormesh(
        easting_grid/KM_TO_M, northing_grid/KM_TO_M, data_grid,
        cmap="RdBu_r", vmin=-limit, vmax=limit, shading='auto'
    )
    plt.colorbar(img, ax=ax, label="mGal", orientation="vertical", shrink=0.8)
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.set_aspect("equal")

plot_2d_map(axes[0, 0], g_z_sphere, "Sphere Anomaly")
plot_2d_map(axes[0, 1], g_z_prism, "Slab (Prism) Anomaly")
plot_2d_map(axes[1, 0], g_z_cyl, "Horizontal Cylinder Anomaly")
plot_2d_map(axes[1, 1], g_z_fault, "Fault Block Anomaly")

plt.show()

# --- 3. Implement Corrections & Processing (Regional-Residual Separation) ---
print("\n--- Starting Part 3: Regional-Residual Separation ---")

# 1. Create a "Regional" trend (a simple 2nd-order polynomial)
#    Scaled to be noticeable but not completely overwhelm the sphere
# ** FIX: Scaled coefficients down by 10x to make trend weaker and more realistic **
regional_trend = (
    (1e-7 * (easting_1d - 4000)**2 +
     5e-8 * (northing_1d + 3000)**2)
)
# 2. Create "Total Field" by adding regional to our target (sphere)
total_field = g_z_sphere + regional_trend

# 3. Use 'verde' to fit a polynomial trend to the total field
trend_fitter = vd.Trend(degree=2)
trend_fitter.fit((easting_1d, northing_1d), total_field)

# 4. Predict the regional trend everywhere
fitted_regional = trend_fitter.predict((easting_1d, northing_1d))

# 5. Calculate the "Residual" by subtracting the fitted trend
residual = total_field - fitted_regional

# --- Plotting Part 3 ---
print("Plotting Regional-Residual results...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
fig.suptitle("Part 3: Regional-Residual Separation", fontsize=16)

plot_2d_map(ax1, total_field, "1. Total Field (Target + Regional)")
plot_2d_map(ax2, fitted_regional, "2. Fitted Regional Trend")
plot_2d_map(ax3, residual, "3. Recovered Residual (Target)")

plt.show()


# --- 4. Introduce Inversion (Parameter Estimation) ---
print("\n--- Starting Part 4: Inversion ---")

# We will use a 1D profile from our sphere model
# A line along the Easting axis at Northing = 0
profile_mask = (northing_1d == 0) 
profile_coords_easting = easting_1d[profile_mask]
profile_true_signal = g_z_sphere[profile_mask]

# ** FIX 2: Using a realistic, VISIBLE noise level for the new signal size **
NOISE_STD_DEV = 0.75  # 0.2 mGal of noise
noise = np.random.normal(0, NOISE_STD_DEV, profile_true_signal.shape)
profile_noisy_data = profile_true_signal + noise

# Define the forward model function for the inverter
def model_for_inversion(easting_coords, log_mass, depth):
    """
    Calculates g_z for a point mass along an East-West profile.
    'easting_coords' is the 1D array of easting coordinates.
    'log_mass' and 'depth' are the parameters to be fit.
    """
    # Rebuild the full 1D coordinate tuple for harmonica
    n = np.zeros_like(easting_coords) # Profile is at northing = 0
    u = np.full_like(easting_coords, 10)    # at 10m height
    
    mass = 10**log_mass # Convert log_mass back to mass
    
    # Define point mass at (0, 0, -depth)
    point = (np.array([0]), np.array([0]), np.array([-depth]))
    mass_arr = np.array([mass])
    
    # ** FIX: Harmonica returns mGal, so NO conversion **
    g_z_mgal = hm.point_gravity(
        coordinates=(easting_coords, n, u),
        points=point,
        masses=mass_arr,
        field="g_z"
    )
    return g_z_mgal

# Initial guess for the inverter
p0 = [np.log10(mass_sphere * 0.5), 1000] # Guess 50% of mass, 1000m depth

print("Running inversion (scipy.optimize.curve_fit)...")
# Run the inversion
popt, _ = curve_fit(model_for_inversion, profile_coords_easting, profile_noisy_data, p0=p0)

# Get the results
fitted_log_mass, fitted_depth = popt
fitted_mass = 10**fitted_log_mass
fitted_signal = model_for_inversion(profile_coords_easting, fitted_log_mass, fitted_depth)

print("--- Inversion Results ---")
print(f"  True Mass: {mass_sphere:,.2e} kg | Fitted Mass: {fitted_mass:,.2e} kg")
print(f"  True Depth: {center_sphere[2] * -1:.1f} m | Fitted Depth: {fitted_depth:.1f} m")

# --- Plotting Part 4 ---
print("Plotting Inversion results...")
plt.figure(figsize=(10, 6))
plt.title("Part 4: Inversion of Noisy Data", fontsize=16)
plt.plot(profile_coords_easting/KM_TO_M, profile_true_signal, 'r-', linewidth=3, label="True Signal")
plt.plot(profile_coords_easting/KM_TO_M, profile_noisy_data, 'k.', markersize=8, alpha=0.6, label=f"Noisy 'Field' Data (std={NOISE_STD_DEV} mGal)")
plt.plot(profile_coords_easting/KM_TO_M, fitted_signal, 'g--', linewidth=4, label="Inverted Model Fit")
plt.xlabel("Easting (km)")
plt.ylabel("Gravity Anomaly (mGal)")
plt.legend()
plt.grid(True)
plt.show()


# --- 5 & 6. Real Data Application and Topography Correlation ---
print("\n--- Starting Part 5 & 6: Real Data Analysis (Bushveld) ---")

# 1. Fetch Gravity Data (Bushveld)
print("\n1. Fetching Gravity Data (Bushveld Complex)...")
try:
    fname = ensaio.fetch_bushveld_gravity(version=1)
    print(f"   Data downloaded to: {fname}")
    
    data = pd.read_csv(fname)
    print("   Gravity data loaded successfully into DataFrame.")
    print("   Data columns found:", data.columns.tolist())

except Exception as e:
    print(f"   CRITICAL ERROR: Could not load gravity data: {e}")
    print("   Skipping Part 5 & 6.")
    data = None 

if data is not None:
    # 2. Identify Elevation Column
    elev_col = None
    possible_names = ['height_sea_level_m', 'height', 'elevation', 'h']
    for col in possible_names:
        if col in data.columns:
            elev_col = col
            break

    if elev_col is None:
        print("\nCRITICAL ERROR: Could not find an elevation column.")
    else:
        print(f"   -> SUCCESS: Found topography data in column '{elev_col}'")

        # 3. Gridding (for visualization)
        print("\n2. Gridding Data for visualization...")
        region = vd.get_region((data.longitude, data.latitude))
        spacing = 1/60 # ~1 minute arc spacing (approx 2km)

        print("   Gridding Gravity...")
        gridder_grav = vd.Spline()
        gridder_grav.fit((data.longitude, data.latitude), data.gravity_bouguer_mgal)
        grid_grav = gridder_grav.grid(region=region, spacing=spacing, dims=["latitude", "longitude"])

        print("   Gridding Topography...")
        gridder_topo = vd.Spline()
        gridder_topo.fit((data.longitude, data.latitude), data[elev_col])
        grid_topo = gridder_topo.grid(region=region, spacing=spacing, dims=["latitude", "longitude"])

        # 4. Visualization
        print("\n3. Generating Correlation Plots...")
        
        # --- Plot 1: Bushveld Gravity Anomaly (Part 5) ---
        print("   Plotting Part 5 (Real Data Map)...")
        plt.figure(figsize=(8, 7))
        ax = plt.axes()
        ax.set_title("Part 5: Real Data - Bushveld Gravity Anomaly")
        tmp = grid_grav.scalars.plot.pcolormesh(
            ax=ax, cmap="RdBu_r", add_colorbar=False, shading='auto'
        )
        plt.colorbar(tmp, ax=ax, label="Bouguer Anomaly (mGal)")
        ax.plot(data.longitude, data.latitude, 'k.', markersize=0.5, alpha=0.1)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Correlation Analysis (Part 6) ---
        print("   Plotting Part 6 (Correlation Analysis)...")
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Part 6: Gravity vs. Topography Correlation (Bushveld Complex)", fontsize=16)

        # Map 1: Bouguer Gravity Anomaly
        ax1 = fig.add_subplot(221)
        ax1.set_title("A. Bouguer Gravity Anomaly (mGal)")
        tmp1 = grid_grav.scalars.plot.pcolormesh(
            ax=ax1, cmap="RdBu_r", add_colorbar=False, shading='auto'
        )
        fig.colorbar(tmp1, ax=ax1, label="mGal")
        ax1.set_aspect('equal')
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

        # Map 2: Topography (from survey data)
        ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1) # share axes
        ax2.set_title("B. Topography (Survey Elevation)")
        tmp2 = grid_topo.scalars.plot.pcolormesh(
            ax=ax2, cmap="terrain", add_colorbar=False, shading='auto'
        )
        fig.colorbar(tmp2, ax=ax2, label="Elevation (m)")
        ax2.set_aspect('equal')
        ax2.set_xlabel("Longitude")

        # Plot 3: Correlation Scatter Plot
        ax3 = fig.add_subplot(212)
        ax3.set_title(f"C. Correlation Analysis: Gravity vs. {elev_col.capitalize()}")
        ax3.scatter(data[elev_col][::10], data.gravity_bouguer_mgal[::10], alpha=0.3, s=5, c='k', label="Station Data (Sampled)")

        # Calculate Trend
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

print("\n--- Full script finished. ---")