"""
Gravity Anomaly Modeling & Geophysics Sandbox
-----------------------------------------------
A project for a beginner in near-surface geophysics to model
gravity anomalies, process data, and learn interpretation.

This script uses the 'harmonica', 'verde', and 'scipy' libraries.

You will need to install these libraries first:
pip install harmonica verde numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
import harmonica as hm
import verde as vd
from scipy.optimize import curve_fit  # For Part 4: Inversion

# --- 0. Define Constants and Survey Parameters ---

# Define the survey area
AREA = [-5000, 5000, -5000, 5000]  # [west, east, south, north] in meters
SPACING = 250  # meters
HEIGHT = 10.0  # Survey height in meters (positive upward)

# Define gravity constants
MGAL_TO_SI = 1e-5  # Conversion factor from mGal to m/s^2

# Create the 2D survey grid coordinates (easting and northing)
(easting_grid, northing_grid) = vd.grid_coordinates(
    region=AREA,
    spacing=SPACING
)
upward_grid = np.full_like(easting_grid, HEIGHT)
shape = easting_grid.shape
print(f"Created a {shape[0]} x {shape[1]} survey grid.")

# Ravel the 2D grids into 1D arrays for the harmonica functions
easting_1d = easting_grid.ravel()
northing_1d = northing_grid.ravel()
upward_1d = upward_grid.ravel()

# Our coordinates tuple needs all three 1D components
coordinates_1d = (easting_1d, northing_1d, upward_1d)


# --- 1 & 2. Forward Modeling: Complex Geometries (Sphere, Slab, Cylinder, Fault) ---

print("Starting Part 1 & 2: Forward Modeling of Geometries...")

# --- Model 1: Sphere (as Point Mass) ---
density_contrast_sphere = 1000  # kg/m^3
radius_sphere = 800           # meters
center_sphere = (0, 0, -1500)   # (easting, northing, upward)

volume_sphere = (4/3) * np.pi * (radius_sphere ** 3)
mass_sphere = density_contrast_sphere * volume_sphere

point_easting = np.array([center_sphere[0]])
point_northing = np.array([center_sphere[1]])
point_upward = np.array([center_sphere[2]])
points_tuple = (point_easting, point_northing, point_upward)
masses_array = np.array([mass_sphere])

g_z_sphere_mgal = hm.point_gravity(
    coordinates_1d,
    points=points_tuple,
    masses=masses_array,
    field="g_z"
) / MGAL_TO_SI

# --- Model 2: Slab (Prism) ---
density_contrast_prism = 1000  # kg/m^3
prism_bounds_slab = [
    -2000, 2000,  # west, east
    -2000, 2000,  # south, north
    -1750, -1250  # bottom, top (negative z)
]
prisms_slab = np.array([prism_bounds_slab])
densities_slab = np.array([density_contrast_prism])

g_z_prism_mgal = hm.prism_gravity(
    coordinates_1d,
    prisms=prisms_slab,
    density=densities_slab,
    field="g_z"
) / MGAL_TO_SI

# --- Model 3: Horizontal Cylinder (as a long, thin prism) ---
density_contrast_cyl = 1000  # kg/m^3
# Long in N-S direction, thin in E-W direction
prism_bounds_cyl = [
    -500, 500,    # west, east
    -4000, 4000,  # south, north
    -1750, -1250  # bottom, top
]
prisms_cyl = np.array([prism_bounds_cyl])
densities_cyl = np.array([density_contrast_cyl])

g_z_cyl_mgal = hm.prism_gravity(
    coordinates_1d,
    prisms=prisms_cyl,
    density=densities_cyl,
    field="g_z"
) / MGAL_TO_SI

# --- Model 4: Fault Block (a "half-slab" creating an asymmetric anomaly) ---
density_contrast_fault = 500  # kg/m^3
# This prism extends from x=0 to "infinity" (or just the edge of our map)
prism_bounds_fault = [
    0, 5000,       # west, east
    -5000, 5000,   # south, north
    -2000, -1000   # bottom, top
]
prisms_fault = np.array([prism_bounds_fault])
densities_fault = np.array([density_contrast_fault])

g_z_fault_mgal = hm.prism_gravity(
    coordinates_1d,
    prisms=prisms_fault,
    density=densities_fault,
    field="g_z"
) / MGAL_TO_SI

# --- Plotting for Part 1 & 2 ---
print("Plotting 2D maps for geometries...")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12), sharex=True, sharey=True)
fig.suptitle("Part 1 & 2: 2D Gravity Anomalies (g_z)", fontsize=16)

# Plot Sphere
ax1.set_title("Symmetric Anomaly (Sphere)")
tmp = ax1.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    g_z_sphere_mgal.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax1, label="mGal")
ax1.set_ylabel("Northing (km)")
ax1.set_aspect("equal")

# Plot Slab
ax2.set_title("Symmetric Anomaly (Slab)")
tmp = ax2.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    g_z_prism_mgal.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax2, label="mGal")
ax2.set_aspect("equal")

# Plot Cylinder
ax3.set_title("Linear Anomaly (H. Cylinder)")
tmp = ax3.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    g_z_cyl_mgal.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax3, label="mGal")
ax3.set_xlabel("Easting (km)")
ax3.set_ylabel("Northing (km)")
ax3.set_aspect("equal")

# Plot Fault
ax4.set_title("Asymmetric Anomaly (Fault Block)")
tmp = ax4.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    g_z_fault_mgal.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax4, label="mGal")
ax4.set_xlabel("Easting (km)")
ax4.set_aspect("equal")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 3. Implement Corrections & Processing (Regional-Residual Separation) ---
print("Starting Part 3: Regional-Residual Separation...")

# 3a. Create a "regional" trend (a 2nd-order polynomial)
# This simulates a deep, large-scale feature we want to remove.
trend_coeffs = [1, -0.5, 0.8]  # Fake coefficients
regional_trend = trend_coeffs[0] * (easting_1d / 1000) \
               + trend_coeffs[1] * (northing_1d / 1000) \
               + trend_coeffs[2] * (easting_1d / 1000)**2

# 3b. Add the regional trend to our sphere anomaly to create "Total" data
total_anomaly = g_z_sphere_mgal + regional_trend

# 3c. Use 'verde' to fit and remove the trend
# We fit a 2nd-order polynomial (degree=2)
trend_estimator = vd.Trend(degree=2)
trend_estimator.fit(coordinates=(easting_1d, northing_1d), data=total_anomaly)

# Predict the regional trend everywhere
fitted_regional = trend_estimator.predict(coordinates=(easting_1d, northing_1d))

# 3d. The "residual" is the total minus the fitted trend
residual_anomaly = total_anomaly - fitted_regional

# --- Plotting for Part 3 ---
print("Plotting Regional-Residual results...")
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
fig.suptitle("Part 3: Regional-Residual Separation", fontsize=16)

# Plot Total Anomaly
ax1.set_title("Total Anomaly (Target + Regional)")
tmp = ax1.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    total_anomaly.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax1, label="mGal")
ax1.set_ylabel("Northing (km)")
ax1.set_xlabel("Easting (km)")

# Plot Fitted Regional
ax2.set_title("Fitted Regional Trend")
tmp = ax2.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    fitted_regional.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax2, label="mGal")
ax2.set_xlabel("Easting (km)")

# Plot Residual Anomaly
ax3.set_title("Residual Anomaly (Target Recovered)")
tmp = ax3.pcolormesh(
    easting_grid / 1000, northing_grid / 1000,
    residual_anomaly.reshape(shape), cmap="viridis", shading="auto"
)
fig.colorbar(tmp, ax=ax3, label="mGal")
ax3.set_xlabel("Easting (km)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# --- 4. Introduce Inversion (Parameter Estimation) ---
print("Starting Part 4: Inversion...")

# 4a. Get the 1D profile data from our original sphere model
center_row_index = shape[0] // 2
profile_easting = easting_grid[center_row_index, :] # in meters
profile_data = g_z_sphere_mgal.reshape(shape)[center_row_index, :]

# Add some random noise to make it more realistic
noise = np.random.normal(0, 0.05, profile_data.shape) # 0.05 mGal std dev
profile_data_noisy = profile_data + noise

# 4b. Define the forward modeling function for the inversion
# This function must match the signature for scipy.optimize.curve_fit:
# f(x, param1, param2, ...)
def model_for_inversion(easting, mass, depth_meters):
    """
    Calculates the g_z anomaly for a point mass.
    'easting' is our x-data (a 1D array).
    'mass' and 'depth_meters' are the parameters we want to find.
    """
    # Build the 1D observation coordinates for this profile
    profile_coords_1d = (
        easting,
        np.zeros_like(easting), # Northing = 0
        np.full_like(easting, HEIGHT) # Constant height
    )

    # Build the 1D source point coordinates
    point_coords = (
        np.array([0]), # Centered at easting=0
        np.array([0]), # Centered at northing=0
        np.array([-depth_meters]) # Use the 'depth' parameter
    )
    
    # Build the 1D mass array
    mass_array = np.array([mass])

    # Calculate the anomaly
    anomaly = hm.point_gravity(
        profile_coords_1d,
        points=point_coords,
        masses=mass_array,
        field="g_z"
    ) / MGAL_TO_SI
    
    return anomaly

# 4c. Run the inversion!
# We provide initial guesses (p0) for [mass, depth_meters]
initial_guess = [1e12, 1000] # Guess 1e12 kg and 1000m depth
# We set bounds to keep the inversion stable
param_bounds = ([1e10, 500], [1e15, 3000]) # (min_mass, min_depth), (max_mass, max_depth)

# curve_fit finds the parameters that make model_for_inversion()
# best fit our noisy data.
popt, pcov = curve_fit(
    model_for_inversion,
    xdata=profile_easting,
    ydata=profile_data_noisy,
    p0=initial_guess,
    bounds=param_bounds
)

# 4d. Get the results
estimated_mass, estimated_depth = popt
print("\n--- Inversion Results ---")
print(f"True Mass:     {mass_sphere:.2e} kg")
print(f"Estimated Mass: {estimated_mass:.2e} kg")
print(f"True Depth:     {-center_sphere[2]} m")
print(f"Estimated Depth: {estimated_depth:.2f} m")

# --- Plotting for Part 4 ---
print("Plotting Inversion results...")
plt.figure(figsize=(10, 6))
plt.plot(profile_easting / 1000, profile_data_noisy, 'k.', label='Noisy "Field" Data')
plt.plot(profile_easting / 1000, profile_data, 'r-', label='Original True Signal', lw=3)

# Plot the best-fit model from our inversion
fitted_profile = model_for_inversion(profile_easting, estimated_mass, estimated_depth)
plt.plot(profile_easting / 1000, fitted_profile, 'g--', label='Inverted Model Fit', lw=3)

plt.title("Part 4: Inversion of 1D Profile")
plt.xlabel("Easting (km)")
plt.ylabel("Gravity Anomaly (mGal)")
plt.legend()
plt.grid(True)
plt.show()


# --- 5. Use Real Gravity Datasets ---
print("Starting Part 5: Loading Real Data (Bushveld)...")

# 5a. Fetch the sample data
# This downloads the data if you don't have it, or loads it from cache
try:
    data = hm.datasets.fetch_bushveld_gravity()

    # 5b. Grid the scattered data using Verde
    # We'll use a bilinear interpolator to put the data onto a regular grid
    gridder = vd.Bilinear()
    gridder.fit((data.easting, data.northing), data.gravity_mgal)
    
    # Create a grid with 2km spacing
    grid = gridder.grid(
        region=vd.get_region((data.easting, data.northing)),
        spacing=2000,
        dims=["northing", "easting"], # Note: dims are (y, x)
    )

    # --- Plotting for Part 5 ---
    print("Plotting real data...")
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_title("Part 5: Real Data - Bushveld Gravity Anomaly")
    
    # Plot the gridded data
    tmp = grid.data.plot.pcolormesh(
        ax=ax,
        cmap="viridis",
        x="easting",
        y="northing",
        add_colorbar=False
    )
    plt.colorbar(tmp, ax=ax, label="Bouguer Anomaly (mGal)")
    
    # Plot the original scatter points
    ax.plot(data.easting, data.northing, 'k.', markersize=0.5, alpha=0.2)
    
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nCould not load or plot real dataset. This may be due to a network issue.")
    print(f"Error: {e}")


print("Full script finished.")