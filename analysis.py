"""
Solar Differential Rotation Analysis Pipeline
===============================================

Tracks sunspot positions across multiple days of helioprojective Cartesian
observations, converts them to heliographic (Stonyhurst) latitude/longitude,
fits each sunspot's synodic rotation rate, converts to sidereal rotation,
and fits the classic differential rotation law:

    Omega_sid(lambda) = A + B * sin^2(lambda)

where lambda is heliographic latitude.

Input:
    sunspot_*.csv files in the working directory, each with columns:
        t_days            - days elapsed since first observation of that feature
        date              - observation date (YYYY/MM/DD), for reference only
        theta_x_arcsec    - helioprojective Cartesian x-coordinate (arcsec)
        theta_y_arcsec    - helioprojective Cartesian y-coordinate (arcsec)

Output:
    Console tables (latitude summary, per-feature rotation fits, final
    differential rotation parameters) and three matplotlib figures.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ======================================
# READ ALL SUNSPOT FILES
# ======================================

files = sorted(glob.glob("data/sunspot_*.csv"))
datasets = {}

for file in files:
    name = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)
    datasets[name] = df

# ======================================
# SOLAR PARAMETERS
# ======================================

# Apparent solar radius as seen from Earth, in arcseconds (approx. 1 AU distance).
# Used to normalise helioprojective x/y into dimensionless solar-disk coordinates.
R_sun = 952.0

# Heliographic latitude of the sub-Earth point (B0 angle) for the observation
# period, in degrees. B0 varies slowly through the year as Earth's orbital
# plane is tilted ~7.25 deg relative to the Sun's equator; here it is treated
# as constant over the short observation window.
B0_deg = -4.0
B0 = np.radians(B0_deg)

# Heliographic longitude of the sub-Earth point (L0 angle) at the reference
# epoch, in degrees. Set to 0 as the longitude origin for this analysis.
L0_deg = 0.0
L0 = np.radians(L0_deg)

# Earth's mean orbital angular velocity, in degrees/day (360 deg / 365.25 days).
# Used to convert a synodic (Earth-relative) rotation rate into a sidereal
# (star-fixed) rotation rate: Omega_sid = Omega_syn + Omega_earth.
OMEGA_EARTH = 0.986


# ======================================
# CONVERSION FUNCTION
# ======================================


def helioprojective_to_heliographic(theta_x, theta_y):
    """
    Convert helioprojective Cartesian coordinates to heliographic
    (Stonyhurst) latitude and longitude.

    Parameters
    ----------
    theta_x, theta_y : float
        Helioprojective Cartesian coordinates of the feature, in arcseconds,
        as measured from Earth.

    Returns
    -------
    lat, lon : float
        Heliographic latitude and longitude, in degrees. Returns
        (np.nan, np.nan) if the point falls outside the visible solar disk.

    Notes
    -----
    x and y are first normalised by the apparent solar radius (R_sun) to put
    them in disk-fraction units, then projected onto the solar sphere using
    the sub-Earth point angles B0 and L0. Points with x^2 + y^2 >= 1 lie off
    the visible disk and cannot be converted.
    """
    x = theta_x / R_sun
    y = theta_y / R_sun

    rho2 = x**2 + y**2

    if rho2 >= 1:
        return np.nan, np.nan

    z = np.sqrt(1 - rho2)

    sin_lat = y * np.cos(B0) + z * np.sin(B0)
    lat = np.arcsin(sin_lat)

    numerator = x
    denominator = z * np.cos(B0) - y * np.sin(B0)

    lon = np.arctan2(numerator, denominator) + L0

    return np.degrees(lat), np.degrees(lon)


# ======================================
# LONGITUDE UNWRAP FUNCTION
# ======================================


def unwrap_longitudes(lons):
    """Prevents artificial statistical jumps if longitudes wrap around 180/-180 boundaries."""
    unwrapped = np.array(lons)
    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]
        if diff > 180:
            unwrapped[i:] -= 360
        elif diff < -180:
            unwrapped[i:] += 360
    return unwrapped


# ======================================
# APPLY CONVERSION & CLEAN JUMPS
# ======================================

for name, df in datasets.items():
    lats = []
    lons = []

    for _, row in df.iterrows():
        lat, lon = helioprojective_to_heliographic(
            row["theta_x_arcsec"], row["theta_y_arcsec"]
        )
        lats.append(lat)
        lons.append(lon)

    df["Latitude (deg)"] = lats
    df["Longitude (deg)"] = unwrap_longitudes(lons)

# ======================================
# MEAN LATITUDE TABLE
# ======================================

# NOTE: assumes every sunspot_*.csv file has exactly 5 daily observations
# (lambda_1 .. lambda_5). If a feature is tracked for a different number of
# days, this section needs to be generalised (e.g. build the lambda_i columns
# dynamically based on len(df) instead of hardcoding 5).
summary_rows = []

for name, df in datasets.items():
    latitudes = df["Latitude (deg)"].values

    mean_lat = np.mean(latitudes)
    abs_mean_lat = abs(mean_lat)

    row = {
        "Feature": name,
        "λ1": latitudes[0],
        "λ2": latitudes[1],
        "λ3": latitudes[2],
        "λ4": latitudes[3],
        "λ5": latitudes[4],
        "Mean Latitude λ̄": mean_lat,
        "|λ̄|": abs_mean_lat,
    }

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print("\nMEAN LATITUDE TABLE")
print(summary_df.round(3))

# ======================================
# SYNODIC ROTATION ANALYSIS
# ======================================

rotation_results = []

for name, df in datasets.items():
    t = df["t_days"].values
    L = df["Longitude (deg)"].values

    slope, intercept, r_value, p_value, std_err = linregress(t, L)

    L_fit = intercept + slope * t
    residuals = L - L_fit

    rotation_results.append(
        {
            "Feature": name,
            "Slope m (deg/day)": slope,
            "σm (deg/day)": std_err,
            "Ωsyn (deg/day)": abs(slope),
            "σΩ (deg/day)": std_err,
            "R²": r_value**2,
        }
    )

    print(f"\n{name} Rotation Fit")
    print(
        pd.DataFrame(
            {"t": t, "L": L, "L_fit": L_fit, "residual": residuals}
        ).round(3)
    )

rotation_summary = pd.DataFrame(rotation_results)

# ======================================
# SIDEREAL + PERIOD
# ======================================

rotation_summary["Ωsid (deg/day)"] = (
    rotation_summary["Ωsyn (deg/day)"] + OMEGA_EARTH
)
rotation_summary["Psid (days)"] = 360 / rotation_summary["Ωsid (deg/day)"]

# ======================================
# UNCERTAINTY IN ROTATION PERIOD
# ======================================

# Propagates the standard error on Omega_sid through P = 360 / Omega_sid via
# standard first-order error propagation: sigma_P = |dP/dOmega| * sigma_Omega.
rotation_summary["σP (days)"] = (
    360 / (rotation_summary["Ωsid (deg/day)"] ** 2)
) * rotation_summary["σΩ (deg/day)"]

print("\nFINAL ROTATION TABLE")
print(rotation_summary.round(3))

# ======================================
# MERGE WITH LATITUDE DATA
# ======================================

final_rows = []

for i, (name, df) in enumerate(datasets.items()):
    mean_lat = summary_df.loc[i, "|λ̄|"]

    final_rows.append(
        {
            "Feature": name,
            "|λ|": mean_lat,
            "Ωsid": rotation_summary.loc[i, "Ωsid (deg/day)"],
            "Psid": rotation_summary.loc[i, "Psid (days)"],
        }
    )

final_df = pd.DataFrame(final_rows)
final_df["sin²λ"] = np.sin(np.radians(final_df["|λ|"])) ** 2

print("\nFINAL ANALYSIS TABLE")
print(final_df.round(3))

# ======================================
# LINEARISED DIFFERENTIAL ROTATION PROFILE
# ======================================

x = final_df["sin²λ"]
y = final_df["Ωsid"]

slope, intercept, r, _, _ = linregress(x, y)
A = intercept
B = slope

print("\nDIFFERENTIAL ROTATION FIT")
print(f"A = {A:.3f} deg/day")
print(f"B = {B:.3f} deg/day")
print(f"R² = {r**2:.3f}")


# ======================================
# PLOTTING HELPERS
# ======================================


def label_points(ax_x, ax_y, labels, offset=0.3):
    """Annotate each scatter point with its feature name."""
    for xi, yi, label in zip(ax_x, ax_y, labels):
        plt.text(xi + offset, yi, label, fontsize=9)


# ======================================
# GRAPH 1: Ωsid vs |λ| WITH OVERLAID PROFILE
# ======================================

plt.figure(figsize=(7, 5))
plt.scatter(final_df["|λ|"], final_df["Ωsid"], color="darkred", label="Observed")

# Smooth theoretical curve generated from the fitted A and B parameters.
lat_grid = np.linspace(0, max(final_df["|λ|"]) + 5, 100)
omega_curve = A + B * (np.sin(np.radians(lat_grid)) ** 2)
plt.plot(
    lat_grid,
    omega_curve,
    color="blue",
    linestyle="--",
    label=r"Fit: $\Omega = A + B\sin^2\lambda$",
)

label_points(final_df["|λ|"], final_df["Ωsid"], final_df["Feature"])

plt.xlabel(r"Absolute Mean Latitude $|\bar{\lambda}|$ (deg)")
plt.ylabel(r"Sidereal Rotation Rate $\Omega_{sid}$ (deg/day)")
plt.title("Solar Differential Rotation Profile")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("rotation_profile.png", dpi=150)
plt.show()

# ======================================
# GRAPH 2: PERIOD vs |λ| (SCATTER ONLY)
# ======================================

plt.figure(figsize=(7, 5))
plt.scatter(
    final_df["|λ|"], final_df["Psid"], color="darkblue", label="Calculated Spots"
)

label_points(final_df["|λ|"], final_df["Psid"], final_df["Feature"])

plt.xlabel(r"Absolute Mean Latitude $|\bar{\lambda}|$ (deg)")
plt.ylabel("Sidereal Period $P_{sid}$ (days)")
plt.title("Rotation Period vs Latitude")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("rotation_period.png", dpi=150)
plt.show()

# ======================================
# GRAPH 3: LINEARISED LAW FIT
# ======================================

x_fit = np.linspace(0, max(x) * 1.1, 100)
y_fit = A + B * x_fit

plt.figure(figsize=(7, 5))
plt.scatter(x, y, color="black", zorder=3, label="Data Points")
plt.plot(
    x_fit,
    y_fit,
    color="red",
    label=f"Linear Fit ($R^2$ = {r**2:.3f})\n$\\Omega$ = {A:.2f} + ({B:.2f})$x$",
)

label_points(x, y, final_df["Feature"], offset=0.003)

plt.xlabel(r"$\sin^2|\lambda|$")
plt.ylabel(r"$\Omega_{sid}$ (deg/day)")
plt.title("Linearised Solar Differential Rotation Fit")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("linearised_fit.png", dpi=150)
plt.show()