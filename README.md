# Solar Differential Rotation Pipeline

An automated pipeline that tracks sunspot positions across multiple days of
observation, converts raw helioprojective Cartesian coordinates into
heliographic latitude/longitude, and fits the classic solar differential
rotation law to derive how rotation rate varies with latitude.

## Background

The Sun does not rotate as a rigid body, its equator rotates faster than
its higher latitudes. This is described by the differential rotation law:

```
Ω_sid(λ) = A + B·sin²(λ)
```

where `Ω_sid` is the sidereal rotation rate (degrees/day, relative to the
stars rather than Earth) and `λ` is heliographic latitude. This project
recovers `A` and `B` empirically from tracked sunspot positions, the same
approach used in classical solar rotation studies (e.g. Snodgrass & Ulrich).

## What the pipeline does

1. **Ingest**: reads multiple `sunspot_*.csv` files, each containing a
   tracked sunspot's helioprojective Cartesian coordinates (`theta_x`,
   `theta_y`) across consecutive days.
2. **Coordinate transform**: converts helioprojective Cartesian
   coordinates to heliographic (Stonyhurst) latitude/longitude using the
   sub-Earth point angles (B0, L0) and the apparent solar radius.
3. **Longitude unwrapping**: corrects for artificial jumps when tracked
   longitude crosses the ±180° boundary.
4. **Per-feature rotation fit**: linear regression of longitude vs. time
   for each sunspot to recover its synodic rotation rate, then converts to
   sidereal rotation by adding Earth's orbital angular velocity.
5. **Differential rotation fit**: linear regression of sidereal rotation
   rate against sin²(latitude) across all tracked features to recover the
   `A` and `B` coefficients of the rotation law.
6. **Visualisation**: produces three plots: rotation rate vs. latitude
   with the fitted curve overlaid, rotation period vs. latitude, and the
   linearised fit used to extract `A` and `B`.

## Input data format

Each `sunspot_*.csv` file represents one tracked sunspot, observed once per
day, with the following columns:

| Column            | Description                                      |
|-------------------|---------------------------------------------------|
| `t_days`          | Days elapsed since the first observation          |
| `date`            | Observation date (YYYY/MM/DD), for reference only |
| `theta_x_arcsec`  | Helioprojective Cartesian x-coordinate (arcsec)    |
| `theta_y_arcsec`  | Helioprojective Cartesian y-coordinate (arcsec)    |

## Usage

```bash
pip install pandas numpy scipy matplotlib
python analysis.py
```

The script expects `sunspot_*.csv` files in the working directory. It
prints intermediate tables (latitude summary, per-feature rotation fits,
final differential rotation parameters) to the console and saves three
PNG figures (`rotation_profile.png`, `rotation_period.png`,
`linearised_fit.png`).

## Known limitations

- The mean-latitude summary table assumes each tracked feature has exactly
  5 daily observations. Features tracked for a different number of days
  would need that section generalised.
- B0 and L0 (the sub-Earth point angles) are treated as constant across
  the observation window, which is a reasonable approximation for short
  tracking periods but introduces small systematic error over longer ones.
- Rotation rate uncertainty is propagated from the linear regression
  standard error only; it does not account for measurement error in the
  original pixel/arcsecond coordinates.

## Background reading

- Thompson, W. T. (2006), *Coordinate systems for solar image data*,
  A&A 449, 791–803 — defines the helioprojective/heliographic coordinate
  transforms used here.
- Snodgrass, H. B. & Ulrich, R. K. (1990), *Rotation of Doppler features
  in the solar photosphere*, ApJ 351, 309 — reference differential
  rotation coefficients for comparison.
