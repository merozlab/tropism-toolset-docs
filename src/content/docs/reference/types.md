---
title: Type Definitions
description: Type aliases and return types used throughout the Tropism Toolset.
---

Semantic type aliases for better code readability, type safety, and IDE support.

## Core Geometric Types

Single-frame spatial measurements along a centerline.

### `AngleArray`

Array of angles at each segment.

**Type:**
```python
pd.Series
```

**Name:** "angles (rad)"

**Units:** radians

**Usage:**
```python
from tropism_toolset import get_angles

angles: pd.Series = get_angles(frame_data)
```

---

### `ArclengthArray`

Array of cumulative arc lengths from start.

**Type:**
```python
pd.Series
```

**Name:** "arclength (unit)"

**Units:** meters or input units

**Usage:**
```python
from tropism_toolset import get_arclengths

arclengths: pd.Series = get_arclengths(frame_data)
```

---

### `TimeSeriesAngles`

Array of angle arrays, one per frame (variable-length).

**Type:**
```python
np.ndarray[object]
```

**Shape:** `(n_frames,)` - each element is an `pd.Series` of angles

**Usage:**
```python
from tropism_toolset import get_angles_over_time

angles: np.ndarray = get_angles_over_time(data)
```

---

### `TimeSeriesArclengths`

DataFrame containing arclengths for all frames.

**Type:**
```python
pd.DataFrame
```

**Usage:**
```python
from tropism_toolset import get_arclengths_over_time

arclengths: pd.DataFrame = get_arclengths_over_time(data)
```

---

### `TimeArray`

Array of frame numbers or time indices.

**Type:**
```python
pd.Series
```

**Units:** frame numbers (dimensionless integers) or seconds

---

### `TimeSeriesValues`

Generic time series of scalar values (one per frame).

**Type:**
```python
pd.Series
```

**Name:** "variable (unit)"

**Usage:**
```python
from tropism_toolset import get_lengths_from_centerlines

length_data: pd.DataFrame = get_lengths_from_centerlines(data)
# length_data has columns for time and length
```

---

### `PolyCoeffs`

Polynomial coefficients for linear fits.

**Type:**
```python
np.ndarray
```

**Shape:** `(2,)` - [slope, intercept]

---

## Complex Return Types

### `LinearFitResult`

Return type for linear fitting.

**Type:**
```python
tuple[float, float, np.ndarray]
```

**Structure:** `(slope, r_squared, coeffs)`

**Usage:**
```python
from tropism_toolset.fitting import fit_linear

result = fit_linear(times, values)
slope, r2, coeffs = result
```

---

### `PiecewiseFitResult`

Return type for piecewise linear fitting.

**Type:**
```python
tuple[float, float, float, tuple[np.ndarray, np.ndarray]]
```

**Structure:** `(slope1, slope2, breakpoint, (p1_coeffs, p2_coeffs))`

**Usage:**
```python
from tropism_toolset.fitting import fit_piecewise_linear_continuous

result = fit_piecewise_linear_continuous(times, values)
slope1, slope2, bp, (p1, p2) = result
```

---

### `SaturatingExponentialResult`

Return type for saturating exponential fit.

**Type:**
```python
tuple[float, float, float, float]
```

**Structure:** `(y_inf, y_0, tau, r_squared)`

**Usage:**
```python
from tropism_toolset.fitting import fit_saturating_exponential

result = fit_saturating_exponential(times, values)
y_inf, y_0, tau, r2 = result
```

---

### `LcFitResult`

Return type for convergence length (Lc) fit.

**Type:**
```python
tuple[float, float, float, float, float]
```

**Structure:** `(x0, Bl, A, Lc, r_squared)`

**Usage:**
```python
from tropism_toolset.fitting import fit_Lc

result = fit_Lc(s_theta_data)
x0, Bl, A, Lc, r2 = result
```

---

### `MaskStabilityResult`

Return type for mask stability analysis.

**Type:**
```python
tuple[pd.Series, int | None, float | None]
```

**Structure:** `(stability_scores, steady_start_frame, median_mask_area)`

**Usage:**
```python
from tropism_toolset.fitting import find_steady_state_from_masks

result = find_steady_state_from_masks(mask_dir, px_per_m)
scores, Tc, area = result
```

---

## Dimension Notation

Throughout the documentation:
- **n**: Number of coordinate points in a single centerline
- **n_frames**: Number of time frames
- **Variable-length**: Arrays where length varies per frame

## Type Safety

These type definitions correspond to the actual return types in the Python code, which heavily utilizes `pandas.DataFrame` and `pandas.Series` for unit-aware data handling.

## See Also

- [Geometric Calculations API](/reference/geometric-calculations/) - Functions returning geometric types
- [Fitting API](/reference/fitting/) - Functions returning fit result types
- [Constants API](/reference/constants/) - Functions using these types
