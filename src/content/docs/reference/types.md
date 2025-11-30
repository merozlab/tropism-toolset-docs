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
NDArray[np.float64]
```

**Shape:** `(n-1,)` where n is number of coordinate points

**Units:** radians

**Usage:**
```python
from constants import get_angles

angles: AngleArray = get_angles(frame_data)
```

---

### `ArclengthArray`

Array of cumulative arc lengths from start.

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n-1,)` where n is number of coordinate points

**Units:** meters

**Usage:**
```python
from constants import get_arclengths

arclengths: ArclengthArray = get_arclengths(frame_data)
```

---

### `CurvatureArray`

Array of curvature values (κ = dθ/ds).

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n-2,)` where n is number of coordinate points

**Units:** m⁻¹ (inverse meters) or rad/m

**Usage:**
```python
from constants.geometric_calculations import calculate_curvature

curvatures: CurvatureArray = calculate_curvature(frame_data)
```

---

### `CoordinateArray`

Array of x,y coordinate pairs or single coordinate dimension.

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n, 2)` or `(n,)` depending on context

**Units:** meters or pixels

**Usage:**
```python
import numpy as np

x_coords: CoordinateArray = frame_data['x'].values
y_coords: CoordinateArray = frame_data['y'].values
```

---

## Time Series Types

Arrays spanning multiple frames.

### `TimeSeriesAngles`

Array of angle arrays, one per frame (variable-length).

**Type:**
```python
NDArray[np.object_]
```

**Shape:** `(n_frames,)` - each element is an `AngleArray`

**Usage:**
```python
from constants import get_angles_over_time

angles: TimeSeriesAngles = get_angles_over_time(data)

# Access angles for specific frame
frame_0_angles = angles[0]  # AngleArray for frame 0
```

---

### `TimeSeriesArclengths`

List of arclength arrays, one per frame.

**Type:**
```python
list[ArclengthArray]
```

**Length:** n_frames

**Usage:**
```python
from constants import get_arclengths_over_time

arclengths: TimeSeriesArclengths = get_arclengths_over_time(data)

# Access arclengths for specific frame
frame_0_arcs = arclengths[0]  # ArclengthArray for frame 0
```

---

### `TimeArray`

Array of frame numbers or time indices.

**Type:**
```python
NDArray[np.int64]
```

**Shape:** `(n_frames,)`

**Units:** frame numbers (dimensionless integers)

**Usage:**
```python
import numpy as np

frames: TimeArray = np.array(data['frame'].unique())
```

---

### `TimeSeriesValues`

Generic time series of scalar values (one per frame).

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n_frames,)`

**Units:** varies (depends on measurement)

**Usage:**
```python
from constants import get_lengths_from_centerlines

times: TimeArray
lengths: TimeSeriesValues
units: str

times, lengths, units = get_lengths_from_centerlines(data)
```

---

### `LengthArray`

Array of total lengths per frame.

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n_frames,)`

**Units:** meters

**Usage:**
```python
from constants import get_lengths_from_centerlines

times, lengths, units = get_lengths_from_centerlines(data)
# lengths is LengthArray
```

---

### `CurvatureTimeSeriesArray`

Array of average curvature per frame.

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(n_frames,)`

**Units:** m⁻¹ (inverse meters)

**Usage:**
```python
from constants.geometric_calculations import calculate_average_curvature_over_time

times, curvatures = calculate_average_curvature_over_time(data)
# curvatures is CurvatureTimeSeriesArray
```

---

## Fitting Types

### `PolyCoeffs`

Polynomial coefficients for linear fits.

**Type:**
```python
NDArray[np.float64]
```

**Shape:** `(2,)` - [slope, intercept]

**Usage:**
```python
from constants.fitting import fit_linear

slope, r2, coeffs = fit_linear(times, values)
# coeffs is PolyCoeffs
```

---

### `GrowthRate`

Linear growth rate scalar.

**Type:**
```python
float
```

**Units:** m/frame or m/s

**Usage:**
```python
from constants import fit_growth_rate

growth_rate: GrowthRate = fit_growth_rate(times, lengths)
```

---

### `AngularVelocity`

Angular velocity scalar.

**Type:**
```python
float
```

**Units:** rad/frame or rad/s

**Usage:**
```python
from constants import fit_angular_velocity

angular_velocity: AngularVelocity = fit_angular_velocity(times, angles)
```

---

### `ConvergenceLength`

Characteristic convergence length (Lc).

**Type:**
```python
float
```

**Units:** meters

**Usage:**
```python
from constants.fitting import fit_Lc

x0, Bl, A, Lc, r2 = fit_Lc(arclengths, angles)
# Lc is ConvergenceLength
```

---

### `RSquared`

Coefficient of determination.

**Type:**
```python
float
```

**Range:** [0, 1] - goodness of fit metric

**Usage:**
```python
from constants.fitting import fit_linear

slope, r_squared, coeffs = fit_linear(times, values)
# r_squared is RSquared
```

---

## Complex Return Types

### `LinearFitResult`

Return type for linear fitting.

**Type:**
```python
tuple[float, float, PolyCoeffs]
```

**Structure:** `(slope, r_squared, coeffs)`

**Usage:**
```python
from constants.fitting import fit_linear

result: LinearFitResult = fit_linear(times, values)
slope, r2, coeffs = result
```

---

### `PiecewiseFitResult`

Return type for piecewise linear fitting.

**Type:**
```python
tuple[float, float, int, tuple[PolyCoeffs, PolyCoeffs]]
```

**Structure:** `(slope1, slope2, breakpoint, (p1_coeffs, p2_coeffs))`

**Usage:**
```python
from constants.fitting import fit_piecewise_linear

result: PiecewiseFitResult = fit_piecewise_linear(frames, values)
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
from constants.fitting import fit_saturating_exponential

result: SaturatingExponentialResult = fit_saturating_exponential(times, values)
y_inf, y_0, tau, r2 = result
```

---

### `LogisticGrowthResult`

Return type for logistic growth fit.

**Type:**
```python
tuple[float, float, float, float, float]
```

**Structure:** `(K, y_0, r, t_m, r_squared)`

**Usage:**
```python
from constants.fitting import fit_logistic_growth

result: LogisticGrowthResult = fit_logistic_growth(times, values)
K, y_0, r, t_m, r2 = result
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
from constants.fitting import fit_Lc

result: LcFitResult = fit_Lc(arclengths, angles)
x0, Bl, A, Lc, r2 = result
```

---

### `BastienDeltaResult`

Return type for Bastien Delta fit.

**Type:**
```python
tuple[float, float, PolyCoeffs]
```

**Structure:** `(Delta_over_R, r_squared, coeffs)`

---

### `MaskStabilityResult`

Return type for mask stability analysis.

**Type:**
```python
tuple[ArclengthArray, int | None, float | None]
```

**Structure:** `(stability_scores, steady_start_frame, median_mask_area)`

**Usage:**
```python
from constants.fitting import find_steady_state_from_masks

result: MaskStabilityResult = find_steady_state_from_masks(mask_dir, px_per_m)
scores, Tc, area = result
```

---

### `LengthOverTimeResult`

Return type for length time series functions.

**Type:**
```python
tuple[TimeArray, LengthArray, str]
```

**Structure:** `(frame_numbers, lengths, units_string)`

**Usage:**
```python
from constants import get_lengths_from_centerlines

result: LengthOverTimeResult = get_lengths_from_centerlines(data)
times, lengths, units = result
```

---

### `TimeSeriesResult`

Generic return type for time series analysis.

**Type:**
```python
tuple[TimeArray, TimeSeriesValues]
```

**Structure:** `(frame_numbers, values)`

**Usage:**
```python
from constants.geometric_calculations import (
    get_tip_angles_averaging,
    calculate_average_curvature_over_time
)

result: TimeSeriesResult = get_tip_angles_averaging(data)
times, angles = result

result: TimeSeriesResult = calculate_average_curvature_over_time(data)
times, curvatures = result
```

---

## Dimension Notation

Throughout the documentation:
- **n**: Number of coordinate points in a single centerline
- **n_frames**: Number of time frames
- **Variable-length**: Arrays where length varies per frame

## Type Safety

These type aliases enable:
- **IDE autocomplete**: Better IntelliSense and code suggestions
- **Type checking**: Static analysis with mypy or similar tools
- **Documentation**: Self-documenting code with semantic names
- **Refactoring safety**: Easier to track data flow through functions

## See Also

- [Geometric Calculations API](/reference/geometric-calculations/) - Functions returning geometric types
- [Fitting API](/reference/fitting/) - Functions returning fit result types
- [Constants API](/reference/constants/) - Functions using these types
