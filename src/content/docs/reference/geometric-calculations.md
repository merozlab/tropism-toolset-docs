---
title: Geometric Calculations API
description: API reference for geometric analysis functions.
---

Functions for calculating angles, arc lengths, and coordinate transformations. The module is organized into sections for single-frame calculations, time series analysis, coordinate transformations, and specialized analyses.

## Single-Frame Geometric Calculations

Core functions for analyzing individual frames of centerline data.

### `get_angles()`

Calculate angles from x,y coordinate differences in a single frame.

**Signature:**
```python
get_angles(data: pd.DataFrame, show: bool = False) -> pd.Series
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with x and y columns for a single frame
- `show` (bool, optional): Display angle plot. Default: False

**Returns:**
- `pd.Series`: Series of angles in radians. Name: "angles (rad)"

**Example:**
```python
from tropism_toolset import get_angles
import pandas as pd

frame_data = data[data['frame'] == 0]
angles = get_angles(frame_data, show=True)
```

**Notes:**
- Accepts column names: 'x', 'y' or 'x (meters)', 'y (meters)' or 'x (pixels)', 'y (pixels)'
- Angles measured from vertical (0° = pointing up), increasing clockwise
- Uses arctan2(dx, dy) for vertical clockwise convention

---

### `get_arclengths()`

Compute cumulative arc length from coordinates.

**Signature:**
```python
get_arclengths(data: pd.DataFrame) -> pd.Series
```

**Parameters:**
- `data` (pd.DataFrame): Single frame coordinate data

**Returns:**
- `pd.Series`: Cumulative arc length in same units as input coordinates

**Example:**
```python
from tropism_toolset import get_arclengths

# Returns lengths in same units as input (e.g., pixels or meters)
arclengths = get_arclengths(frame_data)
total_length = arclengths.iloc[-1]
print(f"Total length: {total_length:.4f}")
```

---

### `compute_strain_rate()`

Compute strain rate ε̇ = ∂v_g/∂s using finite differences.

**Signature:**
```python
compute_strain_rate(
    s: np.ndarray, 
    v_g: np.ndarray, 
    method: str = "central"
) -> Tuple[np.ndarray, np.ndarray]
```

**Parameters:**
- `s` (np.ndarray): Arc length coordinates [m].
- `v_g` (np.ndarray): Growth velocity at each arc length position.
- `method` (str, optional): 'central', 'forward', or 'backward'. Default: 'central'

**Returns:**
- `tuple[np.ndarray, np.ndarray]`: (s_strain, epsilon_dot)
    - `s_strain`: Arc length coordinates where strain rate is computed
    - `epsilon_dot`: Strain rate ε̇ = ∂v_g/∂s

**Example:**
```python
from tropism_toolset.geometric_calculations import compute_strain_rate

s_strain, epsilon_dot = compute_strain_rate(s, v_g, method='central')
```

---

## Time Series Geometric Calculations

Functions for analyzing centerline data across multiple frames.

### `get_angles_over_time()`

Calculate angles for multiple frames.

**Signature:**
```python
get_angles_over_time(
    data: pd.DataFrame,
    percent: Optional[int] = None,
    side: Optional[str] = None,
    indices: Optional[Tuple[int, int]] = None,
    time_slice: Tuple[int, int] | int | None = None,
    show: bool = False
) -> np.ndarray
```

**Parameters:**
- `data` (pd.DataFrame): Multi-frame centerline data
- `percent` (int, optional): Percentage of points to analyze
- `side` (str, optional): 'tip' or 'base' for spatial slicing
- `indices` (tuple, optional): (start, end) indices for slicing
- `time_slice` (tuple | int, optional): (start_frame, end_frame) or single frame index
- `show` (bool, optional): Display plots. Default: False

**Returns:**
- `np.ndarray`: Array of angle arrays (dtype=object), one for each frame

**Example:**
```python
# All frames
angles = get_angles_over_time(data)

# Only tip region
angles_tip = get_angles_over_time(data, percent=30, side='tip')
```

---

### `get_arclengths_over_time()`

Calculate cumulative lengths over multiple time frames for a centerline dataset.

**Signature:**
```python
get_arclengths_over_time(
    data: pd.DataFrame,
    time_indices: Tuple[int, int] | None = None
) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): Multi-frame centerline data with 'frame', 'x', and 'y' columns
- `time_indices` (tuple, optional): Range of frames to analyze (start, end)

**Returns:**
- `pd.DataFrame`: Melted DataFrame with columns [time_col, "arclength (unit)"]

---

### `get_avg_angles_over_time()`

Calculate average angle for each time frame.

**Signature:**
```python
get_avg_angles_over_time(
    data: pd.DataFrame,
    percent: Optional[int] = None,
    side: Optional[str] = None,
    indices: Optional[Tuple[int, int]] = None,
    time_slice: Optional[Tuple[int, int]] = None,
    show: bool = False
) -> pd.Series
```

**Parameters:**
- `data` (pd.DataFrame): Multi-frame centerline data
- `percent` (int, optional): Percentage of points to analyze
- `side` (str, optional): 'tip' or 'base' for spatial slicing
- `indices` (tuple, optional): (start, end) indices for slicing
- `time_slice` (tuple, optional): (start_frame, end_frame) for temporal slicing
- `show` (bool, optional): Display plots. Default: False

**Returns:**
- `pd.Series`: Series of mean angles (one per frame)

**Example:**
```python
# Average angles for tip region, one value per frame
avg_tip_angles = get_avg_angles_over_time(data, percent=30, side='tip')
```

---

## Coordinate System Transformations

Functions for converting between Cartesian (x,y) and arc-length (s,θ) coordinate systems.

### `x_y_to_s_theta()`

Transform Cartesian to arc-length coordinates.

**Signature:**
```python
x_y_to_s_theta(data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns

**Returns:**
- `pd.DataFrame`: DataFrame with 's' (arclength), 'theta' (angle in radians), and 'frame' columns

**Example:**
```python
from tropism_toolset import x_y_to_s_theta

# Convert pixel coordinates to s,theta (units match input)
s_theta_data = x_y_to_s_theta(data)
```

---

### `s_theta_to_xy()`

Transform arc-length to Cartesian coordinates.

**Signature:**
```python
s_theta_to_xy(
    s_theta_data: pd.DataFrame,
    start_x: float = 0.0,
    start_y: float = 0.0,
    output_units: str = "meters"
) -> pd.DataFrame
```

**Parameters:**
- `s_theta_data` (pd.DataFrame): DataFrame with 's' (arclength), 'theta' (angle in radians), and 'frame' columns
- `start_x` (float, optional): Starting x coordinate. Default: 0.0
- `start_y` (float, optional): Starting y coordinate. Default: 0.0
- `output_units` (str, optional): Units for output coordinates (e.g. 'meters', 'pixels'). Default: 'meters'

**Returns:**
- `pd.DataFrame`: DataFrame with 'x (unit)', 'y (unit)', and 'frame' columns

**Example:**
```python
from tropism_toolset import s_theta_to_xy

# Reconstruct in meters
xy_meters = s_theta_to_xy(s_theta_data, output_units='meters')
```

---

## Length Measurements

Functions for extracting length time series from centerline or mask data.

### `get_lengths_from_centerlines()`

Get length time series from centerline coordinate data.

**Signature:**
```python
get_lengths_from_centerlines(
    data: pd.DataFrame,
    smooth: bool = False,
    window_length: int = 5,
    polyorder: int = 2,
) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `smooth` (bool, optional): Apply Savitzky-Golay smoothing. Default: False
- `window_length` (int, optional): Length of smoothing window. Default: 5
- `polyorder` (int, optional): Order of polynomial. Default: 2

**Returns:**
- `pd.DataFrame`: DataFrame with columns containing time and length data.

**Example:**
```python
from tropism_toolset import get_lengths_from_centerlines

# Without smoothing
length_data = get_lengths_from_centerlines(data)

# With smoothing to reduce noise from centerline detection
length_data = get_lengths_from_centerlines(
    data, smooth=True, window_length=7, polyorder=3
)
```

---

### `get_lengths_from_masks()`

Get length time series from binary mask images.

**Signature:**
```python
get_lengths_from_masks(
    mask_dir: Path | str,
    px_to_meters: float,
    frame_to_s: float | None = None,
    R: float | None = None,
) -> pd.DataFrame
```

**Parameters:**
- `mask_dir` (Path or str): Directory containing mask image files (.bmp format)
- `px_to_meters` (float, optional): Pixel to meter conversion factor.
- `frame_to_s` (float, optional): Frame to seconds conversion factor.
- `R` (float, optional): Known physical radius in meters (alternative to px_to_meters)

**Returns:**
- `pd.DataFrame`: DataFrame with time and length columns.

**Example:**
```python
from tropism_toolset.geometric_calculations import get_lengths_from_masks

# Using known conversion factor
length_data = get_lengths_from_masks(
    mask_dir='data/masks/',
    px_to_meters=0.0001
)
```

---

## Tip/Base Angle Analysis

Specialized functions for extracting tip angle time series using different methods.

### `get_tip_angles_linear_fit()`

Get angle time series using linear fit method.

**Signature:**
```python
get_tip_angles_linear_fit(
    data: pd.DataFrame,
    percent: int = 10,
    side: str = "tip",
    indices: Optional[Tuple[int, int]] = None,
    bounds: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `percent` (int, optional): Percentage of points for linear fitting. Default: 10
- `side` (str, optional): 'tip' or 'base'. Default: 'tip'
- `indices` (tuple, optional): (start, end) indices for slicing
- `bounds` (tuple, optional): (min, max) bounds in radians.

**Returns:**
- `pd.DataFrame`: DataFrame with time and angle columns.

**Example:**
```python
from tropism_toolset.geometric_calculations import get_tip_angles_linear_fit

# Fit to tip 10%
tip_angles_df = get_tip_angles_linear_fit(data, percent=10, side='tip')
```

---

## Utility Functions

### `infer_columns_and_units()`

Detect x/y column names and units from DataFrame.

**Signature:**
```python
infer_columns_and_units(df: pd.DataFrame) -> Tuple[str | None, str | None, str | None]
```

**Parameters:**
- `df` (pd.DataFrame): Data with coordinate columns

**Returns:**
- `tuple`: (x_column, y_column, units)

---

### `calculate_initial_base_angle()`

Calculate sin(mean_angle) from the base of the first frame.

**Signature:**
```python
calculate_initial_base_angle(data: pd.DataFrame) -> float
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns

**Returns:**
- `float`: sin(mean_base_angle) from first frame

---

## See Also

- [Constants API](/reference/constants/) - For extracting physical constants (γ, β, Lc)
- [Fitting API](/reference/fitting/) - For curve fitting and model analysis
- [Visualization API](/reference/visualization/) - For plotting functions