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
get_angles(data: pd.DataFrame, show: bool = False) -> AngleArray
```

**Parameters:**
- `data` (DataFrame): DataFrame with x and y columns for a single frame
- `show` (bool, optional): Display angle plot. Default: False

**Returns:**
- `AngleArray`: Array of angles in radians

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
- Result has N-1 angles for N coordinate points

---

### `get_arclengths()`

Compute cumulative arc length from coordinates.

**Signature:**
```python
get_arclengths(data: pd.DataFrame) -> ArclengthArray
```

**Parameters:**
- `data` (DataFrame): Single frame coordinate data

**Returns:**
- `ArclengthArray`: Cumulative arc length in same units as input coordinates

**Example:**
```python
from tropism_toolset import get_arclengths

# Returns lengths in same units as input (e.g., pixels or meters)
arclengths = get_arclengths(frame_data)
total_length = arclengths[-1]
print(f"Total length: {total_length:.4f}")
```

**Notes:**
- No unit conversion is performed - output is in the same units as input
- To convert from pixels to meters, divide the result by your px_to_length factor

---

### `calculate_curvature()`

Calculate curvature from single frame centerline.

**Signature:**
```python
calculate_curvature(data: pd.DataFrame) -> CurvatureArray
```

**Parameters:**
- `data` (DataFrame): Single frame coordinate data

**Returns:**
- `CurvatureArray`: Array of curvature values in m⁻¹ (inverse meters)

**Example:**
```python
from tropism_toolset import calculate_curvature

frame_data = data[data['frame'] == 0]
curvatures = calculate_curvature(frame_data)
```

**Notes:**
- Curvature defined as κ = dθ/ds (rate of change of angle with respect to arclength)
- Positive curvature indicates bending in direction of increasing angle
- Uses central differences for numerical differentiation
- Output length is n-2 where n is number of coordinate points

---

### `calculate_average_curvature()`

Calculate average curvature of single frame.

**Signature:**
```python
calculate_average_curvature(data: pd.DataFrame) -> float
```

**Parameters:**
- `data` (DataFrame): Single frame coordinate data

**Returns:**
- `float`: Average curvature in m⁻¹

**Example:**
```python
from tropism_toolset import calculate_average_curvature

frame_data = data[data['frame'] == 0]
avg_curv = calculate_average_curvature(frame_data)
print(f"Average curvature: {avg_curv:.6f} m⁻¹")
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
    time_slice: Optional[Tuple[int, int]] = None,
    show: bool = False
) -> TimeSeriesAngles
```

**Parameters:**
- `data` (DataFrame): Multi-frame centerline data
- `percent` (int, optional): Percentage of points to analyze
- `side` (str, optional): 'tip' or 'base' for spatial slicing
- `indices` (tuple, optional): (start, end) indices for slicing
- `time_slice` (tuple, optional): (start_frame, end_frame) for temporal slicing
- `show` (bool, optional): Display plots. Default: False

**Returns:**
- `TimeSeriesAngles`: Array of angle arrays (dtype=object), one for each frame

**Example:**
```python
# All frames
angles = get_angles_over_time(data)

# Only tip region
angles_tip = get_angles_over_time(data, percent=30, side='tip')

# Specific time range
angles_range = get_angles_over_time(data, time_slice=(0, 100))

# Access angles for each frame
for i, frame_angles in enumerate(angles):
    print(f"Frame {i}: {len(frame_angles)} angles")
```

---

### `get_arclengths_over_time()`

Calculate arc lengths for multiple frames.

**Signature:**
```python
get_arclengths_over_time(
    data: pd.DataFrame,
    time_indices: Optional[Tuple[int, int]] = None
) -> TimeSeriesArclengths
```

**Parameters:**
- `data` (DataFrame): Multi-frame centerline data with 'frame', 'x', and 'y' columns
- `time_indices` (tuple, optional): Range of frames to analyze (start, end)

**Returns:**
- `TimeSeriesArclengths`: List of arclength arrays, one for each frame

**Example:**
```python
from tropism_toolset import get_arclengths_over_time

arclengths = get_arclengths_over_time(data)

# Track total length over time
lengths = [arc[-1] for arc in arclengths]

# Analyze specific time range
arclengths_subset = get_arclengths_over_time(data, time_indices=(0, 100))
```

**Notes:**
- Returns lengths in the same units as input coordinates
- For spatial slicing (tip/base/indices), use `get_angles_over_time()` pattern
- This function focuses on temporal slicing only

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
    std: bool = False,
    show: bool = False
) -> AngleArray | tuple[AngleArray, AngleArray]
```

**Parameters:**
- `data` (DataFrame): Multi-frame centerline data
- `percent` (int, optional): Percentage of points to analyze
- `side` (str, optional): 'tip' or 'base' for spatial slicing
- `indices` (tuple, optional): (start, end) indices for slicing
- `time_slice` (tuple, optional): (start_frame, end_frame) for temporal slicing
- `std` (bool, optional): If True, also return standard deviations. Default: False
- `show` (bool, optional): Display plots. Default: False

**Returns:**
- `AngleArray`: Array of mean angles (one per frame)
- If `std=True`: `tuple[AngleArray, AngleArray]` - (mean_angles, std_angles)

**Example:**
```python
# Average angles for tip region, one value per frame
avg_tip_angles = get_avg_angles_over_time(data, percent=30, side='tip')

# With standard deviations
means, stds = get_avg_angles_over_time(data, percent=30, side='tip', std=True)
```

**Notes:**
- Calculates angles for each frame, then computes the mean within each frame
- Returns one average angle per frame (not averaged across frames)
- Useful for tracking average angle evolution over time

---

### `calculate_average_curvature_over_time()`

Calculate average curvature for each frame over time.

**Signature:**
```python
calculate_average_curvature_over_time(data: pd.DataFrame) -> TimeSeriesResult
```

**Parameters:**
- `data` (DataFrame): Multi-frame data with 'x', 'y', and 'frame' columns

**Returns:**
- `tuple[TimeArray, np.ndarray]`: (times, curvatures)
  - `times`: Array of frame numbers
  - `curvatures`: Array of average curvature values in m⁻¹

**Example:**
```python
from tropism_toolset import calculate_average_curvature_over_time

times, avg_curvatures = calculate_average_curvature_over_time(data)

# Plot curvature evolution
import matplotlib.pyplot as plt
plt.plot(times, avg_curvatures)
plt.xlabel('Frame')
plt.ylabel('Average Curvature (m⁻¹)')
plt.show()
```

**Notes:**
- Represents overall "bendiness" of centerline at each time point
- Useful for tracking how plant curvature changes over time

---

## Coordinate System Transformations

Functions for converting between Cartesian (x,y) and arc-length (s,θ) coordinate systems.

### `x_y_to_s_theta()`

Transform Cartesian to arc-length coordinates.

**Signature:**
```python
x_y_to_s_theta(
    data: pd.DataFrame,
    px_to_length: Optional[float] = None
) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `px_to_length` (float, optional): Pixel to meter conversion factor. If None and units are inferred from column names, no conversion is applied

**Returns:**
- `DataFrame`: DataFrame with 's' (arclength), 'theta' (angle in radians), and 'frame' columns

**Example:**
```python
from tropism_toolset import x_y_to_s_theta

# Convert pixel coordinates to s,theta in meters
s_theta_data = x_y_to_s_theta(data, px_to_length=100)

# If data already has unit columns like 'x (meters)', 'y (meters)'
s_theta_data = x_y_to_s_theta(data)

# Access specific frame
frame_0 = s_theta_data[s_theta_data['frame'] == 0]
```

**Notes:**
- Works with multi-frame data (groups by 'frame' column)
- Handles length mismatch between angles and arclengths automatically
- Output arclengths skip the first point to match angles length

---

### `s_theta_to_xy()`

Transform arc-length to Cartesian coordinates.

**Signature:**
```python
s_theta_to_xy(
    s_theta_data: pd.DataFrame,
    px_to_length: Optional[float] = None,
    start_x: float = 0.0,
    start_y: float = 0.0,
    output_units: str = "pixels"
) -> pd.DataFrame
```

**Parameters:**
- `s_theta_data` (DataFrame): DataFrame with 's' (arclength in meters), 'theta' (angle in radians), and 'frame' columns
- `px_to_length` (float, optional): Pixel to meter conversion factor. Required if output_units is 'pixels'
- `start_x` (float, optional): Starting x coordinate in meters. Default: 0.0
- `start_y` (float, optional): Starting y coordinate in meters. Default: 0.0
- `output_units` (str, optional): Units for output coordinates: 'pixels', 'meters', or other. Default: 'pixels'

**Returns:**
- `DataFrame`: DataFrame with 'x (unit)', 'y (unit)', and 'frame' columns where unit depends on output_units

**Example:**
```python
from tropism_toolset import s_theta_to_xy

# Reconstruct in pixels
xy_data = s_theta_to_xy(s_theta_data, px_to_length=100, output_units='pixels')

# Reconstruct in meters
xy_meters = s_theta_to_xy(s_theta_data, output_units='meters')

# Access reconstructed coordinates
frame_0 = xy_data[xy_data['frame'] == 0]
```

**Notes:**
- Works with multi-frame data (processes each frame independently)
- Starting position should be specified in meters regardless of output units
- Reconstructs centerline by integrating angles and arc lengths

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
    show: bool = False
) -> LengthOverTimeResult
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `smooth` (bool, optional): Apply Savitzky-Golay smoothing before calculating length. Default: False
- `window_length` (int, optional): Length of smoothing window (must be odd and ≥ 3). Default: 5
- `polyorder` (int, optional): Order of polynomial for smoothing (must be < window_length). Default: 2
- `show` (bool, optional): Display length vs frame plot. Default: False

**Returns:**
- `tuple[TimeArray, np.ndarray, str]`: (times, lengths, units)
  - `times`: Array of frame numbers
  - `lengths`: Array of total centerline lengths
  - `units`: Units of length data ('meters', 'pixels', or None)

**Example:**
```python
from tropism_toolset import get_lengths_from_centerlines

# Without smoothing
times, lengths, units = get_lengths_from_centerlines(data, show=True)

# With smoothing to reduce noise from centerline detection
times, lengths, units = get_lengths_from_centerlines(
    data, smooth=True, window_length=7, polyorder=3
)
```

**Notes:**
- Extracts final (total) length from each frame's arclength array
- Smoothing reduces effects of small errors in centerline detection
- Smoothing is applied per-frame to avoid temporal artifacts

---

### `get_lengths_from_masks()`

Get length time series from binary mask images.

**Signature:**
```python
get_lengths_from_masks(
    mask_dir: Path | str,
    px_to_meters: Optional[float] = None,
    R: Optional[float] = None,
    show: bool = False
) -> LengthOverTimeResult
```

**Parameters:**
- `mask_dir` (Path or str): Directory containing mask image files (.bmp format)
- `px_to_meters` (float, optional): Pixel to meter conversion factor
- `R` (float, optional): Known physical radius in meters (alternative to px_to_meters)
- `show` (bool, optional): Display length vs frame plot. Default: False

**Returns:**
- `tuple[TimeArray, np.ndarray, str]`: (times, lengths, units='meters')

**Example:**
```python
from tropism_toolset.geometric_calculations import get_lengths_from_masks

# Using known radius
times, lengths, units = get_lengths_from_masks(
    mask_dir='data/masks/',
    R=0.0005,  # 0.5mm radius
    show=True
)

# Using known conversion factor
times, lengths, units = get_lengths_from_masks(
    mask_dir='data/masks/',
    px_to_meters=0.0001
)
```

**Notes:**
- Processes all .bmp files in directory in alphabetical order
- Uses contour detection to find plant outline
- Calculates length as: (perimeter - 2×width) / 2
- Must provide either `R` or `px_to_meters`
- `R` parameter useful when you know physical width but not conversion factor

---

## Tip/Base Angle Analysis

Specialized functions for extracting tip angle time series using different methods.

### `get_tip_angles_averaging()`

Get tip angle time series using averaging method.

**Signature:**
```python
get_tip_angles_averaging(
    data: pd.DataFrame,
    tip_percent: int = 10,
    base_percent: int = 10
) -> TimeSeriesResult
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `tip_percent` (int, optional): Percentage of points from tip for averaging. Default: 10
- `base_percent` (int, optional): Percentage of points from base for averaging. Default: 10

**Returns:**
- `tuple[TimeArray, AngleArray]`: (times, angles)
  - `times`: Array of frame numbers
  - `angles`: Array of (average tip angle - average base angle) in radians

**Example:**
```python
from tropism_toolset.geometric_calculations import get_tip_angles_averaging

times, tip_angles = get_tip_angles_averaging(data, tip_percent=15, base_percent=15)
```

**Notes:**
- Averages angles within tip and base regions for each frame
- Returns the difference: mean(tip) - mean(base)

---

### `get_tip_angles_linear_fit()`

Get angle time series using linear fit method.

**Signature:**
```python
get_tip_angles_linear_fit(
    data: pd.DataFrame,
    percent: int = 10,
    side: str = "tip",
    indices: Optional[Tuple[int, int]] = None
) -> TimeSeriesResult
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `percent` (int, optional): Percentage of points for linear fitting. Default: 10
- `side` (str, optional): 'tip' or 'base'. Default: 'tip'
- `indices` (tuple, optional): (start, end) indices for slicing

**Returns:**
- `tuple[TimeArray, AngleArray]`: (times, angles)
  - `times`: Array of frame numbers
  - `angles`: Array of angles from linear fits in radians

**Example:**
```python
from tropism_toolset.geometric_calculations import get_tip_angles_linear_fit

# Fit to tip 10%
times, tip_angles = get_tip_angles_linear_fit(data, percent=10, side='tip')

# Fit to specific indices
times, angles = get_tip_angles_linear_fit(data, indices=(80, 100))
```

**Notes:**
- Fits line to specified region using np.polyfit
- Calculates angle using arctan2(dx, dy) for vertical clockwise convention
- Avoids discontinuities when region passes through vertical
- Direction determined from first to last point in region
- Angle measured from vertical (0° = up), increasing clockwise

---

## Utility Functions

### `infer_columns_and_units()`

Detect x/y column names and units from DataFrame.

**Signature:**
```python
infer_columns_and_units(df: pd.DataFrame) -> Tuple[str, str, str]
```

**Parameters:**
- `df` (DataFrame): Data with coordinate columns

**Returns:**
- `tuple[str, str, str]`: (x_column, y_column, units)

**Supported Formats:**
- 'x', 'y' → units=None
- 'x (meters)', 'y (meters)' → units='meters'
- 'x (m)', 'y (m)' → units='meters'
- 'x (pixels)', 'y (pixels)' → units='pixels'
- 'x (px)', 'y (px)' → units='pixels'

**Example:**
```python
from tropism_toolset import infer_columns_and_units

x_col, y_col, units = infer_columns_and_units(data)
print(f"Using columns: {x_col}, {y_col} with units: {units}")
```

---

### `calculate_initial_base_angle()`

Calculate sin(mean_angle) from the base of the first frame.

**Signature:**
```python
calculate_initial_base_angle(data: pd.DataFrame) -> float
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns

**Returns:**
- `float`: sin(mean_base_angle) from first frame, where angle is measured from vertical

**Example:**
```python
from tropism_toolset import calculate_initial_base_angle

sin_theta_0 = calculate_initial_base_angle(data)
print(f"Initial base angle parameter: {sin_theta_0:.4f}")
```

**Notes:**
- Uses the entire length (100%) of the base from the first frame
- Angles are in vertical clockwise convention (0 = up)
- Useful for initializing Chauvet gravitropism models

---

## Type Definitions

See [Types Reference](/reference/types/) for detailed type definitions:

- `AngleArray`: np.ndarray of angles
- `ArclengthArray`: np.ndarray of arc lengths
- `CoordinateArray`: np.ndarray of coordinates
- `CurvatureArray`: np.ndarray of curvature values
- `TimeSeriesAngles`: np.ndarray (dtype=object) of angle arrays
- `TimeSeriesArclengths`: List of arclength arrays
- `TimeSeriesResult`: tuple[TimeArray, np.ndarray]
- `LengthOverTimeResult`: tuple[TimeArray, np.ndarray, str]
- `TimeArray`: np.ndarray of frame numbers

## See Also

- [Constants API](/reference/constants/) - For extracting physical constants (γ, β, Lc)
- [Fitting API](/reference/fitting/) - For curve fitting and model analysis
- [Visualization API](/reference/visualization/) - For plotting functions
