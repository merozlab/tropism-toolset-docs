---
title: Utils Module API
description: API reference for centerline data manipulation utility functions.
---

Utility functions for manipulating centerline data, including flipping coordinates, reversing point order, checking orientation, and converting units.

## Coordinate Manipulation

### `flip_centerlines()`

Flip centerline coordinates horizontally and/or vertically.

**Signature:**
```python
flip_centerlines(
    data: pd.DataFrame,
    direction: Literal["horizontal", "vertical", "both"] = "horizontal"
) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x' and 'y' columns (or 'x (unit)', 'y (unit)')
- `direction` (str, optional): Direction to flip:
  - `'horizontal'`: flip x-coordinates (left-right mirror)
  - `'vertical'`: flip y-coordinates (top-bottom mirror)
  - `'both'`: flip both x and y coordinates
  - Default: `'horizontal'`

**Returns:**
- `DataFrame`: DataFrame with flipped coordinates

**Example:**
```python
from tropism_toolset.utils import flip_centerlines

# Flip horizontally (mirror left-right)
df_flipped = flip_centerlines(df, direction='horizontal')

# Flip vertically (mirror top-bottom)
df_flipped = flip_centerlines(df, direction='vertical')

# Flip both directions
df_flipped = flip_centerlines(df, direction='both')
```

**Notes:**
- Flipping is performed around the center of the coordinate range
- Automatically detects column names (x/y or x (unit)/y (unit))
- All frames in the DataFrame are flipped using the same center point
- Returns a new DataFrame; does not modify the original

---

### `flip_centerlines_horizontal()`

Flip centerline coordinates horizontally (left-right mirror).

**Signature:**
```python
flip_centerlines_horizontal(data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x' and 'y' columns

**Returns:**
- `DataFrame`: DataFrame with horizontally flipped coordinates

**Example:**
```python
from tropism_toolset.utils import flip_centerlines_horizontal

df_flipped = flip_centerlines_horizontal(df)
```

**Notes:**
- Convenience function that calls `flip_centerlines` with `direction='horizontal'`

---

### `flip_centerlines_vertical()`

Flip centerline coordinates vertically (top-bottom mirror).

**Signature:**
```python
flip_centerlines_vertical(data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x' and 'y' columns

**Returns:**
- `DataFrame`: DataFrame with vertically flipped coordinates

**Example:**
```python
from tropism_toolset.utils import flip_centerlines_vertical

df_flipped = flip_centerlines_vertical(df)
```

**Notes:**
- Convenience function that calls `flip_centerlines` with `direction='vertical'`

---

### `reverse_centerline_order()`

Reverse the order of points along the centerline for all frames.

**Signature:**
```python
reverse_centerline_order(data: pd.DataFrame) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'frame' column

**Returns:**
- `DataFrame`: DataFrame with reversed centerline point order

**Example:**
```python
from tropism_toolset.utils import reverse_centerline_order

# Reverse point order (swap base and tip)
df_reversed = reverse_centerline_order(df)
```

**Notes:**
- Reverses the order of centerline points within each frame
- Effectively swaps the base and tip
- Useful when centerline extraction started from the wrong end
- The reversal is applied independently to each frame
- Frame numbers remain unchanged
- Only the order of points within each frame is reversed
- Returns a new DataFrame; does not modify the original

---

## Orientation Analysis

### `check_centerline_orientation()`

Check the orientation of a centerline to help determine if flipping is needed.

**Signature:**
```python
check_centerline_orientation(
    data: pd.DataFrame,
    frame: int = -1,
    verbose: bool = True
) -> dict
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `frame` (int, optional): Frame number to analyze. If -1, uses the last frame. Default: -1
- `verbose` (bool, optional): If True, print orientation information. Default: True

**Returns:**
- `dict`: Dictionary containing:
  - `'frame_n'`: Frame number analyzed
  - `'first_point'`: (x, y) coordinates of first point
  - `'last_point'`: (x, y) coordinates of last point
  - `'x_direction'`: 'left-to-right' or 'right-to-left'
  - `'y_direction'`: 'bottom-to-top' or 'top-to-bottom'
  - `'total_points'`: Number of points in the centerline

**Example:**
```python
from tropism_toolset.utils import check_centerline_orientation

# Check orientation of last frame
info = check_centerline_orientation(df)

# Check orientation of specific frame
info = check_centerline_orientation(df, frame=50)

# Get info without printing
info = check_centerline_orientation(df, verbose=False)
```

**Notes:**
- Helps determine if `flip_centerlines()` or `reverse_centerline_order()` is needed
- Assumes image coordinates (y increases downward) for direction descriptions
- Provides suggestions for correction if needed

---

## Unit Conversion

### `convert_centerline_units()`

Convert centerline data from pixels and frames to meters and seconds.

**Signature:**
```python
convert_centerline_units(
    data: pd.DataFrame,
    px_to_m: Optional[float] = None,
    frame_to_s: Optional[float] = None
) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `px_to_m` (float, optional): Conversion factor from pixels to meters (e.g., 0.001 means 1 pixel = 0.001 meters). If None, spatial coordinates are not converted. Default: None
- `frame_to_s` (float, optional): Conversion factor from frames to seconds (e.g., 0.5 means 1 frame = 0.5 seconds). If None, temporal coordinates are not converted. Default: None

**Returns:**
- `DataFrame`: DataFrame with converted coordinates and updated column names

**Example:**
```python
from tropism_toolset.utils import convert_centerline_units

# Convert pixels to meters (1 pixel = 0.001 m)
df_m = convert_centerline_units(df, px_to_m=0.001)

# Convert frames to seconds (1 frame = 0.5 s)
df_s = convert_centerline_units(df, frame_to_s=0.5)

# Convert both spatial and temporal units
df_converted = convert_centerline_units(
    df,
    px_to_m=0.001,      # 1 pixel = 1 mm
    frame_to_s=0.5      # 1 frame = 0.5 seconds
)
```

**Notes:**
- Automatically detects current units using `infer_columns_and_units()`
- Updates column names to reflect new units (e.g., 'x' → 'x (meters)')
- If data already has unit labels, they are preserved and updated
- Conversion is applied to all frames in the DataFrame
- Frame numbers are converted to time values if `frame_to_s` is provided
- If data already has 'time (seconds)' column, spatial conversion is skipped
- Returns a new DataFrame; does not modify the original

---

### `convert_px_to_m()`

Convert centerline spatial coordinates from pixels to meters.

**Signature:**
```python
convert_px_to_m(
    data: pd.DataFrame,
    px_to_m: float
) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with centerline data
- `px_to_m` (float): Conversion factor from pixels to meters

**Returns:**
- `DataFrame`: DataFrame with coordinates converted to meters

**Example:**
```python
from tropism_toolset.utils import convert_px_to_m

# Convert pixels to meters (1 pixel = 1 mm)
df_meters = convert_px_to_m(df, px_to_m=0.001)
```

**Notes:**
- Convenience function that calls `convert_centerline_units` with only spatial conversion

---

### `convert_frames_to_s()`

Convert centerline temporal coordinates from frames to seconds.

**Signature:**
```python
convert_frames_to_s(
    data: pd.DataFrame,
    frame_to_s: float
) -> pd.DataFrame
```

**Parameters:**
- `data` (DataFrame): DataFrame with 'frame' column
- `frame_to_s` (float): Conversion factor from frames to seconds

**Returns:**
- `DataFrame`: DataFrame with 'time (seconds)' column added

**Example:**
```python
from tropism_toolset.utils import convert_frames_to_s

# Convert frames to seconds (1 frame = 0.5 s)
df_with_time = convert_frames_to_s(df, frame_to_s=0.5)
```

**Notes:**
- Convenience function that calls `convert_centerline_units` with only temporal conversion
- Adds 'time (seconds)' column while keeping the original 'frame' column

---

## Common Workflows

### Correct Centerline Orientation

If your centerline extraction has the wrong orientation:

```python
from tropism_toolset.utils import (
    check_centerline_orientation,
    flip_centerlines,
    reverse_centerline_order
)

# Step 1: Check current orientation
info = check_centerline_orientation(df)

# Step 2: Apply corrections based on output
# If centerline goes right-to-left but should go left-to-right:
df_corrected = flip_centerlines(df, direction='horizontal')

# If centerline starts at tip but should start at base:
df_corrected = reverse_centerline_order(df)

# If both issues exist:
df_corrected = flip_centerlines(df, direction='horizontal')
df_corrected = reverse_centerline_order(df_corrected)
```

### Convert Units for Analysis

Before extracting physical constants, convert to SI units:

```python
from tropism_toolset.utils import convert_centerline_units

# Convert from pixel coordinates at 15-minute intervals
df_si = convert_centerline_units(
    df,
    px_to_m=0.0001,     # 0.1 mm per pixel
    frame_to_s=900      # 15 minutes per frame
)

# Now use df_si for analysis - all functions will use SI units
from tropism_toolset import get_lengths_from_centerlines, fit_growth_rate

length_data = get_lengths_from_centerlines(df_si, smooth=True)
growth_rate = fit_growth_rate(length_data, display=True)
print(f"Growth rate: {growth_rate:.6e} m/s")
```

---

## See Also

- [Geometric Calculations API](/reference/geometric-calculations/) - Uses unit-aware DataFrames
- [Data Handling Guide](/guides/data-handling/) - Complete data preparation workflow
- [Getting Started](/guides/installation/) - Setup and basic usage
