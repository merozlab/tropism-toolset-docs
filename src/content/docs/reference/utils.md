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
- `data` (pd.DataFrame): DataFrame with 'x' and 'y' columns (or 'x (unit)', 'y (unit)')
- `direction` (str, optional): Direction to flip:
  - `'horizontal'`: flip x-coordinates (left-right mirror)
  - `'vertical'`: flip y-coordinates (top-bottom mirror)
  - `'both'`: flip both x and y coordinates
  - Default: `'horizontal'`

**Returns:**
- `pd.DataFrame`: DataFrame with flipped coordinates

**Example:**
```python
from tropism_toolset.utils import flip_centerlines

# Flip horizontally (mirror left-right)
df_flipped = flip_centerlines(df, direction='horizontal')
```

---

### `flip_centerlines_horizontal()`

Flip centerline coordinates horizontally.

**Signature:**
```python
flip_centerlines_horizontal(data: pd.DataFrame) -> pd.DataFrame
```

---

### `flip_centerlines_vertical()`

Flip centerline coordinates vertically.

**Signature:**
```python
flip_centerlines_vertical(data: pd.DataFrame) -> pd.DataFrame
```

---

### `rotate_centerlines_to_horizontal()`

Rotate all centerline data so that the reference frame's base-to-tip direction becomes horizontal or vertical.

**Signature:**
```python
rotate_centerlines_to_horizontal(
    data: pd.DataFrame,
    reference_frame: int = 0,
    orientation: str = "horizontal"
) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame containing centerline data with x, y, and frame columns
- `reference_frame` (int, optional): Frame number to use for determining the rotation angle. Default: 0
- `orientation` (str, optional): Target orientation: "horizontal" or "vertical". Default: "horizontal"

**Returns:**
- `pd.DataFrame`: DataFrame with rotated coordinates

---

### `reverse_centerline_order()`

Reverse the order of points along the centerline for all frames.

**Signature:**
```python
reverse_centerline_order(
    data: pd.DataFrame,
    correct_direction: Literal["ltr", "rtl"] | None = None
) -> pd.DataFrame
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame with 'frame' column
- `correct_direction` (str, optional): Check if reversal is needed based on expected direction ("ltr" for left-to-right, "rtl" for right-to-left). If condition is met, no reversal occurs. Default: None

**Returns:**
- `pd.DataFrame`: DataFrame with reversed centerline point order

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
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `frame` (int, optional): Frame number to analyze. Default: -1 (last frame)
- `verbose` (bool, optional): If True, print orientation information. Default: True

**Returns:**
- `dict`: Dictionary containing orientation details.

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
- `data` (pd.DataFrame): DataFrame with 'x', 'y', and 'frame' columns
- `px_to_m` (float, optional): Conversion factor from pixels to meters.
- `frame_to_s` (float, optional): Conversion factor from frames to seconds.

**Returns:**
- `pd.DataFrame`: DataFrame with converted coordinates and updated column names

**Example:**
```python
from tropism_toolset.utils import convert_centerline_units

# Convert units
df_si = convert_centerline_units(
    df,
    px_to_m=0.001,      # 1 pixel = 1 mm
    frame_to_s=0.5      # 1 frame = 0.5 seconds
)
```

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

---

## Metadata inference

### `infer_columns_and_units()`

Detect x/y column names and units from DataFrame.

**Signature:**
```python
infer_columns_and_units(df: pd.DataFrame) -> Tuple[str | None, str | None, str | None]
```

**Returns:**
- `tuple`: (x_column, y_column, units)

### `extract_series_name_and_unit()`

Extract the variable name and unit from a pandas Series name.

**Signature:**
```python
extract_series_name_and_unit(series: pd.Series) -> tuple[str, str | None]
```

**Example:**
```python
from tropism_toolset.utils import extract_series_name_and_unit
s = pd.Series([1, 2], name="length (m)")
name, unit = extract_series_name_and_unit(s)
# name="length", unit="m"
```

## Common Workflows

### Correct Centerline Orientation

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
```

## See Also

- [Geometric Calculations API](/reference/geometric-calculations/) - Uses unit-aware DataFrames
- [Data Handling Guide](/guides/data-handling/) - Complete data preparation workflow