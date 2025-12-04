---
title: Working with Data
description: Load, verify, orient, and prepare centerline data for gravitropic analysis.
---

Learn how to load centerline data, verify its orientation, correct any issues, convert units, and prepare it for analysis. This workflow ensures your data is properly oriented and calibrated before extracting physical constants.

## Data Preparation Workflow

When working with centerline data, follow this systematic workflow to catch and fix common issues:

1. **Load data** - Read CSV file into DataFrame
2. **Check orientation** - Verify points start at base and end at tip
3. **Fix orientation** - Flip or reverse if needed
4. **Convert units** - Transform from pixels to meters, frames to seconds
5. **Visualize** - Confirm data looks correct
6. **Transform coordinates** - Convert (x, y) to (s, θ) if needed

---

## Step 1: Load Centerline Data

Start by loading your CSV file with centerline coordinates:

```python
import pandas as pd

# Load centerline data
df = pd.read_csv("experiment_centerlines.csv")

# Inspect the data
print(df.head())
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Frames: {df['frame'].min()} to {df['frame'].max()}")
```

### Expected Data Format

```csv
frame,x,y
0,245.0,1088.0
0,246.2,1076.3
0,248.1,1064.8
1,245.5,1089.2
1,246.8,1077.1
...
```

**Required columns:**
- `frame`: Integer frame number (0-indexed or sequential)
- `x`: Horizontal coordinate (in pixels initially)
- `y`: Vertical coordinate (in pixels initially)

**Critical:** Points should be ordered from **base to tip** within each frame. We'll verify this in the next step.

---

## Step 2: Check Centerline Orientation

Before any analysis, verify that your centerline points are ordered correctly and oriented as expected. The `display_centerline_endpoints()` function shows you the first and last points.

```python
from tropism_toolset import display_centerline_endpoints

# Check orientation of a specific frame
display_centerline_endpoints(
    df,
    frame=20,  # Choose a middle frame
    display=True
)
```

**What to look for:**
- **Point order:** Points should start at the **base** (typically bottom) and end at the **tip** (typically top)
- **Coordinate system:** Image coordinates have (0,0) at top-left, y increases downward
- **Expected orientation:** For vertical plants, base should be at higher y-values, tip at lower y-values

### Understanding the Output

![Display Centerline Endpoints Example](/display-endpoints.png)

The plot shows:
- **Green circle** = First point (should be at base)
- **Red circle** = Last point (should be at tip)
- **Blue line** = Centerline trajectory

**Common issues:**
1. **Reversed order:** Tip and base are swapped → Use `reverse_centerline_order()`
2. **Wrong orientation:** Plant points wrong direction → Use `flip_centerlines()`
3. **Both issues:** Need both corrections

---

## Step 3: Fix Orientation Issues

Based on what you saw in Step 2, apply corrections:

### Reverse Point Order

If the first point is at the tip instead of the base:

```python
from tropism_toolset import reverse_centerline_order

# Reverse the order of points within each frame
df = reverse_centerline_order(df)

# Verify the fix
display_centerline_endpoints(df, frame=20, display=True)
```

### Flip Coordinates

If the plant is oriented incorrectly (e.g., pointing left instead of right):

```python
from tropism_toolset import flip_centerlines

# Flip horizontally (left-right mirror)
df = flip_centerlines(df, direction='horizontal')

# Or flip vertically (top-bottom mirror)
df = flip_centerlines(df, direction='vertical')

# Or flip both
df = flip_centerlines(df, direction='both')

# Verify the fix
display_centerline_endpoints(df, frame=20, display=True)
```

**Best practice:** Aim for points starting at **bottom-left** and progressing upward/rightward. This matches the typical gravitropic experiment setup where:
- Base is at bottom (higher y in image coordinates)
- Plant grows upward (decreasing y)
- Gravitropic response bends to the right (increasing x)

### Check Multiple Frames

Always verify corrections across several frames:

```python
# Check early, middle, and late frames
for frame_num in [0, len(df['frame'].unique())//2, df['frame'].max()]:
    print(f"\nFrame {frame_num}:")
    display_centerline_endpoints(df, frame=frame_num, display=True)
```

---

## Step 4: Convert Units

Once orientation is correct, convert from pixels and frames to SI units (meters and seconds):

```python
from tropism_toolset import convert_centerline_units

# Define conversion factors
PX_TO_M = 35 / 1_000_000  # 35 pixels = 1 mm = 0.001 m
FRAME_TO_S = 9 * 60        # 9 minutes per frame = 540 seconds

# Convert to SI units
df = convert_centerline_units(
    df,
    px_to_m=PX_TO_M,
    frame_to_s=FRAME_TO_S
)

# Check the result
print(df.head())
print(f"Columns after conversion: {list(df.columns)}")
```

**After conversion, you'll have:**
- `x (meters)` instead of `x`
- `y (meters)` instead of `y`
- `time (seconds)` column added
- Original `frame` column retained

### Determining Conversion Factors

**Spatial calibration (px_to_m):**

```python
# Method 1: Known scale bar
scale_bar_pixels = 100
scale_bar_mm = 10
px_to_m = scale_bar_pixels / (scale_bar_mm / 1000)

# Method 2: Known organ radius
known_radius_mm = 1.0  # 1 mm radius
measured_radius_px = 35  # measured in image
px_to_m = measured_radius_px / (known_radius_mm / 1000)
```

**Temporal calibration (frame_to_s):**

```python
# From imaging settings
minutes_per_frame = 15
frame_to_s = minutes_per_frame * 60  # = 900 seconds

# Or directly in seconds
frame_to_s = 540  # 9 minutes
```

---

## Step 5: Visualize Data

Before proceeding with analysis, visually confirm your data is correct:

```python
from tropism_toolset import plot_centerline_data
import matplotlib.pyplot as plt

# Plot all frames
plot_centerline_data(df)
plt.show()
```

![Plot Centerline Data Example](/plot-centerlines.png)

**What to check:**
- **Smooth progression:** Centerlines should evolve smoothly over time
- **No sudden jumps:** Discontinuities indicate tracking errors
- **Correct direction:** Plant should grow/bend in expected direction
- **Reasonable scale:** If units are converted, check values are in meters

### Plotting Specific Frame Ranges

```python
# Plot only first 50 frames
subset = df[df['frame'] <= 50]
plot_centerline_data(subset)
plt.title("Early Growth Phase (Frames 0-50)")
plt.show()

# Plot steady-state region
steady_frames = df[df['frame'] >= 120]
plot_centerline_data(steady_frames)
plt.title("Steady State (Frame 120+)")
plt.show()
```

---

## Step 6: Transform to Arc-Length Coordinates

For many analyses (especially fitting convergence length), you need to convert from Cartesian (x, y) to arc-length (s, θ) coordinates:

```python
from tropism_toolset import x_y_to_s_theta

# Convert entire dataset
s_theta_data = x_y_to_s_theta(df)

# Inspect result
print(s_theta_data.head())
print(f"Columns: {list(s_theta_data.columns)}")
```

**Output columns:**
- `frame`: Frame number (preserved)
- `s`: Arc length from base (in same units as input coordinates)
- `theta`: Angle from vertical in radians (0 = pointing up, clockwise positive)

### Using Arc-Length Data

```python
# Get a specific frame in s-theta coordinates
frame_100 = s_theta_data[s_theta_data['frame'] == 100]

# Plot angle vs arc length
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(frame_100['s'], frame_100['theta'], 'o-')
plt.xlabel('Arc length s (m)')
plt.ylabel('Angle θ (rad)')
plt.title('Angle Profile at Frame 100')
plt.grid(True, alpha=0.3)
plt.show()
```

### Converting Back to Cartesian

If needed, you can reconstruct (x, y) from (s, θ):

```python
from tropism_toolset import s_theta_to_xy

# Reconstruct in meters
xy_reconstructed = s_theta_to_xy(
    s_theta_data,
    start_x=0.0,
    start_y=0.0,
    output_units='meters'
)

# Or reconstruct in pixels
xy_pixels = s_theta_to_xy(
    s_theta_data,
    start_x=0.0,
    start_y=0.0,
    output_units='pixels'
)
```

---

## Complete Data Preparation Script

Here's a complete workflow for preparing centerline data:

```python
import pandas as pd
from tropism_toolset import (
    display_centerline_endpoints,
    flip_centerlines,
    reverse_centerline_order,
    convert_centerline_units,
    plot_centerline_data,
    x_y_to_s_theta
)
import matplotlib.pyplot as plt

# ============================================================================
# STEP 1: Load data
# ============================================================================
df = pd.read_csv('experiment_centerlines.csv')
print(f"Loaded {len(df['frame'].unique())} frames with {len(df)} total points")

# ============================================================================
# STEP 2: Check orientation
# ============================================================================
print("\nChecking orientation...")
display_centerline_endpoints(df, frame=20, display=True)

# ============================================================================
# STEP 3: Fix orientation (adjust based on Step 2 results)
# ============================================================================
# Uncomment if needed:
# df = reverse_centerline_order(df)  # If points go from tip to base
# df = flip_centerlines(df, direction='horizontal')  # If left-right is wrong
# df = flip_centerlines(df, direction='vertical')  # If top-bottom is wrong

# ============================================================================
# STEP 4: Convert units
# ============================================================================
print("\nConverting units...")
df = convert_centerline_units(
    df,
    px_to_m=35 / 1_000_000,  # 35 px = 1 mm
    frame_to_s=9 * 60         # 9 minutes per frame
)
print(f"Columns after conversion: {list(df.columns)}")

# ============================================================================
# STEP 5: Visualize
# ============================================================================
print("\nVisualizing data...")
plot_centerline_data(df)
plt.savefig('centerlines_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# STEP 6: Transform coordinates (optional, for Lc fitting)
# ============================================================================
print("\nTransforming to arc-length coordinates...")
s_theta_data = x_y_to_s_theta(df)
print(f"Arc-length data shape: {s_theta_data.shape}")

# ============================================================================
# STEP 7: Save processed data
# ============================================================================
df.to_csv('experiment_centerlines_processed.csv', index=False)
s_theta_data.to_csv('experiment_s_theta.csv', index=False)
print("\nData preparation complete!")
print("Saved:")
print("  - experiment_centerlines_processed.csv (x, y in meters)")
print("  - experiment_s_theta.csv (s, theta coordinates)")
```

---

## Troubleshooting Common Issues

### Issue 1: "Points seem to be in wrong order"

**Symptoms:** First point (green) is at the tip, last point (red) is at the base

**Solution:**
```python
df = reverse_centerline_order(df)
display_centerline_endpoints(df, frame=20, display=True)  # Verify fix
```

### Issue 2: "Plant is upside down"

**Symptoms:** Plant grows downward instead of upward in plots

**Solution:**
```python
df = flip_centerlines(df, direction='vertical')
plot_centerline_data(df)  # Verify fix
```

### Issue 3: "Plant points wrong direction"

**Symptoms:** Gravitropic bending goes left instead of expected right

**Solution:**
```python
df = flip_centerlines(df, direction='horizontal')
plot_centerline_data(df)  # Verify fix
```

### Issue 4: "Units are still in pixels after conversion"

**Symptoms:** Column names don't change to 'x (meters)', 'y (meters)'

**Check:**
```python
# Make sure conversion factors are correct
print(f"px_to_m = {35 / 1_000_000}")  # Should be ~3.5e-5
print(f"frame_to_s = {9 * 60}")       # Should be 540

# Verify column names after conversion
print(df.columns)  # Should see 'x (meters)', 'y (meters)', 'time (seconds)'
```

### Issue 5: "Arc-length conversion gives unexpected values"

**Symptoms:** s values are huge or tiny, angles don't make sense

**Root cause:** Usually a units issue. Make sure you:
1. Convert to meters **before** calling `x_y_to_s_theta()`
2. Don't mix pixels and meters

```python
# WRONG: mixing units
df_pixels = pd.read_csv('data.csv')
s_theta = x_y_to_s_theta(df_pixels)  # Will give s in pixels!

# RIGHT: convert first
df_meters = convert_centerline_units(df_pixels, px_to_m=35/1e6, frame_to_s=540)
s_theta = x_y_to_s_theta(df_meters)  # Now s is in meters
```

---

## Next Steps

With properly oriented and calibrated data, you're ready for analysis:

- [Finding β̃ and γ̃](/guides/finding-beta-gamma/) - Extract gravitropic parameters
- [Growth Analysis](/guides/growth-analysis/) - Analyze elongation dynamics
- [Geometric Analysis](/guides/geometric-analysis/) - Calculate angles and curvatures
- [Visualization](/guides/visualization/) - Create publication-quality plots
