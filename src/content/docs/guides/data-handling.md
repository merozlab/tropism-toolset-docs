---
title: Working with Data
description: Load, slice, transform, and prepare your data for analysis.
---

Learn how to load centerline data, perform spatial and temporal slicing, convert between coordinate systems, and smooth noisy data.

## Loading Centerline Data

### Basic Loading

The most common data format is a CSV file with three columns:

```python
import pandas as pd

# Load centerline data
data = pd.read_csv("experiment_centerlines.csv")

# Inspect the data
print(data.head())
print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
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

**Requirements:**
- `frame`: Integer frame number (0-indexed or sequential)
- `x`: Horizontal coordinate (pixels or meters)
- `y`: Vertical coordinate (pixels or meters)
- Points ordered from base to tip within each frame

### Column Name Variations

The toolkit automatically detects column names with units:

```python
# All of these are supported:
# x, y                    - plain column names
# x (meters), y (meters)  - with unit labels
# x (m), y (m)            - short unit labels
# x (pixels), y (pixels)  - pixel coordinates
# x (px), y (px)          - short pixel labels

from constants.geometric_calculations import infer_columns_and_units

x_col, y_col, units = infer_columns_and_units(data)
print(f"Detected: {x_col}, {y_col}, units={units}")
```

## Spatial Slicing

Often you want to analyze only part of the organ (e.g., just the tip or just the base).

### Percentage-Based Slicing

Analyze a percentage of the organ from either end:

```python
from constants.geometric_calculations import apply_spatial_slicing

# Get a single frame
frame_data = data[data['frame'] == 0]

# Analyze the apical 30% (tip region)
tip_30pct = apply_spatial_slicing(
    frame_data,
    percent=30,
    side='tip'
)

# Analyze the basal 50% (base region)
base_50pct = apply_spatial_slicing(
    frame_data,
    percent=50,
    side='base'
)
```

### Index-Based Slicing

For precise control, use explicit indices:

```python
# Analyze points 10 through 80
subset = apply_spatial_slicing(
    frame_data,
    indices=(10, 80)
)

# Or use standard pandas slicing
subset = frame_data.iloc[10:80]
```

### Applying to All Frames

To slice all frames consistently:

```python
# Apply tip slicing to all frames
sliced_data = data.groupby('frame').apply(
    lambda group: apply_spatial_slicing(group, percent=30, side='tip')
).reset_index(drop=True)
```

## Temporal Slicing

Select specific time ranges for analysis:

```python
# Analyze frames 0 through 100
early_frames = data[data['frame'] <= 100]

# Analyze frames after steady state (e.g., frame 120 onwards)
steady_state_data = data[data['frame'] >= 120]

# Analyze a specific range
growth_phase = data[(data['frame'] >= 50) & (data['frame'] <= 150)]

# Every Nth frame (reduce temporal resolution)
every_10th = data[data['frame'] % 10 == 0]
```

## Coordinate System Transformations

### Cartesian to Arc-Length Coordinates

Transform from (x, y) to (s, θ) for analysis:

```python
from constants import x_y_to_s_theta
import numpy as np

# Get single frame coordinates
frame_data = data[data['frame'] == 0]
x = frame_data['x'].values
y = frame_data['y'].values

# Convert to arc-length coordinates
px_to_m = 100  # calibration factor
s, theta = x_y_to_s_theta(x, y, px_to_m)

print(f"Arc length range: {s.min():.4f} to {s.max():.4f} m")
print(f"Angle range: {theta.min():.4f} to {theta.max():.4f} rad")
```

### Arc-Length to Cartesian Coordinates

Convert back to Cartesian for visualization:

```python
from constants import s_theta_to_xy

# Reconstruct coordinates from s, theta
x_reconstructed, y_reconstructed = s_theta_to_xy(s, theta, x[0], y[0])

# Compare with original
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o-', label='Original', alpha=0.6)
plt.plot(x_reconstructed, y_reconstructed, 's--', label='Reconstructed', alpha=0.6)
plt.legend()
plt.axis('equal')
plt.show()
```

## Data Smoothing

Centerline extraction can be noisy. Smooth the data to improve analysis quality.

### Savitzky-Golay Filter

The toolkit provides frame-by-frame smoothing:

```python
from constants.fitting import smooth_centerlines

# Smooth all frames
smoothed_data = smooth_centerlines(
    data,
    window_length=5,  # Must be odd
    polyorder=2       # Polynomial order
)

# Compare original vs smoothed
import matplotlib.pyplot as plt
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

frame_num = 0
original = data[data['frame'] == frame_num]
smoothed = smoothed_data[smoothed_data['frame'] == frame_num]

ax1.plot(original['x'], original['y'], 'o-', label='Original', alpha=0.6)
ax1.set_title('Original')
ax1.axis('equal')

ax2.plot(smoothed['x'], smoothed['y'], 'o-', label='Smoothed', alpha=0.6)
ax2.set_title('Smoothed')
ax2.axis('equal')

plt.tight_layout()
plt.show()
```

:::caution[Smoothing Parameters]
- `window_length`: Larger values = more smoothing but may remove real features
- `polyorder`: Higher values preserve shape better but may preserve noise
- Always visually verify that smoothing doesn't remove biological features
:::

### When to Smooth

**Smooth when:**
- Centerline extraction is noisy
- You see high-frequency oscillations in angle plots
- Derivative calculations (curvature) are unstable

**Don't smooth when:**
- Data is already clean
- You need to preserve fine-scale features (e.g., undulations)
- Analyzing high-frequency dynamics

## Unit Conversions

### Pixel to Physical Units

Always calibrate your spatial measurements:

```python
# Method 1: Known scale bar
scale_bar_pixels = 100
scale_bar_mm = 10
px_to_m = scale_bar_pixels / (scale_bar_mm / 1000)  # pixels per meter

# Method 2: Known object size
known_diameter_mm = 2.9  # Plant radius
measured_diameter_px = 290
px_to_m = measured_diameter_px / (known_diameter_mm / 1000)

print(f"Calibration: {px_to_m:.2f} pixels per meter")

# Apply to get arclengths in meters
from constants import get_arclengths
arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters
```

### Time Conversions

Convert frame numbers to actual time:

```python
# Frame interval
minutes_per_frame = 15
seconds_per_frame = minutes_per_frame * 60

# Convert frame number to time
frame_number = 120
time_minutes = frame_number * minutes_per_frame
time_hours = time_minutes / 60
time_seconds = frame_number * seconds_per_frame

print(f"Frame {frame_number}:")
print(f"  = {time_minutes} minutes")
print(f"  = {time_hours:.2f} hours")
print(f"  = {time_seconds} seconds")
```

## Data Quality Checks

### Check Frame Continuity

```python
# Verify frames are sequential
frames = data['frame'].unique()
missing_frames = set(range(frames.min(), frames.max() + 1)) - set(frames)

if missing_frames:
    print(f"Warning: Missing frames {missing_frames}")
else:
    print("✓ All frames present")
```

### Check Point Count Consistency

```python
# Count points per frame
points_per_frame = data.groupby('frame').size()

print(f"Points per frame:")
print(f"  Mean: {points_per_frame.mean():.1f}")
print(f"  Std: {points_per_frame.std():.1f}")
print(f"  Min: {points_per_frame.min()}")
print(f"  Max: {points_per_frame.max()}")

# Flag frames with unusual point counts
mean_points = points_per_frame.mean()
std_points = points_per_frame.std()
outlier_frames = points_per_frame[
    (points_per_frame < mean_points - 2*std_points) |
    (points_per_frame > mean_points + 2*std_points)
]

if len(outlier_frames) > 0:
    print(f"\nWarning: Unusual point counts in frames:")
    print(outlier_frames)
```

### Visualize Data Coverage

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 4))

# Plot data availability
for frame in data['frame'].unique():
    frame_data = data[data['frame'] == frame]
    ax.scatter([frame] * len(frame_data), range(len(frame_data)),
               s=1, alpha=0.5, c='blue')

ax.set_xlabel('Frame')
ax.set_ylabel('Point Index')
ax.set_title('Data Coverage Map')
plt.tight_layout()
plt.show()
```

## Saving Processed Data

After preprocessing, save for later use:

```python
# Save smoothed data
smoothed_data.to_csv("experiment_centerlines_smoothed.csv", index=False)

# Save sliced region
tip_data.to_csv("experiment_tip_region.csv", index=False)

# Save with arc-length coordinates
frame_data['s'] = s
frame_data['theta'] = theta
frame_data.to_csv("experiment_s_theta.csv", index=False)
```

## Working with Mask Data

Load and process binary mask images for growth zone analysis:

```python
import cv2
from pathlib import Path
import glob

# Load all masks
mask_dir = Path("data/experiment_masks")
mask_files = sorted(glob.glob(str(mask_dir / "*.bmp")))

print(f"Found {len(mask_files)} mask files")

# Load a specific mask
mask = cv2.imread(mask_files[0], cv2.IMREAD_GRAYSCALE)
print(f"Mask shape: {mask.shape}")
print(f"Unique values: {np.unique(mask)}")

# Display mask
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')
plt.axis('off')
plt.show()
```

## Next Steps

- Learn [Geometric Analysis](/guides/geometric-analysis/) to calculate angles and arclengths
- Explore [Visualization](/guides/visualization/) options for your data
- Understand [Steady State Detection](/guides/steady-state/) for temporal analysis
