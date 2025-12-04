---
title: Geometric Analysis
description: Calculate angles, arclengths, and curvatures from centerline data.
---

Transform centerline coordinates into geometric descriptors needed for tropism analysis.

## Computing Angles

Angles represent the local tangent direction along the plant organ.

### Single Frame Analysis

```python
from tropism_toolset import get_angles
import pandas as pd

# Load and select a frame
data = pd.read_csv("centerlines.csv")
frame_data = data[data['frame'] == 0]

# Calculate angles
angles = get_angles(frame_data, show=True)

print(f"Calculated {len(angles)} angles")
print(f"Range: [{angles.min():.3f}, {angles.max():.3f}] rad")
```

The `show=True` parameter displays a plot with angles in terms of π.

### Angle Calculation Method

Angles are computed from coordinate differences:

```python
import numpy as np

# Manually calculate for understanding
x = frame_data['x'].values
y = frame_data['y'].values

dx = np.diff(x)
dy = np.diff(y)
angles = np.arctan2(dy, dx)
```

:::tip[Angle Presets]
The toolkit supports two angle conventions:
- **mathematical** (default): Standard `arctan2(dy, dx)`
- **mathieu**: Transformed as `π/2 - arctan2(dy, dx)`

Set via `PRESET` variable in `geometric_calculations.py`
:::

### Multi-Frame Analysis

```python
from tropism_toolset import get_angles_over_time

# Calculate angles for all frames
angles_per_frame, unique_frames = get_angles_over_time(data)

# Access specific frame
frame_num = 50
angles_frame_50 = angles_per_frame[frame_num]

print(f"Processed {len(unique_frames)} frames")
```

### Spatial Slicing Before Angle Calculation

```python
from tropism_toolset.geometric_calculations import apply_spatial_slicing

# Analyze only the tip region
tip_data = data.groupby('frame').apply(
    lambda g: apply_spatial_slicing(g, percent=30, side='tip')
).reset_index(drop=True)

angles_per_frame_tip, _ = get_angles_over_time(tip_data)
```

## Computing Arc Lengths

Arc length is the cumulative distance along the centerline from base to tip.

### Single Frame

```python
from tropism_toolset import get_arclengths

px_to_m = 100  # Calibration factor

# Calculate cumulative arc length (in same units as input)
arclengths_px = get_arclengths(frame_data)
arclengths = arclengths_px / px_to_m  # Convert to meters

# Total length
total_length = arclengths[-1]
print(f"Total organ length: {total_length:.4f} m")
```

### Multiple Frames

```python
from tropism_toolset import get_arclengths_over_time

arclengths_per_frame = get_arclengths_over_time(data)
# Convert to meters
arclengths_per_frame = [arc / px_to_m for arc in arclengths_per_frame]

# Track length over time
lengths = [arc[-1] for arc in arclengths_per_frame]

import matplotlib.pyplot as plt
plt.plot(unique_frames, lengths, 'o-')
plt.xlabel('Frame')
plt.ylabel('Length (m)')
plt.title('Organ Length Over Time')
plt.grid(alpha=0.3)
plt.show()
```

## Curvature Calculation

Curvature κ = dθ/ds describes how quickly the angle changes with arc length.

```python
import numpy as np
from tropism_toolset import get_angles, get_arclengths

# Get angles and arc lengths
angles = get_angles(frame_data)
arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters

# Compute curvature (numerical derivative)
# Note: angles has one less point than arclengths due to diff
ds = np.diff(arclengths)
dtheta = np.diff(angles)
curvature = dtheta / ds

# Plot curvature along the organ
s_mid = (arclengths[1:-1] + arclengths[2:]) / 2
plt.plot(s_mid, curvature, 'o-')
plt.xlabel('Arc Length s (m)')
plt.ylabel('Curvature κ (m⁻¹)')
plt.title('Curvature Along Organ')
plt.grid(alpha=0.3)
plt.show()
```

:::caution
Curvature calculation amplifies noise. Consider [smoothing the data](/guides/data-handling/#data-smoothing) first.
:::

## Coordinate Transformations

### Cartesian → Arc-Length

Transform (x, y) to (s, θ) coordinates:

```python
from tropism_toolset import x_y_to_s_theta

x = frame_data['x'].values
y = frame_data['y'].values

s, theta = x_y_to_s_theta(x, y, px_to_m)

# Create DataFrame with new coordinates
import pandas as pd
stheta_data = pd.DataFrame({
    's': s,
    'theta': theta
})
```

### Arc-Length → Cartesian

Reconstruct (x, y) from (s, θ):

```python
from tropism_toolset import s_theta_to_xy

# Need initial position
x0, y0 = x[0], y[0]

x_recon, y_recon = s_theta_to_xy(s, theta, x0, y0)

# Verify reconstruction
error = np.sqrt((x - x_recon)**2 + (y - y_recon)**2)
print(f"Reconstruction error: {error.mean():.6f} m (mean)")
```

## Angle vs Arc Length Plots

A fundamental visualization in tropism analysis:

```python
from tropism_toolset import plot_theta_vs_arclength_over_time

# Calculate for all frames
angles_per_frame = get_angles_over_time(data)
arclengths_per_frame = get_arclengths_over_time(data)
# Convert to meters
arclengths_per_frame = [arc / px_to_m for arc in arclengths_per_frame]

# Plot evolution over time
plot_theta_vs_arclength_over_time(
    angles_per_frame,
    arclengths_per_frame,
    px_to_length=px_to_m,
    title="Gravitropic Response Over Time"
)
```

This creates a multi-frame plot showing how the angle profile evolves.

## Analyzing Initial Conditions

Extract base angle (initial orientation):

```python
from tropism_toolset.geometric_calculations import calculate_initial_base_angle

# Base angle from first few points
n_points = 5
base_angle = calculate_initial_base_angle(frame_data, n_points)

print(f"Base angle: {base_angle:.4f} rad ({np.degrees(base_angle):.2f}°)")
```

## Growth Zone Identification

Identify regions of active growth based on geometric changes:

```python
# Compare first and last frames
first_frame = data[data['frame'] == data['frame'].min()]
last_frame = data[data['frame'] == data['frame'].max()]

angles_first = get_angles(first_frame)
angles_last = get_angles(last_frame)

# Ensure same length for comparison
min_len = min(len(angles_first), len(angles_last))
angle_change = angles_last[:min_len] - angles_first[:min_len]

arclengths = get_arclengths(first_frame) / px_to_m  # Convert to meters

plt.plot(arclengths[1:min_len+1], np.abs(angle_change), 'o-')
plt.xlabel('Arc Length s (m)')
plt.ylabel('|Δθ| (rad)')
plt.title('Angle Change (Growth Zone)')
plt.grid(alpha=0.3)
plt.show()
```

## Practical Examples

### Example 1: Tip Angle Tracking

```python
# Track tip angle over time
tip_angles = []
for frame in data['frame'].unique():
    frame_data = data[data['frame'] == frame]
    angles = get_angles(frame_data)
    tip_angles.append(angles[-1])  # Last angle = tip

plt.plot(data['frame'].unique(), tip_angles, 'o-')
plt.xlabel('Frame')
plt.ylabel('Tip Angle θ_tip (rad)')
plt.title('Tip Reorientation Over Time')
plt.grid(alpha=0.3)
plt.show()
```

### Example 2: Base vs Tip Comparison

```python
# Analyze base and tip separately
base_data = data.groupby('frame').apply(
    lambda g: apply_spatial_slicing(g, percent=30, side='base')
).reset_index(drop=True)

tip_data = data.groupby('frame').apply(
    lambda g: apply_spatial_slicing(g, percent=30, side='tip')
).reset_index(drop=True)

# Calculate mean angles
base_angles = []
tip_angles = []

for frame in data['frame'].unique():
    b_frame = base_data[base_data['frame'] == frame]
    t_frame = tip_data[tip_data['frame'] == frame]

    if len(b_frame) > 0:
        base_angles.append(get_angles(b_frame).mean())
    if len(t_frame) > 0:
        tip_angles.append(get_angles(t_frame).mean())

# Plot comparison
frames = data['frame'].unique()
plt.plot(frames, base_angles, 'o-', label='Base')
plt.plot(frames, tip_angles, 's-', label='Tip')
plt.xlabel('Frame')
plt.ylabel('Mean Angle (rad)')
plt.legend()
plt.title('Base vs Tip Dynamics')
plt.grid(alpha=0.3)
plt.show()
```

## Common Issues

### Length Mismatch

Angles have one fewer element than coordinates:

```python
x = frame_data['x'].values  # N points
angles = get_angles(frame_data)  # N-1 angles
arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters, N-1 lengths

# To match: angles and arclengths have same length (N-1)
arclengths_matched = arclengths[1:]  # N-1 lengths
# Now len(angles) == len(arclengths_matched)
```

### Noisy Angle Calculations

If angle plots show high-frequency noise:

```python
from tropism_toolset.fitting import smooth_centerlines

# Smooth coordinates first
smoothed_data = smooth_centerlines(data, window_length=5, polyorder=2)

# Then calculate angles
angles_smooth = get_angles(smoothed_data[smoothed_data['frame'] == 0])
```

## Next Steps

- Use geometric data to [extract physical constants](/guides/constants-extraction/)
- Create [visualizations](/guides/visualization/) of your geometric analysis
- Apply [growth analysis](/guides/growth-analysis/) to temporal geometric data
