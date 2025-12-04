---
title: Quick Start
description: Get started with your first tropism analysis in minutes.
---

This guide walks you through a complete analysis workflow, from loading data to extracting physical constants.

## Prerequisites

- Tropism Toolset installed ([Installation Guide](/guides/installation/))
- Centerline data in CSV format with columns: `frame`, `x`, `y`
- Python environment activated

:::tip[Data Format]
Your CSV should contain centerline coordinates extracted from plant images. We recommend using [SAP (Segmentation App)](https://github.com/merozlab/plant-segmentation-app) for extracting centerlines from images.
:::

## Step 1: Import the Package

```python
import pandas as pd
import numpy as np
from tropism_toolset import (
    get_angles,
    get_arclengths,
    get_angles_over_time,
    get_arclengths_over_time,
    fit_Lc,
    get_gamma,
    get_beta,
    plot_centerline_data
)
```

## Step 2: Load Your Data

```python
# Load centerline data from CSV
data = pd.read_csv("data/experiment_centerlines.csv")

# Check the data structure
print(data.head())
print(f"Total frames: {data['frame'].nunique()}")
print(f"Points per frame: {len(data[data['frame'] == 0])}")
```

Expected output:
```
   frame    x      y
0      0  245  1088
1      0  246  1076
2      0  248  1064
...
Total frames: 453
Points per frame: 96
```

## Step 3: Visualize the Centerlines

```python
# Plot centerlines over time
px_to_m = 100  # Conversion factor: pixels to meters

plot_centerline_data(
    data,
    px_to_length=px_to_m,
    units="meters",
    plant_part="Root",
    time_per_frame=15,  # 15 minutes per frame
    time_unit="minutes",
    show_scale_bar=True
)
```

This creates a time-lapse visualization showing how the plant grows and bends over time.

## Step 4: Calculate Geometric Properties

```python
# For a single frame
single_frame = data[data['frame'] == 0]
angles = get_angles(single_frame, show=True)
arclengths = get_arclengths(single_frame)  # Returns lengths in same units as input

# For all frames
angles_per_frame = get_angles_over_time(data)
arclengths_per_frame = get_arclengths_over_time(data)
# Convert to meters
arclengths_per_frame = [arc / px_to_m for arc in arclengths_per_frame]
```

## Step 5: Extract Convergence Length (Lc)

The convergence length is a key parameter describing the characteristic length scale of the gravitropic response.

```python
# Use data from a specific frame (e.g., the last frame at steady state)
final_frame = data['frame'].max()
final_data = data[data['frame'] == final_frame]

# Get angles and arclengths for this frame
angles = get_angles(final_data)
arclengths = get_arclengths(final_data)
# Convert to meters if needed: arclengths_m = arclengths / px_to_m

# Fit the saturating exponential model
# Returns: (x0, Bl, A, Lc, r_squared)
x0, Bl, A, Lc, r_squared = fit_Lc(
    arclengths,
    angles,
    show=True,
    crop_start=0,
    crop_end=3
)

print(f"\nFitted Parameters:")
print(f"Convergence Length (Lc): {Lc:.4f} m")
print(f"Baseline Angle (Bl): {Bl:.4f} rad")
print(f"Amplitude (A): {A:.4f} rad")
print(f"R²: {r_squared:.4f}")
```

## Step 6: Calculate Physical Constants

```python
# Determine steady state frame (Tc)
# This can be done manually or using find_steady_state()
Tc = 120  # Frame where steady state begins

# Calculate gamma (proprioceptive sensitivity)
period = 15 * 60  # 15 minutes per frame in seconds
gamma = get_gamma(Tc, period)

# Calculate beta (gravitropic sensitivity)
beta = get_beta(Lc, gamma)

print(f"\nPhysical Constants:")
print(f"Gamma (γ): {gamma:.6f} s⁻¹")
print(f"Beta (β): {beta:.4f} m⁻¹")
print(f"Characteristic time: {1/gamma:.2f} s")
```

## Complete Example

Here's the complete workflow in one script:

```python
import pandas as pd
from tropism_toolset import *

# Configuration
data_file = "data/experiment_centerlines.csv"
px_to_m = 100
period = 15 * 60  # seconds per frame
Tc = 120  # steady state frame

# Load data
data = pd.read_csv(data_file)
print(f"Loaded {len(data)} points from {data['frame'].nunique()} frames")

# Visualize
plot_centerline_data(
    data,
    px_to_length=px_to_m,
    units="meters",
    time_per_frame=period/60,
    time_unit="minutes",
    show_scale_bar=True
)

# Analyze final frame
final_frame = data['frame'].max()
final_data = data[data['frame'] == final_frame]
angles = get_angles(final_data)
arclengths = get_arclengths(final_data)
arclengths_m = arclengths / px_to_m  # Convert to meters

# Fit Lc
x0, Bl, A, Lc, r_squared = fit_Lc(arclengths_m, angles, show=True)

# Calculate constants
gamma = get_gamma(Tc, period)
beta = get_beta(Lc, gamma)

# Print results
print("\n" + "="*50)
print("ANALYSIS RESULTS")
print("="*50)
print(f"Convergence Length (Lc): {Lc:.4f} m")
print(f"Proprioceptive Sensitivity (γ): {gamma:.6f} s⁻¹")
print(f"Gravitropic Sensitivity (β): {beta:.4f} m⁻¹")
print(f"Fit Quality (R²): {r_squared:.4f}")
print("="*50)
```

## Understanding the Results

- **Lc (Convergence Length)**: The length scale over which the plant organ responds to gravitropic stimulus (~0.01-0.1 m for roots)
- **γ (Gamma)**: How quickly the plant corrects its orientation (higher = faster response)
- **β (Beta)**: Spatial sensitivity to gravity (β = γ/Lc)
- **R²**: Goodness of fit (>0.95 is excellent)

## Next Steps

Now that you've completed your first analysis:

- Learn about [Core Concepts](/guides/concepts/) to understand the theory
- Explore [Geometric Analysis](/guides/geometric-analysis/) for more detailed spatial analysis
- Read about [Steady State Detection](/guides/steady-state/) for automated Tc determination
- Check out [Growth Analysis](/guides/growth-analysis/) for temporal dynamics

## Common Issues

### "No module named 'constants'"

Make sure you're using the correct Jupyter kernel. See [Installation Guide](/guides/installation/#jupyter-setup).

### Poor Fit Quality (Low R²)

Try adjusting the `crop_start` and `crop_end` parameters in `fit_Lc()` to exclude noisy regions at the tip or base.

### Unexpected Angle Values

Check your angle preset. The toolkit uses mathematical convention by default. See `PRESET` in [geometric_calculations.py](/reference/geometric-calculations/).
