---
title: Quick Start
description: Analyze your first plant tropism experiment in 5 minutes.
---

This guide shows you how to extract physical constants from a single plant gravitropism experiment. You'll go from raw centerline data to gamma (γ) and beta (β) in minutes.

## What You Need

- Tropism Toolset installed ([Installation Guide](/guides/installation/))
- A centerline CSV file with columns: `frame`, `x`, `y`
- Two calibration values: pixels-to-meters and minutes-per-frame

:::tip[Getting Centerline Data]
Extract centerlines from your timelapse images using [SAP (Segmentation App)](https://github.com/merozlab/plant-segmentation-app) or ImageJ/Fiji.
:::

## The 5-Minute Analysis

Here's a complete analysis in one script. Copy this and adjust the configuration values:

```python
import pandas as pd
from tropism_toolset import (
    convert_centerline_units,
    fit_Lc,
    find_steady_state,
    get_gamma,
    get_beta,
    get_lengths_from_centerlines
)

# ============================================
# CONFIGURATION - Edit these values
# ============================================
data_file = "my_experiment.csv"
px_to_m = 0.0001          # Your calibration: meters per pixel
minutes_per_frame = 15    # Time between frames

# ============================================
# LOAD DATA
# ============================================
data = pd.read_csv(data_file)
print(f"✓ Loaded {data['frame'].nunique()} frames")

# Convert to physical units
period = minutes_per_frame * 60  # Convert to seconds
data = convert_centerline_units(data, px_to_m=px_to_m, frame_to_s=period)

# ============================================
# FIND STEADY STATE (Tc)
# ============================================
length_df = get_lengths_from_centerlines(data)
Tc, _ = find_steady_state(
    length_df["length (meters)"].values,
    show=True
)
print(f"✓ Steady state at frame {Tc} ({Tc * minutes_per_frame:.0f} min)")

# ============================================
# FIT CONVERGENCE LENGTH (Lc)
# ============================================
x0, Bl, A, Lc, r_squared = fit_Lc(
    data,
    display=True,
    crop_end=3  # Remove noisy tip points
)
print(f"✓ Convergence length: {Lc:.4f} m (R² = {r_squared:.3f})")

# ============================================
# CALCULATE CONSTANTS
# ============================================
gamma = get_gamma(Tc, period)
beta = get_beta(Lc, gamma)

# ============================================
# RESULTS
# ============================================
print("\n" + "="*60)
print("GRAVITROPISM CONSTANTS")
print("="*60)
print(f"Convergence Length (Lc):        {Lc*100:.2f} cm")
print(f"Proprioceptive Sensitivity (γ): {gamma:.6e} s⁻¹")
print(f"Gravitropic Sensitivity (β):    {beta:.4f} m⁻¹")
print(f"Characteristic Time (1/γ):      {1/gamma/3600:.2f} hours")
print("="*60)
```

## Understanding the Output

The script will show you two plots:

1. **Steady State Detection**: Shows when the plant's growth enters steady state
2. **Lc Fit**: Shows the angle profile along the plant and the exponential fit

And print three key physical constants:

| Constant | Symbol | Meaning |
|----------|--------|---------|
| Convergence Length | $L_c$ | Length scale of gravitropic response |
| Proprioceptive Sensitivity | $\gamma$ | How fast the plant reorients (s⁻¹) |
| Gravitropic Sensitivity | $\beta$ | Spatial sensitivity to gravity (m⁻¹) |

**Good results:** R² > 0.95 indicates a good fit. If lower, try adjusting `crop_end` or check your data quality.

## What Just Happened?

1. **Loaded data**: Read your CSV with (frame, x, y) coordinates
2. **Converted units**: Transformed pixels → meters and frames → seconds
3. **Found Tc**: Detected when the plant reached steady-state bending
4. **Fitted Lc**: Measured the convergence length from the angle profile
5. **Calculated γ and β**: Derived the fundamental tropism constants

## Troubleshooting

**Poor fit (R² < 0.90)?**
- Try different `crop_end` values (1-5) to exclude noisy tip regions
- Check that your centerline points go from base to tip consistently

**Can't find steady state?**
- Manually set `Tc` to the frame where bending stabilizes
- Your experiment may need more frames to reach steady state

**Import errors?**
- Verify installation: `pip show tropism-toolset`
- Check you're using the correct Python environment

## Next Steps

Now that you've run a basic analysis:

- **Batch processing**: [Complete Workflows](/examples/workflows/) shows how to process multiple plants
- **Understanding the theory**: [Core Concepts](/guides/concepts/) explains the mathematical models
- **Visualization**: [Visualization Guide](/guides/visualization/) for publication-quality figures
- **Advanced fitting**: [Growth Analysis](/guides/growth-analysis/) for temporal dynamics
