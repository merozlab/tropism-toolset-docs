---
title: Growth Analysis
description: Analyze growth rates, angular velocities, and temporal dynamics.
---

Extract and analyze growth dynamics including elongation rates, angular velocities, and growth zone characteristics.

## Growth Rate Analysis

### Extracting Growth Rate (dL/dt)

Growth rate is the rate of organ elongation over time:

```python
from constants import fit_growth_rate, get_arclengths
import pandas as pd
import numpy as np

# Load data
data = pd.read_csv("centerlines.csv")
px_to_m = 100

# Extract length over time
frames = []
lengths = []

for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]
    arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters
    total_length = arclengths[-1]

    frames.append(frame)
    lengths.append(total_length)

frames = np.array(frames)
lengths = np.array(lengths)

# Fit linear growth model
growth_rate = fit_growth_rate(
    frames,
    lengths,
    show=True,
    ylabel="Length (m)",
    title="Organ Growth Over Time",
    slope_label="growth rate",
    slope_units="m/frame"
)

print(f"Growth rate: {growth_rate:.6f} m/frame")

# Convert to biological units
minutes_per_frame = 15
growth_rate_per_hour = growth_rate * (60 / minutes_per_frame)
print(f"Growth rate: {growth_rate_per_hour:.6f} m/hour")
```

### Piecewise Growth Analysis

For two-phase growth (fast then slow):

```python
from constants.fitting import fit_piecewise_linear_continuous

# Fit piecewise linear model
slope1, slope2, breakpoint, (p1_coeffs, p2_coeffs) = fit_piecewise_linear_continuous(
    frames,
    lengths,
    show=True
)

print(f"Phase 1 growth rate: {slope1:.6f} m/frame")
print(f"Phase 2 growth rate: {slope2:.6f} m/frame")
print(f"Transition at frame: {breakpoint:.1f}")
```

## Angular Velocity Analysis

### Tip Angular Velocity (dθ/dt)

Track how quickly the tip reorients:

```python
from constants import fit_angular_velocity, get_angles

# Extract tip angle over time
frames = []
tip_angles = []

for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]
    angles = get_angles(frame_data)
    tip_angles.append(angles[-1])  # Last angle is tip
    frames.append(frame)

frames = np.array(frames)
tip_angles = np.array(tip_angles)

# Fit angular velocity
angular_velocity = fit_angular_velocity(
    frames,
    tip_angles,
    show=True,
    title="Tip Reorientation Dynamics"
)

print(f"Angular velocity: {angular_velocity:.6f} rad/frame")

# Convert to degrees per hour
degrees_per_hour = np.degrees(angular_velocity) * (60 / minutes_per_frame)
print(f"Angular velocity: {degrees_per_hour:.2f} °/hour")
```

### Calculating Gravitropic Sensitivity (β̃)

Using the Chauvet model:

```python
# Calculate relative differential growth parameter
R = 0.00145  # Plant radius in meters (1.45 mm)

# dθ/dt / dL/dt
beta_tilde = R * angular_velocity / growth_rate

print(f"Gravitropic sensitivity (β̃): {beta_tilde:.4f} m⁻¹")
```

## Growth Zone Analysis

### Identifying the Growth Zone

Compare initial and final organ shapes:

```python
from constants import get_angles, get_arclengths

# Get first and last frames
first_frame = data[data['frame'] == data['frame'].min()]
last_frame = data[data['frame'] == data['frame'].max()]

# Calculate lengths
lengths_first = get_arclengths(first_frame) / px_to_m  # Convert to meters
lengths_last = get_arclengths(last_frame) / px_to_m  # Convert to meters

# Growth occurred
growth_length = lengths_last[-1] - lengths_first[-1]
print(f"Total growth: {growth_length:.4f} m")

# Growth zone is approximately the region where last > first
# in terms of arc length
print(f"Growth zone: apical ~{growth_length:.4f} m")
```

### Mask-Based Growth Zone

Using binary masks to identify stable (non-growing) regions:

```python
import cv2
import numpy as np
from pathlib import Path

# Load first and last masks
mask_dir = Path("data/experiment_masks")
first_mask = cv2.imread(str(mask_dir / "00001_mask.bmp"), 0)
last_mask = cv2.imread(str(mask_dir / f"{len(list(mask_dir.glob('*.bmp'))):05d}_mask.bmp"), 0)

# Find intersection (stable region)
intersection = cv2.bitwise_and(first_mask, last_mask)

# Count pixels
first_area = np.sum(first_mask > 0)
last_area = np.sum(last_mask > 0)
stable_area = np.sum(intersection > 0)

overlap_percentage = (stable_area / first_area) * 100
print(f"Stable region: {overlap_percentage:.1f}% of initial length")
print(f"Growing region: {100 - overlap_percentage:.1f}%")
```

## Temporal Curve Fitting

### Saturating Exponential Growth

For processes approaching steady state:

```python
from constants.fitting import fit_saturating_exponential

# Fit length approaching L_infinity
y_inf, y_0, tau, r_squared = fit_saturating_exponential(
    frames,
    lengths,
    data_type="length",
    show=True
)

print(f"Final length (L∞): {y_inf:.4f} m")
print(f"Initial length (L₀): {y_0:.4f} m")
print(f"Time constant (τ): {tau:.2f} frames = {tau * minutes_per_frame:.1f} min")
print(f"R²: {r_squared:.4f}")
```

### Logistic Growth

For S-shaped growth curves:

```python
from constants.fitting import fit_logistic_growth

# Fit logistic model
K, y_0, r, t_m, r_squared = fit_logistic_growth(
    frames,
    lengths,
    data_type="length",
    show=True
)

print(f"Carrying capacity (K): {K:.4f} m")
print(f"Growth rate (r): {r:.6f}")
print(f"Inflection point (t_m): {t_m:.2f} frames")
print(f"R²: {r_squared:.4f}")
```

## Batch Growth Analysis

Process multiple experiments automatically:

```python
from pathlib import Path
import pandas as pd

# Directory with multiple experiments
experiment_dir = Path("data/full_experiment")

results = []

for exp_folder in experiment_dir.iterdir():
    if not exp_folder.is_dir():
        continue

    exp_name = exp_folder.name
    centerlines_file = exp_folder / f"{exp_name}_centerlines.csv"

    if not centerlines_file.exists():
        print(f"⚠ Skipping {exp_name}: no centerlines file")
        continue

    print(f"Processing: {exp_name}")

    # Load data
    data = pd.read_csv(centerlines_file)

    # Extract lengths
    frames_exp = []
    lengths_exp = []
    for frame in sorted(data['frame'].unique()):
        frame_data = data[data['frame'] == frame]
        arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters
        frames_exp.append(frame)
        lengths_exp.append(arclengths[-1])

    # Fit growth rate
    growth_rate = fit_growth_rate(
        np.array(frames_exp),
        np.array(lengths_exp),
        show=False
    )

    # Store results
    results.append({
        'experiment': exp_name,
        'initial_length': lengths_exp[0],
        'final_length': lengths_exp[-1],
        'total_growth': lengths_exp[-1] - lengths_exp[0],
        'growth_rate': growth_rate,
        'num_frames': len(frames_exp)
    })

# Create summary
summary_df = pd.DataFrame(results)
print("\n" + summary_df.to_string())
summary_df.to_csv("growth_analysis_summary.csv", index=False)
```

## Visualizing Growth Dynamics

### Growth Rate vs Time

```python
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Smooth the length data
lengths_smooth = savgol_filter(lengths, window_length=11, polyorder=2)

# Calculate instantaneous growth rate
dt = np.diff(frames)
dL = np.diff(lengths_smooth)
instantaneous_rate = dL / dt

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(frames, lengths, 'o', alpha=0.5, label='Data')
ax1.plot(frames, lengths_smooth, '-', label='Smoothed')
ax1.set_ylabel('Length (m)')
ax1.set_title('Organ Length Over Time')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(frames[1:], instantaneous_rate, 'o-')
ax2.axhline(growth_rate, color='r', linestyle='--', label=f'Mean: {growth_rate:.6f} m/frame')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Growth Rate (m/frame)')
ax2.set_title('Instantaneous Growth Rate')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Growth Distribution

Compare growth rates across experiments:

```python
import matplotlib.pyplot as plt

growth_rates = summary_df['growth_rate'].values

plt.figure(figsize=(8, 6))
plt.hist(growth_rates, bins=20, alpha=0.7, edgecolor='black')
plt.axvline(growth_rates.mean(), color='r', linestyle='--',
            label=f'Mean: {growth_rates.mean():.6f}')
plt.xlabel('Growth Rate (m/frame)')
plt.ylabel('Frequency')
plt.title('Growth Rate Distribution Across Experiments')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print(f"Mean growth rate: {growth_rates.mean():.6f} m/frame")
print(f"Std deviation: {growth_rates.std():.6f} m/frame")
```

## Relating Growth to Constants

### Growth Zone Length

```python
# Estimate growth zone length from Lc and growth data
Lc = 0.042  # From fit_Lc
gamma = 2.3e-4  # From get_gamma

# Growth zone length approximation
L_gz = Lc * 2  # Approximately 2*Lc

print(f"Estimated growth zone length: {L_gz:.4f} m")
print(f"As percentage of total: {(L_gz/lengths[-1])*100:.1f}%")
```

### Relating dL/dt and dθ/dt

```python
# Using the Chauvet model
# β̃ = R * (dθ/dt) / (dL/dt)

R = 0.00145  # meters
beta_tilde_calc = R * angular_velocity / growth_rate

print(f"Calculated β̃: {beta_tilde_calc:.4f} m⁻¹")
```

## Next Steps

- Use growth rate in [calculating physical constants](/guides/constants-extraction/)
- Combine with [steady state detection](/guides/steady-state/) for complete temporal analysis
- Explore [batch processing](/advanced/batch-processing/) for multiple experiments
