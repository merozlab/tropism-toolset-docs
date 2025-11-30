---
title: Complete Workflows
description: End-to-end analysis examples for common use cases.
---

Complete workflows from data loading to final results for typical tropism analysis scenarios.

## Workflow 1: Basic Gravitropism Analysis

Extract all physical constants from a gravitropism experiment.

### Data Requirements
- Centerline CSV with (frame, x, y) columns
- Binary masks (optional, for steady state detection)
- Calibration: pixels to meters conversion

### Complete Script

```python
import pandas as pd
import numpy as np
from constants import (
    get_angles, get_arclengths,
    get_angles_over_time, get_arclengths_over_time,
    fit_Lc, get_gamma, get_beta,
    find_steady_state,
    plot_centerline_data
)

# ============================================
# 1. CONFIGURATION
# ============================================
data_file = "data/experiment_centerlines.csv"
px_to_m = 100  # pixels per meter
minutes_per_frame = 15
period = minutes_per_frame * 60  # seconds

# ============================================
# 2. LOAD AND VISUALIZE DATA
# ============================================
data = pd.read_csv(data_file)
print(f"Loaded {len(data)} points from {data['frame'].nunique()} frames")

# Visualize centerlines over time
fig, ax = plot_centerline_data(
    data,
    px_to_length=px_to_m,
    units="meters",
    plant_part="Root",
    time_per_frame=minutes_per_frame,
    time_unit="minutes",
    show_scale_bar=True
)

# ============================================
# 3. FIND STEADY STATE (Tc)
# ============================================
# Extract lengths over time
frames = []
lengths = []
for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]
    arclengths = get_arclengths(frame_data, px_to_m)
    frames.append(frame)
    lengths.append(arclengths[-1])

frames = np.array(frames)
lengths = np.array(lengths)

# Detect steady state
Tc, lengths_smooth = find_steady_state(
    lengths,
    threshold_factor=0.15,
    min_steady_duration=20,
    show=True
)

if Tc is None:
    print("⚠ No steady state detected, using manual estimate")
    Tc = 120

print(f"Steady state frame (Tc): {Tc}")
print(f"Time to steady state: {Tc * minutes_per_frame:.1f} minutes")

# ============================================
# 4. EXTRACT CONVERGENCE LENGTH (Lc)
# ============================================
# Use final frame for Lc fitting
final_frame = data['frame'].max()
frame_data = data[data['frame'] == final_frame]

# Calculate angles and arc lengths
angles = get_angles(frame_data)
arclengths = get_arclengths(frame_data, px_to_m)

# Fit Lc
x0, Bl, A, Lc, r_squared = fit_Lc(
    arclengths,
    angles,
    show=True,
    crop_end=3
)

print(f"\nConvergence length (Lc): {Lc:.4f} m")
print(f"Fit quality (R²): {r_squared:.4f}")

# ============================================
# 5. CALCULATE PHYSICAL CONSTANTS
# ============================================
gamma = get_gamma(Tc, period)
beta = get_beta(Lc, gamma)

# ============================================
# 6. RESULTS SUMMARY
# ============================================
print("\n" + "="*70)
print("GRAVITROPISM ANALYSIS RESULTS")
print("="*70)
print(f"Experiment: {data_file}")
print(f"Frames analyzed: {data['frame'].nunique()}")
print(f"Duration: {data['frame'].max() * minutes_per_frame / 60:.1f} hours")
print("-"*70)
print("PHYSICAL CONSTANTS:")
print(f"  Convergence Length (Lc):         {Lc:.4f} m = {Lc*100:.2f} cm")
print(f"  Proprioceptive Sensitivity (γ):  {gamma:.6e} s⁻¹")
print(f"  Gravitropic Sensitivity (β):     {beta:.4f} m⁻¹")
print(f"  Characteristic Time (1/γ):       {1/gamma:.2f} s = {1/gamma/3600:.2f} h")
print(f"  Characteristic Length (1/β):     {1/beta:.4f} m")
print("-"*70)
print("FITTING PARAMETERS:")
print(f"  Steady State Frame (Tc):         {Tc}")
print(f"  Transition Point (x₀):           {x0:.4f} m")
print(f"  Baseline Angle (Bℓ):            {Bl:.4f} rad = {np.degrees(Bl):.2f}°")
print(f"  Amplitude (A):                   {A:.4f} rad = {np.degrees(A):.2f}°")
print(f"  Fit Quality (R²):                {r_squared:.4f}")
print("="*70)

# Save results
results = {
    'Lc_m': Lc,
    'gamma_per_s': gamma,
    'beta_per_m': beta,
    'Tc_frames': Tc,
    'r_squared': r_squared,
    'x0_m': x0,
    'Bl_rad': Bl,
    'A_rad': A
}

results_df = pd.DataFrame([results])
results_df.to_csv("analysis_results.csv", index=False)
print(f"\n✓ Results saved to analysis_results.csv")
```

---

## Workflow 2: Growth Dynamics Analysis

Analyze growth rate and angular velocity using the Chauvet model.

```python
from constants import (
    fit_growth_rate, fit_angular_velocity,
    get_angles, get_arclengths
)

# Load data
data = pd.read_csv("data/experiment.csv")
px_to_m = 100
minutes_per_frame = 15

# Extract time series
frames, lengths, tip_angles = [], [], []

for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]

    arclengths = get_arclengths(frame_data, px_to_m)
    angles = get_angles(frame_data)

    frames.append(frame)
    lengths.append(arclengths[-1])
    tip_angles.append(angles[-1])

frames = np.array(frames)
lengths = np.array(lengths)
tip_angles = np.array(tip_angles)

# Fit growth rate
growth_rate = fit_growth_rate(
    frames,
    lengths,
    show=True,
    ylabel="Length (m)",
    title="Growth Rate Analysis",
    slope_units="m/frame"
)

# Fit angular velocity
angular_velocity = fit_angular_velocity(
    frames,
    tip_angles,
    show=True,
    title="Angular Reorientation"
)

# Chauvet model parameter
R = 0.00145  # Plant radius in meters
beta_tilde = R * angular_velocity / growth_rate

# Convert to biological units
seconds_per_frame = minutes_per_frame * 60
growth_rate_per_sec = growth_rate / seconds_per_frame
angular_vel_per_sec = angular_velocity / seconds_per_frame

print("\nGROWTH DYNAMICS RESULTS:")
print(f"Growth rate: {growth_rate:.6e} m/frame")
print(f"Growth rate: {growth_rate_per_sec*1e6:.2f} μm/s")
print(f"Angular velocity: {angular_velocity:.6e} rad/frame")
print(f"Angular velocity: {np.degrees(angular_vel_per_sec)*3600:.2f} °/hour")
print(f"Chauvet β̃: {beta_tilde:.4f} m⁻¹")
```

---

## Workflow 3: Batch Processing Multiple Experiments

Process multiple experiments and create comparison statistics.

```python
from pathlib import Path
import glob

# Directory with experiments
exp_dir = Path("data/experiments")
results = []

# Process each experiment
for csv_file in glob.glob(str(exp_dir / "*_centerlines.csv")):
    exp_name = Path(csv_file).stem

    print(f"\nProcessing: {exp_name}")
    print("-" * 50)

    try:
        # Load data
        data = pd.read_csv(csv_file)

        # Extract lengths
        lengths = []
        for frame in sorted(data['frame'].unique()):
            frame_data = data[data['frame'] == frame]
            arclengths = get_arclengths(frame_data, px_to_m=100)
            lengths.append(arclengths[-1])

        lengths = np.array(lengths)

        # Find steady state
        Tc, _ = find_steady_state(lengths, show=False)

        # Fit Lc from final frame
        final_data = data[data['frame'] == data['frame'].max()]
        angles = get_angles(final_data)
        arclengths = get_arclengths(final_data, px_to_m=100)
        x0, Bl, A, Lc, r_squared = fit_Lc(arclengths, angles, show=False)

        # Calculate constants
        gamma = get_gamma(Tc, period=15*60)
        beta = get_beta(Lc, gamma)

        # Store results
        results.append({
            'experiment': exp_name,
            'Lc_m': Lc,
            'gamma_s-1': gamma,
            'beta_m-1': beta,
            'Tc_frames': Tc,
            'r_squared': r_squared
        })

        print(f"  ✓ Lc = {Lc:.4f} m, γ = {gamma:.6e} s⁻¹, R² = {r_squared:.3f}")

    except Exception as e:
        print(f"  ✗ Failed: {e}")

# Create summary
summary = pd.DataFrame(results)

# Statistics
print("\n" + "="*60)
print("BATCH ANALYSIS SUMMARY")
print("="*60)
print(f"Total experiments: {len(results)}")
print(f"Successful: {len(summary)}")
print("\nMean ± Std:")
print(f"Lc:    {summary['Lc_m'].mean():.4f} ± {summary['Lc_m'].std():.4f} m")
print(f"Gamma: {summary['gamma_s-1'].mean():.6e} ± {summary['gamma_s-1'].std():.6e} s⁻¹")
print(f"Beta:  {summary['beta_m-1'].mean():.4f} ± {summary['beta_m-1'].std():.4f} m⁻¹")
print("="*60)

# Save
summary.to_csv("batch_results.csv", index=False)
print(f"\n✓ Saved to batch_results.csv")
```

---

## Workflow 4: Mask-Based Steady State

Use binary masks for steady state detection.

```python
from constants import find_steady_state_from_masks, create_video_with_colored_frames
from pathlib import Path

# Mask directory
mask_dir = Path("data/experiment_masks")

# Find steady state from masks
Tc, stability_scores = find_steady_state_from_masks(
    mask_dir,
    method="overlap",
    threshold_factor=0.15,
    min_steady_duration=20,
    show=True
)

print(f"Mask-based Tc: {Tc}")

# Create annotated video
create_video_with_colored_frames(
    mask_dir=mask_dir,
    output_file="steady_state_analysis.mp4",
    Tc=Tc,
    framerate=6
)

print("✓ Video created: steady_state_analysis.mp4")
```

---

## Workflow 5: Publication Figure Generation

Create a comprehensive multi-panel figure.

```python
import matplotlib.pyplot as plt

# Set publication style
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 10,
    'figure.dpi': 300,
})

# Create figure
fig = plt.figure(figsize=(12, 8))

# A: Centerlines
ax1 = plt.subplot(2, 3, 1)
plot_centerline_data(data, px_to_length=px_to_m, show_scale_bar=True)
ax1.text(-0.15, 1.05, 'A', transform=ax1.transAxes,
         fontsize=16, fontweight='bold')
ax1.set_title('Centerline Evolution')

# B: Lc Fit
ax2 = plt.subplot(2, 3, 2)
# ... Lc fitting plot
ax2.text(-0.15, 1.05, 'B', transform=ax2.transAxes,
         fontsize=16, fontweight='bold')
ax2.set_title('Convergence Length')

# C: Length over time
ax3 = plt.subplot(2, 3, 3)
ax3.plot(frames, lengths, 'o-', markersize=3)
ax3.axvline(Tc, color='r', linestyle='--', alpha=0.5)
ax3.set_xlabel('Frame')
ax3.set_ylabel('Length (m)')
ax3.text(-0.15, 1.05, 'C', transform=ax3.transAxes,
         fontsize=16, fontweight='bold')
ax3.set_title('Growth Dynamics')
ax3.grid(alpha=0.3)

# D-F: Additional panels...

plt.tight_layout()
plt.savefig('figure_1.png', dpi=300, bbox_inches='tight')
plt.savefig('figure_1.pdf', bbox_inches='tight')
print("✓ Figure saved as PNG and PDF")
```

---

## Next Steps

- Explore [Batch Processing](/advanced/batch-processing/) for automated workflows
- Review [API Reference](/reference/geometric-calculations/) for advanced options
- Check [Visualization Guide](/guides/visualization/) for plot customization
