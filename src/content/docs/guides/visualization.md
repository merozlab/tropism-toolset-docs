---
title: Visualization
description: Create publication-ready plots and analysis videos.
---

The toolkit provides comprehensive visualization functions for all aspects of tropism analysis.

## Centerline Visualization

### Basic Centerline Plot

```python
from tropism_toolset import plot_centerline_data
import pandas as pd

data = pd.read_csv("centerlines.csv")

fig, ax = plot_centerline_data(
    data,
    px_to_length=100,
    units="meters"
)
```

### With Time Information

```python
fig, ax = plot_centerline_data(
    data,
    px_to_length=100,
    units="meters",
    plant_part="Root",
    time_per_frame=15,      # minutes
    time_unit="minutes"
)
```

### With Scale Bar

For publication-quality figures without axes:

```python
fig, ax = plot_centerline_data(
    data,
    px_to_length=100,
    units="meters",
    show_scale_bar=True,
    scale_bar_location="lower right",
    scale_bar_length=0.01  # 1 cm
)
```

### Customization Options

```python
fig, ax = plot_centerline_data(
    data,
    px_to_length=100,
    units="millimeters",           # or "meters", "pixels"
    scatter_only=False,             # True for scatter points only
    reverse_frame_order=True,       # Latest frames on top
    point_size=2,                   # Marker size
    plant_part="Stem",              # For title
    time_per_frame=10,
    time_unit="minutes",
    show_scale_bar=False,
    scale_bar_location="upper left"
)

# Save figure
fig.savefig("centerline_timeseries.png", dpi=300, bbox_inches='tight')
```

## Angle Plots

### Simple Angle Display

```python
from tropism_toolset import display_angles
import numpy as np

angles = get_angles(frame_data)

display_angles(
    angles,
    title="Angles Along Organ",
    x_axis_title="Point Index"
)
```

### Angle vs Arc Length

```python
import matplotlib.pyplot as plt
from tropism_toolset import get_angles, get_arclengths

angles = get_angles(frame_data)
arclengths = get_arclengths(frame_data) / 100  # Convert pixels to meters

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(arclengths[1:], angles, 'o-', markersize=4)
ax.set_xlabel('Arc Length s (m)', fontsize=12)
ax.set_ylabel('Angle θ (rad)', fontsize=12)
ax.set_yticks(np.arange(-np.pi, np.pi + np.pi/2, np.pi/2))
ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax.set_title('Angle Profile', fontsize=14)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

### Multi-Frame Angle Evolution

```python
from tropism_toolset import plot_theta_vs_arclength_over_time

angles_per_frame = get_angles_over_time(data)
arclengths_per_frame = get_arclengths_over_time(data)
# Convert to meters
arclengths_per_frame = [arc / 100 for arc in arclengths_per_frame]

plot_theta_vs_arclength_over_time(
    angles_per_frame,
    arclengths_per_frame,
    px_to_length=100,
    title="Gravitropic Response Over Time"
)
```

## Convergence Length Fits

### Lc Fit Visualization

The `fit_Lc` function automatically creates a publication-ready plot:

```python
from tropism_toolset import fit_Lc

x0, Bl, A, Lc, r_squared = fit_Lc(
    arclengths,
    angles,
    show=True,  # Creates the plot
    crop_end=3
)

# The plot includes:
# - Data points
# - Fitted curve
# - Parameters in legend
# - Proper mathematical notation
# - π-based y-axis
```

### Custom Lc Plot

```python
from tropism_toolset.display_utils import plot_piecewise_saturating_exponential
from tropism_toolset.fitting import piecewise_constant_saturating_exponential

# After fitting
fig, ax = plot_piecewise_saturating_exponential(
    x_data=arclengths[1:],
    y_data=angles,
    cum_len=arclengths,
    x0=x0,
    Bl=Bl,
    A=A,
    Lc=Lc,
    r_squared=r_squared,
    piecewise_func=piecewise_constant_saturating_exponential
)

# Customize further
ax.set_title('My Custom Title', fontsize=16)
fig.savefig('Lc_fit_custom.png', dpi=300)
```

## Growth Dynamics

### Length Over Time

```python
from tropism_toolset import display_length_over_time

frames = data['frame'].unique()
lengths = [...]  # Your length data

display_length_over_time(
    times=frames,
    lengths=np.array(lengths),
    title="Root Elongation",
    units="meters"
)
```

### Linear Fits

```python
from tropism_toolset.display_utils import display_linear_fit

# After fitting
coeffs, r_squared = fit_linear(frames, lengths)

display_linear_fit(
    times=frames,
    values=lengths,
    coeffs=coeffs,
    r_squared=r_squared,
    ylabel="Length (m)",
    title="Growth Rate Analysis",
    slope_label="growth rate",
    slope_units="m/frame"
)
```

### Piecewise Linear Fits

```python
from tropism_toolset.display_utils import display_piecewise_fit

display_piecewise_fit(
    times=frames,
    values=angles,
    p1_coeffs=phase1_coeffs,
    p2_coeffs=phase2_coeffs,
    breakpoint=transition_frame,
    ylabel="Angle (rad)",
    title="Two-Phase Angular Dynamics",
    use_radian_ticks=True,
    slope_label="angular velocity",
    slope_units="rad/frame"
)
```

### Saturating Exponential

```python
from tropism_toolset.display_utils import plot_saturating_exponential
from tropism_toolset.fitting import saturating_exponential_func

# After fitting
y_inf, y_0, tau, r_squared = fit_saturating_exponential(frames, lengths)

fig, ax = plot_saturating_exponential(
    x_data=frames,
    y_data=lengths,
    y_inf=y_inf,
    y_0=y_0,
    tau=tau,
    r_squared=r_squared,
    data_type="length",
    saturating_func=saturating_exponential_func
)
```

### Logistic Growth

```python
from tropism_toolset.display_utils import plot_logistic_growth
from tropism_toolset.fitting import logistic_growth_func

K, y_0, r, t_m, r_squared = fit_logistic_growth(frames, lengths)

fig, ax = plot_logistic_growth(
    x_data=frames,
    y_data=lengths,
    K=K,
    y_0=y_0,
    r=r,
    t_m=t_m,
    r_squared=r_squared,
    data_type="length",
    logistic_func=logistic_growth_func
)
```

## Steady State Analysis

### Length-Based Steady State

```python
from tropism_toolset.display_utils import display_steady_state_analysis

display_steady_state_analysis(
    data=lengths,
    data_smoothed=lengths_smooth,
    steady_start_orig=None,
    steady_start_smooth=Tc,
    title="Length Steady State Analysis",
    ylabel="Length (m)",
    time_unit="Frame",
    conversion_factor=15*60  # frames to seconds
)
```

### Mask Stability

```python
from tropism_toolset.display_utils import display_mask_stability_analysis

display_mask_stability_analysis(
    stability_scores=scores,
    steady_start=Tc,
    title="Shape Stability Analysis",
    ylabel="Overlap Score",
    time_unit="Frame",
    method="overlap"
)
```

## Video Creation

### Basic Video from Masks

```python
from tropism_toolset import create_video_with_colored_frames
from pathlib import Path

create_video_with_colored_frames(
    mask_dir=Path("data/experiment_masks"),
    output_file="analysis.mp4",
    Tc=120,              # Frame where steady state begins
    framerate=6          # Frames per second in video
)
```

The video will show:
- White text before Tc (growth phase)
- Red text from Tc onwards (steady state)
- Frame numbers
- Status labels

## Heatmaps

### Angle Evolution Heatmap

```python
from tropism_toolset.display_utils import plot_timelapse_heatmap
import pandas as pd

# Create s-theta-frame dataframe
heatmap_data = []
for frame in data['frame'].unique():
    frame_data = data[data['frame'] == frame]
    s, theta = x_y_to_s_theta(
        frame_data['x'].values,
        frame_data['y'].values,
        px_to_m
    )
    for s_val, theta_val in zip(s, theta):
        heatmap_data.append({
            'frame': frame,
            's': s_val,
            'theta': theta_val
        })

heatmap_df = pd.DataFrame(heatmap_data)

# Plot heatmap
fig, ax = plot_timelapse_heatmap(
    heatmap_df,
    title="Spatiotemporal Angle Evolution",
    cmap="viridis",
    vmin=0.0,
    vmax=np.pi/2
)
```

## Publication-Ready Figures

### Multi-Panel Figures

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 10))

# Panel A: Centerline evolution
ax1 = plt.subplot(2, 3, 1)
plot_centerline_data(data, px_to_length=100, units="meters")
ax1.text(-0.1, 1.1, 'A', transform=ax1.transAxes,
         fontsize=16, fontweight='bold')

# Panel B: Angle profile
ax2 = plt.subplot(2, 3, 2)
ax2.plot(arclengths[1:], angles, 'o-')
ax2.set_xlabel('Arc length (m)')
ax2.set_ylabel('Angle (rad)')
ax2.text(-0.1, 1.1, 'B', transform=ax2.transAxes,
         fontsize=16, fontweight='bold')

# Panel C: Lc fit
ax3 = plt.subplot(2, 3, 3)
# ... Lc fit plot
ax3.text(-0.1, 1.1, 'C', transform=ax3.transAxes,
         fontsize=16, fontweight='bold')

# Panel D: Length over time
ax4 = plt.subplot(2, 3, 4)
ax4.plot(frames, lengths, 'o-')
ax4.set_xlabel('Frame')
ax4.set_ylabel('Length (m)')
ax4.text(-0.1, 1.1, 'D', transform=ax4.transAxes,
         fontsize=16, fontweight='bold')

# Panel E: Tip angle
ax5 = plt.subplot(2, 3, 5)
ax5.plot(frames, tip_angles, 'o-')
ax5.set_xlabel('Frame')
ax5.set_ylabel('Tip angle (rad)')
ax5.text(-0.1, 1.1, 'E', transform=ax5.transAxes,
         fontsize=16, fontweight='bold')

# Panel F: Summary statistics
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
Constants:
Lc = {Lc:.4f} m
γ = {gamma:.6f} s⁻¹
β = {beta:.4f} m⁻¹

Dynamics:
Growth rate = {growth_rate:.6f} m/frame
Tc = {Tc} frames
"""
ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
         fontsize=12, verticalalignment='top', fontfamily='monospace')
ax6.text(-0.1, 1.1, 'F', transform=ax6.transAxes,
         fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('figure_multipanel.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Styling for Publication

```python
import matplotlib.pyplot as plt

# Set global style
plt.rcParams.update({
    'font.family': 'Helvetica',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.5,
    'grid.alpha': 0.3,
})

# Now all plots will use this style
```

## Batch Visualization

### Compare Multiple Experiments

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6']

for i, (ax, exp_name) in enumerate(zip(axes.flat, experiments)):
    # Load each experiment
    data = pd.read_csv(f"data/{exp_name}_centerlines.csv")

    # Plot on subplot
    plot_centerline_data(data, px_to_length=100, show_scale_bar=True)
    ax.set_title(exp_name)

plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300)
```

### Distribution Plots

```python
from tropism_toolset.display_utils import plot_constant_histogram

# Dictionary of {experiment: [values]}
Lc_results = {
    'exp1': [0.042, 0.038, 0.045],
    'exp2': [0.051, 0.048],
    'exp3': [0.039, 0.041, 0.043, 0.037]
}

plot_constant_histogram(
    Lc_results,
    constant_name="Convergence Length Lc (m)",
    show_stats=True,
    remove_nonpositive=True
)
```

## Saving Figures

```python
# High-resolution PNG
fig.savefig('figure.png', dpi=300, bbox_inches='tight')

# Vector format (PDF)
fig.savefig('figure.pdf', bbox_inches='tight')

# SVG for editing
fig.savefig('figure.svg', bbox_inches='tight')

# With transparency
fig.savefig('figure.png', dpi=300, bbox_inches='tight', transparent=True)
```

## Interactive Plots (Jupyter)

```python
%matplotlib widget
import matplotlib.pyplot as plt

# Now plots will be interactive
fig, ax = plt.subplots()
ax.plot(arclengths[1:], angles, 'o-')
plt.show()

# Zoom, pan, save from the interface
```

## Next Steps

- Create comprehensive [examples](/examples/workflows/) using these visualizations
- Apply to [batch processing](/advanced/batch-processing/) for multiple experiments
- Combine with [mathematical models](/guides/models/chauvet/) for theoretical comparisons
