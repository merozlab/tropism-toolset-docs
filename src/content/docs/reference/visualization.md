---
title: Visualization Module API
description: API reference for plotting and display functions.
---

Functions for visualizing plant data, model fits, and analysis results using matplotlib.

## Basic Plotting

### `display_angles()`

Display angles as scatter plot with π-notation.

**Signature:**
```python
display_angles(
    angles: AngleArray | list,
    title: str,
    x_axis_title: str
) -> None
```

**Parameters:**
- `angles` (array-like): Angles in radians to plot
- `title` (str): Plot title
- `x_axis_title` (str): X-axis label

**Example:**
```python
from constants import get_angles
from constants.display_utils import display_angles

angles = get_angles(frame_data)
display_angles(angles, "Angle Distribution", "Point Index")
```

**Notes:**
- Y-axis uses π-based ticks (-π, -π/2, 0, π/2, π)
- Simple scatter plot for quick angle visualization

---

## Centerline Visualization

### `plot_centerline_data()`

Plot centerline trajectories over time with color-coded frames.

**Signature:**
```python
plot_centerline_data(
    data: pd.DataFrame,
    units: str = "meters",
    px_to_length: Optional[float] = None,
    scatter_only: bool = False,
    reverse_frame_order: bool = True,
    point_size: int = 1,
    plant_part: Optional[str] = None,
    time_per_frame: Optional[float] = None,
    time_unit: str = "minutes",
    show_scale_bar: bool = True,
    scale_bar_location: str = "lower right",
    scale_bar_length: Optional[float] = None
) -> tuple[Figure, Axes]
```

**Parameters:**
- `data` (DataFrame): Centerline data with 'x', 'y', 'frame' columns
- `units` (str, optional): Length units ('meters', 'mm', 'pixels'). Default: 'meters'
- `px_to_length` (float, optional): Pixel to length conversion. Default: None
- `scatter_only` (bool, optional): Plot only points, no lines. Default: False
- `reverse_frame_order` (bool, optional): Latest frames on top. Default: True
- `point_size` (int, optional): Scatter point size. Default: 1
- `plant_part` (str, optional): Plant part name for title. Default: None
- `time_per_frame` (float, optional): Time per frame for colorbar. Default: None
- `time_unit` (str, optional): Time unit ('minutes', 'hours', etc.). Default: 'minutes'
- `show_scale_bar` (bool, optional): Show map-style scale bar. Default: True
- `scale_bar_location` (str, optional): Scale bar position. Default: 'lower right'
- `scale_bar_length` (float, optional): Explicit scale bar length. Default: None (auto)

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from constants.display_utils import plot_centerline_data

# Basic usage
fig, ax = plot_centerline_data(
    data,
    units='meters',
    plant_part='Root'
)

# With time-based colorbar
fig, ax = plot_centerline_data(
    data,
    units='meters',
    time_per_frame=15,  # 15 minutes per frame
    time_unit='minutes',
    show_scale_bar=True
)

# Pixel data with conversion
fig, ax = plot_centerline_data(
    data,
    units='pixels',
    px_to_length=10000,  # pixels per meter
    scatter_only=True
)
```

**Notes:**
- Color progression: yellow (early) to purple (late)
- Automatically infers column names ('x', 'y' or 'x (meters)', 'y (meters)')
- Scale bar auto-calculates "nice" rounded lengths
- Works with both Cartesian (x,y) and arc-length (s,θ) coordinates

---

### `plot_theta_vs_arclength_over_time()`

Plot angle θ vs arc length s for multiple frames.

**Signature:**
```python
plot_theta_vs_arclength_over_time(
    angles_per_frame: dict,
    arclengths_per_frame: dict,
    px_to_length: float = 1.0,
    title: str = "Angle θ vs Arclength over Time"
) -> None
```

**Parameters:**
- `angles_per_frame` (dict): Frame ID → angle array mapping
- `arclengths_per_frame` (dict): Frame ID → arclength array mapping
- `px_to_length` (float, optional): Pixel to meter conversion. Default: 1.0
- `title` (str, optional): Plot title

**Example:**
```python
from constants import get_angles_over_time, get_arclengths_over_time
from constants.display_utils import plot_theta_vs_arclength_over_time

angles = get_angles_over_time(data)
arclengths = get_arclengths_over_time(data)

# Convert to dict format
angles_dict = {i: angles[i] for i in range(len(angles))}
arcs_dict = {i: arclengths[i] for i in range(len(arclengths))}

plot_theta_vs_arclength_over_time(
    angles_dict,
    arcs_dict,
    px_to_length=100
)
```

**Notes:**
- Y-axis uses π-based ticks
- Color-coded by frame using viridis colormap
- Handles length mismatches between angles and arclengths
- Limits legend to first 10 frames for readability

---

### `plot_timelapse_heatmap()`

Create spatiotemporal heatmap of angle vs arclength over time.

**Signature:**
```python
plot_timelapse_heatmap(
    data: pd.DataFrame,
    title: str = "Angle θ vs Arclength Heatmap Over Time",
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = np.pi / 2
) -> tuple[Figure, Axes]
```

**Parameters:**
- `data` (DataFrame): Must have 'frame', 's', 'theta' columns
- `title` (str, optional): Plot title
- `cmap` (str, optional): Colormap name. Default: 'viridis'
- `vmin` (float, optional): Min color value. Default: 0.0
- `vmax` (float, optional): Max color value. Default: π/2

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from constants import x_y_to_s_theta
from constants.display_utils import plot_timelapse_heatmap

# Convert to s,theta coordinates
s_theta_data = x_y_to_s_theta(data, px_to_length=100)

fig, ax = plot_timelapse_heatmap(
    s_theta_data,
    title="Root Angle Evolution",
    cmap="RdBu_r",
    vmin=-np.pi/4,
    vmax=np.pi/4
)
```

**Notes:**
- X-axis: Arclength (s)
- Y-axis: Time (frames or seconds)
- Useful for visualizing gravitropic response patterns
- Handles variable-length centerlines per frame

---

## Model Fit Visualization

### `plot_piecewise_saturating_exponential()`

Display Lc convergence length fit results.

**Signature:**
```python
plot_piecewise_saturating_exponential(
    x_data: ArclengthArray,
    y_data: AngleArray,
    cum_len: ArclengthArray,
    x0: float,
    Bl: float,
    A: float,
    Lc: float,
    r_squared: float,
    piecewise_func: callable
) -> tuple[Figure, Axes]
```

**Parameters:**
- `x_data` (array): Arclength data used for fitting
- `y_data` (array): Angle data used for fitting
- `cum_len` (array): Full cumulative length for plot range
- `x0` (float): Transition point parameter
- `Bl` (float): Baseline angle parameter
- `A` (float): Amplitude parameter
- `Lc` (float): Convergence length
- `r_squared` (float): R² value
- `piecewise_func` (callable): Piecewise function for smooth curve

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from constants.fitting import fit_Lc

x0, Bl, A, Lc, r2 = fit_Lc(
    arclengths,
    angles,
    show=True  # Automatically calls this function
)
```

**Notes:**
- Called automatically when `fit_Lc(show=True)`
- Y-axis uses π-based ticks
- Displays fitted parameters in legend
- Shows gravitropism model equation

---

### `plot_saturating_exponential()`

Plot saturating exponential fit for time series.

**Signature:**
```python
plot_saturating_exponential(
    x_data: np.ndarray,
    y_data: np.ndarray,
    y_inf: float,
    y_0: float,
    tau: float,
    r_squared: float,
    data_type: str = "angle",
    saturating_func: Optional[callable] = None
) -> tuple[Figure, Axes]
```

**Parameters:**
- `x_data` (array): Time/frame values
- `y_data` (array): Angle or length measurements
- `y_inf` (float): Steady-state value
- `y_0` (float): Initial value
- `tau` (float): Time constant [frames]
- `r_squared` (float): R² value
- `data_type` (str, optional): 'angle' or 'length'. Default: 'angle'
- `saturating_func` (callable, optional): Function for smooth curve

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from constants.fitting import fit_saturating_exponential

theta_inf, theta_0, tau, r2 = fit_saturating_exponential(
    frames,
    angles,
    show=True,  # Automatically calls this function
    data_type='angle'
)
```

**Notes:**
- Called automatically when `fit_saturating_exponential(show=True)`
- Adapts labels/symbols based on data_type
- Shows horizontal line for steady-state value
- Displays time constant in title

---

### `plot_logistic_growth()`

Plot logistic (S-curve) growth fit.

**Signature:**
```python
plot_logistic_growth(
    x_data: np.ndarray,
    y_data: np.ndarray,
    K: float,
    y_0: float,
    r: float,
    t_m: float,
    r_squared: float,
    data_type: str = "angle",
    logistic_func: Optional[callable] = None
) -> tuple[Figure, Axes]
```

**Parameters:**
- `x_data` (array): Time/frame values
- `y_data` (array): Angle or length measurements
- `K` (float): Carrying capacity
- `y_0` (float): Initial value
- `r` (float): Growth rate parameter
- `t_m` (float): Inflection point time
- `r_squared` (float): R² value
- `data_type` (str, optional): 'angle' or 'length'. Default: 'angle'
- `logistic_func` (callable, optional): Function for smooth curve

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from constants.fitting import fit_logistic_growth

K, y_0, r, t_m, r2 = fit_logistic_growth(
    frames,
    lengths,
    show=True,  # Automatically calls this function
    data_type='length'
)
```

**Notes:**
- Called automatically when `fit_logistic_growth(show=True)`
- Shows horizontal line for carrying capacity
- Vertical line marks inflection point
- Displays growth rate and R² in title

---

## Fit Display Functions

### `display_linear_fit()`

Display data with single linear fit.

**Signature:**
```python
display_linear_fit(
    times: TimeArray,
    values: TimeSeriesValues,
    coeffs: PolyCoeffs,
    r_squared: RSquared,
    ylabel: str,
    title: str,
    use_radian_ticks: bool = False,
    slope_label: str = "slope",
    slope_units: str = "",
    xlabel: str = "Frame",
    xvalues: Optional[TimeSeriesValues] = None,
    use_xvalues_for_fit: bool = False
) -> None
```

**Parameters:**
- `times` (array): Frame numbers (used for fitting)
- `values` (array): Y-axis values
- `coeffs` (array): [slope, intercept] coefficients
- `r_squared` (float): R² value
- `ylabel` (str): Y-axis label
- `title` (str): Plot title
- `use_radian_ticks` (bool, optional): Use π-based ticks. Default: False
- `slope_label` (str, optional): Label for slope. Default: 'slope'
- `slope_units` (str, optional): Units for slope. Default: ''
- `xlabel` (str, optional): X-axis label. Default: 'Frame'
- `xvalues` (array, optional): Alternative x-values for plotting. Default: None
- `use_xvalues_for_fit` (bool, optional): Use xvalues for fit line. Default: False

**Example:**
```python
from constants.fitting import fit_linear
from constants.display_utils import display_linear_fit

slope, r2, coeffs = fit_linear(frames, lengths, show=False)

display_linear_fit(
    frames,
    lengths,
    coeffs,
    r2,
    ylabel="Length (m)",
    title="Growth Rate",
    slope_label="growth rate",
    slope_units="m/frame"
)
```

**Notes:**
- Shows slope and R² in text box
- Gray scatter points with red fit line
- Supports custom x-values for non-time-series plots

---

### `display_piecewise_fit()`

Display data with two-phase piecewise linear fit.

**Signature:**
```python
display_piecewise_fit(
    times: TimeArray,
    values: TimeSeriesValues,
    p1_coeffs: PolyCoeffs,
    p2_coeffs: PolyCoeffs,
    breakpoint: int,
    ylabel: str,
    title: str,
    use_radian_ticks: bool = False,
    slope_label: str = "slope",
    slope_units: str = ""
) -> None
```

**Parameters:**
- `times` (array): Frame numbers
- `values` (array): Y-axis values
- `p1_coeffs` (array): Phase 1 [slope, intercept]
- `p2_coeffs` (array): Phase 2 [slope, intercept]
- `breakpoint` (int): Frame index for transition
- `ylabel` (str): Y-axis label
- `title` (str): Plot title
- `use_radian_ticks` (bool, optional): Use π-based ticks. Default: False
- `slope_label` (str, optional): Label for slopes. Default: 'slope'
- `slope_units` (str, optional): Units for slopes. Default: ''

**Example:**
```python
from constants.fitting import fit_piecewise_linear_continuous
from constants.display_utils import display_piecewise_fit

m1, m2, x0, (p1, p2) = fit_piecewise_linear_continuous(
    frames,
    angles,
    show=False
)

bp_idx = int(np.argmin(np.abs(frames - x0)))
display_piecewise_fit(
    frames,
    angles,
    p1,
    p2,
    bp_idx,
    ylabel="Angle (rad)",
    title="Two-Phase Angular Velocity",
    use_radian_ticks=True,
    slope_label="angular velocity",
    slope_units="rad/frame"
)
```

**Notes:**
- Phase 1: red line, Phase 2: blue line
- Vertical dashed line marks transition
- Shows both slopes in text box

---

## Analysis Visualization

### `display_steady_state_analysis()`

Display steady state analysis with statistics.

**Signature:**
```python
display_steady_state_analysis(
    data: AngleArray | ArclengthArray,
    data_smoothed: Optional[AngleArray | ArclengthArray] = None,
    steady_start_orig: Optional[int] = None,
    steady_start_smooth: Optional[int] = None,
    title: str = "Steady State Analysis",
    ylabel: str = "Value",
    time_unit: str = "Frame",
    conversion_factor: float = 1.0
) -> None
```

**Parameters:**
- `data` (array): Original time series data
- `data_smoothed` (array, optional): Smoothed version. Default: None
- `steady_start_orig` (int, optional): Steady state start in original. Default: None
- `steady_start_smooth` (int, optional): Steady state start in smoothed. Default: None
- `title` (str, optional): Plot title
- `ylabel` (str, optional): Y-axis label
- `time_unit` (str, optional): Time unit name. Default: 'Frame'
- `conversion_factor` (float, optional): Frame to time conversion. Default: 1.0

**Example:**
```python
from constants import get_lengths_from_centerlines
from constants.fitting import find_steady_state, smooth_centerlines
from constants.display_utils import display_steady_state_analysis

times, lengths, _ = get_lengths_from_centerlines(data)

# Find steady state
Tc_orig = find_steady_state(lengths)

# Smooth and find again
smoothed_data = smooth_centerlines(data)
times_smooth, lengths_smooth, _ = get_lengths_from_centerlines(smoothed_data)
Tc_smooth = find_steady_state(lengths_smooth)

display_steady_state_analysis(
    lengths,
    lengths_smooth,
    Tc_orig,
    Tc_smooth,
    title="Growth Stabilization",
    ylabel="Length (m)",
    time_unit="Frame"
)
```

**Notes:**
- Two-panel plot: full time series + zoomed steady state region
- Prints detailed statistics (mean, std, range)
- Compares original vs smoothed detection
- Blue marker for original, red for smoothed

---

### `display_mask_stability_analysis()`

Display mask stability analysis results.

**Signature:**
```python
display_mask_stability_analysis(
    stability_scores: ArclengthArray,
    steady_start: Optional[int] = None,
    title: str = "Mask Stability Analysis",
    ylabel: str = "Instability Score",
    time_unit: str = "Frame",
    method: str = "overlap"
) -> None
```

**Parameters:**
- `stability_scores` (array): Instability scores between frames
- `steady_start` (int, optional): Steady state start frame. Default: None
- `title` (str, optional): Plot title
- `ylabel` (str, optional): Y-axis label
- `time_unit` (str, optional): Time unit name
- `method` (str, optional): Analysis method used ('overlap' or 'edge_change')

**Example:**
```python
from constants.fitting import find_steady_state_from_masks

scores, Tc, area = find_steady_state_from_masks(
    mask_dir='data/masks/',
    px_per_m=10000,
    method='edge_change',
    show=True  # Automatically calls this function
)
```

**Notes:**
- Called automatically when `find_steady_state_from_masks(show=True)`
- Vertical line marks steady state onset
- Prints statistics for full and steady state periods
- Higher scores = more instability

---

### `display_length_over_time()`

Display total length time series.

**Signature:**
```python
display_length_over_time(
    times: TimeArray,
    lengths: TimeSeriesValues,
    title: str,
    units: Optional[str] = None
) -> None
```

**Parameters:**
- `times` (array): Frame numbers
- `lengths` (array): Total lengths
- `title` (str): Plot title
- `units` (str, optional): Length units ('meters', 'pixels', etc.). Default: None

**Example:**
```python
from constants import get_lengths_from_centerlines
from constants.display_utils import display_length_over_time

times, lengths, units = get_lengths_from_centerlines(data)

display_length_over_time(
    times,
    lengths,
    "Root Elongation Over Time",
    units=units
)
```

**Notes:**
- Simple scatter plot for quick visualization
- Auto-detects unit labels from units parameter

---

## Statistical Visualization

### `plot_constant_histogram()`

Plot logarithmically binned histogram of constant values.

**Signature:**
```python
plot_constant_histogram(
    results: dict,
    constant_name: str,
    show_stats: bool = True,
    remove_nonpositive: bool = True
) -> None
```

**Parameters:**
- `results` (dict): Experiment name → list of constant values
- `constant_name` (str): Name of constant for labels
- `show_stats` (bool, optional): Print statistics. Default: True
- `remove_nonpositive` (bool, optional): Filter out ≤0 values. Default: True

**Example:**
```python
from constants.display_utils import plot_constant_histogram

# Collect Lc values from multiple experiments
results = {
    'experiment1': [0.042, 0.038, 0.045],
    'experiment2': [0.041, 0.039, 0.044],
    'experiment3': [0.040, 0.043, 0.037]
}

plot_constant_histogram(
    results,
    constant_name="L_c (m)",
    show_stats=True
)
```

**Notes:**
- Log-scale x-axis
- One bin per decade (powers of 10)
- Red dashed line shows mean
- Prints: total values, range, mean, median
- Useful for batch analysis results

---

## Video Generation

### `create_video_with_colored_frames()`

Create annotated video from mask files.

**Signature:**
```python
create_video_with_colored_frames(
    mask_dir: str | Path,
    output_file: str,
    Tc: Optional[int],
    framerate: int = 6
) -> bool
```

**Parameters:**
- `mask_dir` (str or Path): Directory with mask files (*.bmp)
- `output_file` (str): Output video path
- `Tc` (int or None): Steady state start frame
- `framerate` (int, optional): Video framerate. Default: 6

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
from constants.fitting import find_steady_state_from_masks
from constants.display_utils import create_video_with_colored_frames

# Find steady state
scores, Tc, area = find_steady_state_from_masks(
    mask_dir='data/masks/',
    px_per_m=10000
)

# Create video
success = create_video_with_colored_frames(
    mask_dir='data/masks/',
    output_file='growth_timelapse.mp4',
    Tc=Tc,
    framerate=10
)
```

**Notes:**
- White frame numbers before Tc (growth phase)
- Red frame numbers from Tc onwards (steady state)
- Adds "GROWTH PHASE" or "STEADY STATE" labels
- Horizontally flips images for correct orientation
- Uses H.264 encoding for MP4 compatibility
- Does not modify original mask files

---

## Helper Functions

### `_calculate_nice_scale_bar_length()`

Calculate rounded scale bar length.

**Signature:**
```python
_calculate_nice_scale_bar_length(
    data_range: float,
    target_fraction: float = 0.2
) -> float
```

**Parameters:**
- `data_range` (float): Total data range
- `target_fraction` (float, optional): Target fraction of range. Default: 0.2 (20%)

**Returns:**
- `float`: Nice rounded scale bar length

**Notes:**
- Internal helper for `plot_centerline_data()`
- Rounds to powers of 10 × {1, 2, or 5}
- E.g., for range 0.047 m → suggests 0.01 m scale bar

---

## See Also

- [Fitting API](/reference/fitting/) - Functions that call these visualization tools
- [Geometric Calculations API](/reference/geometric-calculations/) - Data preparation functions
- [Visualization Guide](/guides/visualization/) - Usage examples and workflows
