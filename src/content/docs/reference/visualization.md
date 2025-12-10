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
    angles: pd.Series,
    display: bool = False,
    save_path: str | Path | None = None,
) -> None
```

**Parameters:**
- `angles` (pd.Series): Angles in radians to plot. Name should include unit, e.g., "theta (rad)".
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Example:**
```python
from tropism_toolset import get_angles
from tropism_toolset.display_utils import display_angles

angles = get_angles(frame_data)
display_angles(angles, display=True)
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
    scatter_only: bool = False,
    reverse_frame_order: bool = True,
    point_size: int = 1,
    plant_part: Optional[str] = None,
    show_scale_bar: bool = True,
    scale_bar_location: str = "lower right",
    scale_bar_length: Optional[float] = None,
    display: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

**Parameters:**
- `data` (pd.DataFrame): Centerline data with 'x', 'y', 'frame' columns (or s, theta)
- `scatter_only` (bool, optional): Plot only points, no lines. Default: False
- `reverse_frame_order` (bool, optional): Latest frames on top. Default: True
- `point_size` (int, optional): Scatter point size. Default: 1
- `plant_part` (str, optional): Plant part name for title. Default: None
- `show_scale_bar` (bool, optional): Show map-style scale bar. Default: True
- `scale_bar_location` (str, optional): Scale bar position. Default: 'lower right'
- `scale_bar_length` (float, optional): Explicit scale bar length. Default: None (auto)
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

**Example:**
```python
from tropism_toolset.display_utils import plot_centerline_data

# Basic usage
fig, ax = plot_centerline_data(
    data,
    plant_part='Root',
    display=True
)
```

**Notes:**
- Color progression: yellow (early) to purple (late)
- Automatically infers column names and units
- Scale bar auto-calculates "nice" rounded lengths
- Works with both Cartesian (x,y) and arc-length (s,θ) coordinates

---

### `display_centerline_endpoints()`

Display a specific frame of centerline data with colored endpoint markers.

**Signature:**
```python
display_centerline_endpoints(
    data: pd.DataFrame,
    frame: int = -1,
    units: str = "meters",
    px_to_length: float | None = None,
    display: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

**Parameters:**
- `data` (pd.DataFrame): DataFrame containing centerline data with 'x', 'y', and 'frame' columns
- `frame` (int, optional): Frame number to display. If -1, displays the last frame. Default: -1
- `units` (str, optional): Units for length ('meters', 'millimeters'/'mm', 'pixels'/'px'). Default: "meters"
- `px_to_length` (float, optional): Pixel to length conversion factor. If None, plot in data units. Default: None
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

---

### `plot_theta_vs_arclength_over_time()`

Plot angle θ vs arc length s for multiple frames.

**Signature:**
```python
plot_theta_vs_arclength_over_time(
    angles_per_frame: dict,
    arclengths_per_frame: dict,
    px_to_length: float = 1.0,
    title: str = "Angle θ vs Arclength over Time",
    display: bool = False,
    save_path: str | Path | None = None,
) -> None
```

**Parameters:**
- `angles_per_frame` (dict): Frame ID → angle array mapping
- `arclengths_per_frame` (dict): Frame ID → arclength array mapping
- `px_to_length` (float, optional): Pixel to meter conversion. Default: 1.0
- `title` (str, optional): Plot title
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

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
    vmax: float = np.pi / 2,
    display: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

**Parameters:**
- `data` (pd.DataFrame): Must have 'frame', 's', 'theta' columns
- `title` (str, optional): Plot title
- `cmap` (str, optional): Colormap name. Default: 'viridis'
- `vmin` (float, optional): Min color value. Default: 0.0
- `vmax` (float, optional): Max color value. Default: π/2
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

---

## Model Fit Visualization

### `plot_piecewise_saturating_exponential()`

Display Lc convergence length fit results.

**Signature:**
```python
plot_piecewise_saturating_exponential(
    x_data: pd.Series,
    y_data: pd.Series,
    x0: float,
    Bl: float,
    A: float,
    Lc: float,
    r_squared: float,
    piecewise_func,
    display: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

**Parameters:**
- `x_data` (pd.Series): Arclength data used for fitting
- `y_data` (pd.Series): Angle data used for fitting
- `x0` (float): Transition point parameter
- `Bl` (float): Baseline angle parameter
- `A` (float): Amplitude parameter
- `Lc` (float): Convergence length
- `r_squared` (float): R² value
- `piecewise_func` (callable): Piecewise function for smooth curve
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

---

### `plot_saturating_exponential()`

Plot saturating exponential fit for time series.

**Signature:**
```python
plot_saturating_exponential(
    x_data: pd.Series,
    y_data: pd.Series,
    y_inf: float,
    y_0: float,
    tau: float,
    r_squared: float,
    saturating_func=None,
    display: bool = False,
    save_path: str | Path | None = None,
) -> tuple[Figure, Axes]
```

**Parameters:**
- `x_data` (pd.Series): Time/frame values
- `y_data` (pd.Series): Angle or length measurements
- `y_inf` (float): Steady-state value
- `y_0` (float): Initial value
- `tau` (float): Time constant [frames]
- `r_squared` (float): R² value
- `saturating_func` (callable, optional): Function for smooth curve
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

**Returns:**
- `tuple[Figure, Axes]`: Matplotlib figure and axes

---

## Fit Display Functions

### `display_linear_fit()`

Display data with single linear fit.

**Signature:**
```python
display_linear_fit(
    x: pd.Series,
    y: pd.Series,
    coeffs: np.ndarray,
    r_squared: float,
    title: str | None = None,
    slope_label: str | None = None,
    xvalues: pd.Series | None = None,
    use_xvalues_for_fit: bool = False,
    display: bool = False,
    save_path: str | Path | None = None,
) -> None
```

**Parameters:**
- `x` (pd.Series): Frame numbers (used for fitting)
- `y` (pd.Series): Y-axis values
- `coeffs` (np.ndarray): [slope, intercept] coefficients
- `r_squared` (float): R² value
- `title` (str, optional): Plot title
- `slope_label` (str, optional): Label for slope. Default: inferred
- `xvalues` (pd.Series, optional): Alternative x-values for plotting. Default: None
- `use_xvalues_for_fit` (bool, optional): Use xvalues for fit line. Default: False
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

---

### `display_piecewise_fit()`

Display data with two-phase piecewise linear fit.

**Signature:**
```python
display_piecewise_fit(
    times: pd.Series,
    values: pd.Series,
    p1_coeffs: np.ndarray,
    p2_coeffs: np.ndarray,
    breakpoint: int,
    title: str | None = None,
    slope_label: str | None = None,
    display: bool = False,
    save_path: str | Path | None = None,
) -> None
```

**Parameters:**
- `times` (pd.Series): Frame numbers
- `values` (pd.Series): Y-axis values
- `p1_coeffs` (np.ndarray): Phase 1 [slope, intercept]
- `p2_coeffs` (np.ndarray): Phase 2 [slope, intercept]
- `breakpoint` (int): Frame index for transition
- `title` (str, optional): Plot title
- `slope_label` (str, optional): Label for slopes. Default: inferred
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

---

## Analysis Visualization

### `display_mask_stability_analysis()`

Display mask stability analysis results.

**Signature:**
```python
display_mask_stability_analysis(
    stability_scores: np.ndarray,
    steady_start: int | None = None,
    method: str = "overlap",
    display: bool = False,
    save_path: str | Path | None = None,
) -> None
```

**Parameters:**
- `stability_scores` (np.ndarray): Instability scores between frames
- `steady_start` (int, optional): Steady state start frame. Default: None
- `method` (str, optional): Analysis method used ('overlap' or 'edge_change')
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

---

## Statistical Visualization

### `plot_constant_histogram()`

Plot logarithmically binned histogram of constant values.

**Signature:**
```python
plot_constant_histogram(
    results,
    constant_name,
    show_stats=True,
    remove_nonpositive=True,
    display: bool = False,
    save_path: str | Path | None = None,
)
```

**Parameters:**
- `results` (dict): Experiment name → list of constant values
- `constant_name` (str): Name of constant for labels
- `show_stats` (bool, optional): Print statistics. Default: True
- `remove_nonpositive` (bool, optional): Filter out ≤0 values. Default: True
- `display` (bool, optional): Display plot inline. Default: False
- `save_path` (str | Path | None, optional): Save plot to path.

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

---

## See Also

- [Fitting API](/reference/fitting/) - Functions that call these visualization tools
- [Geometric Calculations API](/reference/geometric-calculations/) - Data preparation functions
- [Visualization Guide](/guides/visualization/) - Usage examples and workflows