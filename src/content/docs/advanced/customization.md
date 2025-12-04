---
title: Customization
description: Extend and customize the toolkit for your specific needs.
---

Learn how to customize and extend the Tropism Toolset for your specific research needs.

## Customizing Angle Calculation

### Change Angle Convention

Edit `src/constants/geometric_calculations.py`:

```python
# At the top of the file
PRESET = "mathematical"  # or "mathieu"
```

**mathematical** (default): Standard arctan2(dy, dx)
**mathieu**: Transformed as π/2 - arctan2(dy, dx)

### Create Custom Angle Function

```python
def get_angles_custom(data, transformation=None):
    """Custom angle calculation with optional transformation."""
    from tropism_toolset.geometric_calculations import infer_columns_and_units
    import numpy as np

    x_col, y_col, _ = infer_columns_and_units(data)
    dx = np.diff(data[x_col])
    dy = np.diff(data[y_col])

    angles = np.arctan2(dy, dx)

    # Apply custom transformation
    if transformation == 'vertical_reference':
        angles = np.pi/2 - angles
    elif transformation == 'horizontal_reference':
        angles = angles
    elif callable(transformation):
        angles = transformation(angles)

    return angles
```

## Custom Fitting Functions

### Add New Growth Model

```python
import numpy as np
from scipy.optimize import curve_fit

def gompertz_growth(t, A, b, c):
    """Gompertz growth model."""
    return A * np.exp(-b * np.exp(-c * t))

def fit_gompertz(times, values, show=False):
    """Fit Gompertz model to growth data."""
    # Initial guess
    A_guess = values[-1]
    b_guess = 2.0
    c_guess = 0.01

    # Fit
    params, _ = curve_fit(
        gompertz_growth,
        times,
        values,
        p0=[A_guess, b_guess, c_guess]
    )

    A, b, c = params

    # Calculate R²
    y_pred = gompertz_growth(times, A, b, c)
    ss_res = np.sum((values - y_pred)**2)
    ss_tot = np.sum((values - np.mean(values))**2)
    r_squared = 1 - (ss_res / ss_tot)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(times, values, 'o', label='Data')
        plt.plot(times, y_pred, '-', label=f'Gompertz Fit (R²={r_squared:.3f})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return A, b, c, r_squared
```

## Custom Visualization

### Create Custom Plot Style

```python
import matplotlib.pyplot as plt

def set_publication_style():
    """Set custom publication-ready style."""
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 11,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2,
        'axes.prop_cycle': plt.cycler(color=[
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ])
    })

# Use it
set_publication_style()
# Now all plots will use this style
```

### Custom Centerline Plot

```python
def plot_centerline_custom(data, px_to_m, **kwargs):
    """Custom centerline visualization."""
    import matplotlib.pyplot as plt
    from tropism_toolset import get_angles_over_time

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Centerline evolution
    unique_frames = sorted(data['frame'].unique())
    cmap = plt.get_cmap('plasma')

    for i, frame in enumerate(unique_frames[::10]):  # Every 10th frame
        frame_data = data[data['frame'] == frame]
        x = frame_data['x'].values / px_to_m
        y = frame_data['y'].values / px_to_m
        color = cmap(i / (len(unique_frames[::10]) - 1))
        ax1.plot(x, y, color=color, alpha=0.6, linewidth=2)

    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Centerline Evolution')
    ax1.axis('equal')
    ax1.grid(alpha=0.2)

    # Right: Angle evolution
    angles_per_frame, frames = get_angles_over_time(data)
    for i, frame in enumerate(frames[::10]):
        angles = angles_per_frame[frame]
        color = cmap(i / (len(frames[::10]) - 1))
        ax2.plot(range(len(angles)), angles, color=color, alpha=0.6)

    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Angle (rad)')
    ax2.set_title('Angle Evolution')
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    return fig, (ax1, ax2)
```

## Adding New Constants

### Define Custom Constant

```python
def get_custom_sensitivity(Lc, growth_rate, R):
    """
    Calculate custom gravitropic parameter.

    Parameters
    ----------
    Lc : float
        Convergence length (m)
    growth_rate : float
        Growth rate (m/s)
    R : float
        Radius (m)

    Returns
    -------
    float
        Custom sensitivity parameter
    """
    return (R * growth_rate) / (Lc ** 2)

# Use it
custom_param = get_custom_sensitivity(Lc=0.042, growth_rate=2e-6, R=0.00145)
```

### Add to Batch Processing

```python
# In your batch script
from tropism_toolset import get_beta, get_gamma

# Standard constants
gamma = get_gamma(Tc, period)
beta = get_beta(Lc, gamma)

# Add custom constant
from my_custom_module import get_custom_sensitivity
custom = get_custom_sensitivity(Lc, growth_rate, R)

# Store all
results.append({
    'gamma': gamma,
    'beta': beta,
    'custom_param': custom
})
```

## Data Preprocessing

### Custom Data Loader

```python
def load_experiment_data(exp_dir, exp_name):
    """
    Custom data loader with validation and preprocessing.

    Parameters
    ----------
    exp_dir : Path
        Directory containing experiments
    exp_name : str
        Experiment name

    Returns
    -------
    dict
        Dictionary with 'centerlines', 'masks', 'metadata'
    """
    from pathlib import Path
    import pandas as pd
    import glob

    exp_path = Path(exp_dir) / exp_name

    # Load centerlines
    csv_file = exp_path / f"{exp_name}_centerlines.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"No centerlines: {csv_file}")

    centerlines = pd.read_csv(csv_file)

    # Validate
    required_cols = ['frame', 'x', 'y']
    if not all(col in centerlines.columns for col in required_cols):
        raise ValueError(f"Missing columns. Need: {required_cols}")

    # Load masks
    mask_dir = exp_path / f"{exp_name}_masks"
    masks = sorted(glob.glob(str(mask_dir / "*.bmp"))) if mask_dir.exists() else []

    # Load metadata if exists
    metadata_file = exp_path / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        import json
        with open(metadata_file) as f:
            metadata = json.load(f)

    return {
        'centerlines': centerlines,
        'masks': masks,
        'metadata': metadata,
        'path': exp_path
    }
```

### Custom Smoothing

```python
from scipy.ndimage import gaussian_filter1d

def smooth_centerlines_gaussian(data, sigma=2.0):
    """
    Gaussian smoothing alternative to Savitzky-Golay.

    Parameters
    ----------
    data : DataFrame
        Centerline data
    sigma : float
        Standard deviation for Gaussian kernel

    Returns
    -------
    DataFrame
        Smoothed data
    """
    import pandas as pd

    df_smooth = data.copy()

    for frame in data['frame'].unique():
        mask = df_smooth['frame'] == frame
        frame_data = df_smooth[mask].copy()

        x_smooth = gaussian_filter1d(frame_data['x'].values, sigma=sigma)
        y_smooth = gaussian_filter1d(frame_data['y'].values, sigma=sigma)

        df_smooth.loc[mask, 'x'] = x_smooth
        df_smooth.loc[mask, 'y'] = y_smooth

    return df_smooth
```

## Extending Analysis Pipeline

### Create Analysis Class

```python
class TropismAnalysis:
    """Custom analysis pipeline."""

    def __init__(self, data_file, px_to_m=100, period=900):
        """Initialize analysis."""
        import pandas as pd

        self.data = pd.read_csv(data_file)
        self.px_to_m = px_to_m
        self.period = period
        self.results = {}

    def compute_all_constants(self):
        """Compute all constants."""
        from tropism_toolset import *

        # Extract Lc
        final_data = self.data[self.data['frame'] == self.data['frame'].max()]
        angles = get_angles(final_data)
        arclengths = get_arclengths(final_data, self.px_to_m)
        _, _, _, Lc, r2 = fit_Lc(arclengths, angles, show=False)

        # Find Tc
        lengths = []
        for frame in sorted(self.data['frame'].unique()):
            fd = self.data[self.data['frame'] == frame]
            al = get_arclengths(fd, self.px_to_m)
            lengths.append(al[-1])

        Tc, _ = find_steady_state(np.array(lengths), show=False)

        # Calculate constants
        gamma = get_gamma(Tc, self.period)
        beta = get_beta(Lc, gamma)

        self.results = {
            'Lc': Lc,
            'gamma': gamma,
            'beta': beta,
            'Tc': Tc,
            'r_squared': r2
        }

        return self.results

    def export_results(self, filename):
        """Export results to CSV."""
        import pandas as pd
        df = pd.DataFrame([self.results])
        df.to_csv(filename, index=False)
        print(f"✓ Results saved to {filename}")

# Use it
analysis = TropismAnalysis("data/exp1.csv")
results = analysis.compute_all_constants()
analysis.export_results("exp1_results.csv")
```

## Configuration Files

### Create Config System

```python
# config.yaml
import yaml

config = {
    'calibration': {
        'px_to_m': 100,
        'plant_radius_m': 0.00145
    },
    'temporal': {
        'minutes_per_frame': 15
    },
    'fitting': {
        'crop_start': 0,
        'crop_end': 3,
        'initial_guess': [0.01, 0.0, 1.5, 0.05]
    },
    'steady_state': {
        'threshold_factor': 0.15,
        'min_steady_duration': 20
    }
}

# Save
with open('analysis_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Load
with open('analysis_config.yaml') as f:
    config = yaml.safe_load(f)

# Use
px_to_m = config['calibration']['px_to_m']
```

## Testing Custom Functions

```python
def test_custom_function():
    """Test your custom function."""
    import numpy as np

    # Create test data
    test_angles = np.linspace(0, np.pi/2, 100)
    test_lengths = np.linspace(0, 0.1, 100)

    # Run function
    result = your_custom_function(test_angles, test_lengths)

    # Validate
    assert result is not None, "Function returned None"
    assert 0 < result < 10, f"Result {result} outside expected range"

    print("✓ Test passed")

# Run test
test_custom_function()
```

## See Also

- [API Reference](/reference/geometric-calculations/) - Function signatures
- [Examples](/examples/workflows/) - Usage patterns
- [Batch Processing](/advanced/batch-processing/) - Automation
