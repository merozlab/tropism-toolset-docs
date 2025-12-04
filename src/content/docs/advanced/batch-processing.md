---
title: Batch Processing
description: Automate analysis of multiple experiments efficiently.
---

Process multiple plant experiments automatically with the batch growth analysis pipeline.

## Overview

The `batch_growth_analysis.py` script automates the analysis of multiple experiments, extracting growth rates, growth zones, and summary statistics.

## Directory Structure

Organize your experiments in this structure:

```
data/full_experiment/
├── caroline_1/
│   ├── caroline_1_centerlines.csv
│   └── caroline_1_masks/
│       ├── 00001_mask.bmp
│       ├── 00002_mask.bmp
│       └── ...
├── caroline_5/
│   ├── caroline_5_centerlines.csv
│   └── caroline_5_masks/
│       └── ...
└── charli_3/
    ├── charli_3_centerlines.csv
    └── charli_3_masks/
        └── ...
```

**Requirements:**
- Each experiment has its own folder
- CSV filename: `{experiment_name}_centerlines.csv`
- Masks folder: `{experiment_name}_masks/`
- Masks are `.bmp` files with sequential numbering (00001, 00002, ...)

## Running Batch Analysis

### Basic Usage

```bash
cd /path/to/constants
poetry run python batch_growth_analysis.py
```

### What It Does

For each experiment, the script automatically:

1. **Growth Rate Analysis**
   - Loads centerline coordinates
   - Extracts plant length over time
   - Applies Savitzky-Golay smoothing
   - Fits linear growth model
   - Calculates growth statistics

2. **Growth Zone Analysis**
   - Loads first and last masks
   - Creates intersection mask (stable base region)
   - Uses PCA-based dimension calculation
   - Calibrates using plant radius
   - Calculates growth zone length

3. **Generates Outputs**
   - Visualization plots (PNG)
   - Console summary table
   - CSV results file

## Output Examples

### Visualizations

For each experiment, creates:

```
growth_analysis_results/
├── caroline_1_growth_rate.png
├── caroline_1_mask_overlay.png
├── caroline_5_growth_rate.png
├── caroline_5_mask_overlay.png
└── ...
```

**growth_rate.png**: Length vs time with linear fit
**mask_overlay.png**: 4-panel mask comparison showing:
- First mask
- Last mask
- Intersection (stable region)
- Overlay visualization

### Console Summary

```
Experiment             Frames  Initial(mm)    Final(mm)   Growth(mm)    %Growth    Rate(mm/fr)   Stable(mm)
----------------------------------------------------------------------------------------------------
caroline_1                453       134.12       160.54        26.42      19.7%        0.05300        75.35
caroline_5                421       128.45       155.23        26.78      20.8%        0.06358        70.12
charli_3                  389       142.67       168.89        26.22      18.4%        0.06739        81.45
```

## Configuration

### Modify Parameters

Edit `batch_growth_analysis.py`:

```python
# Directory settings
EXPERIMENT_DIR = Path("data/full_experiment")
OUTPUT_DIR = Path("growth_analysis_results")

# Plant calibration
R = 0.00145  # Plant radius in meters (1.45 mm)

# Smoothing parameters
WINDOW_LENGTH = 11
POLYORDER = 2
```

### Custom Analysis Function

Add custom processing for each experiment:

```python
def analyze_experiment(exp_folder, output_dir):
    """Custom analysis function."""
    exp_name = exp_folder.name
    centerlines_file = exp_folder / f"{exp_name}_centerlines.csv"

    # Your custom analysis
    data = pd.read_csv(centerlines_file)

    # Extract additional parameters
    # ... your code ...

    return results
```

## Advanced Batch Scripts

### Extract All Constants

```python
from pathlib import Path
import pandas as pd
from tropism_toolset import *

def batch_extract_constants(exp_dir, px_to_m=100, period=900):
    """Extract Lc, gamma, beta for all experiments."""

    results = []

    for exp_folder in Path(exp_dir).iterdir():
        if not exp_folder.is_dir():
            continue

        exp_name = exp_folder.name
        csv_file = exp_folder / f"{exp_name}_centerlines.csv"

        if not csv_file.exists():
            continue

        print(f"Processing: {exp_name}")

        try:
            # Load data
            data = pd.read_csv(csv_file)

            # Find Tc
            lengths = []
            for frame in sorted(data['frame'].unique()):
                fd = data[data['frame'] == frame]
                arclengths = get_arclengths(fd, px_to_m)
                lengths.append(arclengths[-1])

            Tc, _ = find_steady_state(np.array(lengths), show=False)

            # Fit Lc
            final_data = data[data['frame'] == data['frame'].max()]
            angles = get_angles(final_data)
            arclengths = get_arclengths(final_data, px_to_m)
            x0, Bl, A, Lc, r2 = fit_Lc(arclengths, angles, show=False)

            # Calculate constants
            gamma = get_gamma(Tc, period)
            beta = get_beta(Lc, gamma)

            results.append({
                'experiment': exp_name,
                'Lc': Lc,
                'gamma': gamma,
                'beta': beta,
                'Tc': Tc,
                'r_squared': r2
            })

            print(f"  ✓ Complete")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("batch_constants.csv", index=False)

    return df

# Run batch analysis
results_df = batch_extract_constants("data/full_experiment")
print(results_df)
```

### Parallel Processing

For large datasets, use parallel processing:

```python
from multiprocessing import Pool
from functools import partial

def analyze_single(exp_folder, px_to_m, period):
    """Analyze single experiment (for parallel execution)."""
    # Same as above but for single experiment
    # ... code ...
    return result

def batch_parallel(exp_dir, px_to_m=100, period=900, n_workers=4):
    """Run batch analysis in parallel."""

    exp_folders = [f for f in Path(exp_dir).iterdir() if f.is_dir()]

    # Create analysis function with fixed parameters
    analyze_func = partial(analyze_single, px_to_m=px_to_m, period=period)

    # Run in parallel
    with Pool(n_workers) as pool:
        results = pool.map(analyze_func, exp_folders)

    # Filter out failures (None values)
    results = [r for r in results if r is not None]

    return pd.DataFrame(results)

# Use 4 CPU cores
results_df = batch_parallel("data/full_experiment", n_workers=4)
```

## Quality Control

### Automated QC Checks

```python
def quality_control(results_df):
    """Run QC checks on batch results."""

    print("QUALITY CONTROL REPORT")
    print("="*60)

    # Check R² values
    poor_fits = results_df[results_df['r_squared'] < 0.90]
    if len(poor_fits) > 0:
        print(f"⚠ {len(poor_fits)} experiments with poor fits (R² < 0.90):")
        for idx, row in poor_fits.iterrows():
            print(f"  - {row['experiment']}: R² = {row['r_squared']:.3f}")

    # Check for outliers
    lc_mean = results_df['Lc'].mean()
    lc_std = results_df['Lc'].std()
    outliers = results_df[
        (results_df['Lc'] < lc_mean - 2*lc_std) |
        (results_df['Lc'] > lc_mean + 2*lc_std)
    ]

    if len(outliers) > 0:
        print(f"\n⚠ {len(outliers)} Lc outliers (>2σ from mean):")
        for idx, row in outliers.iterrows():
            print(f"  - {row['experiment']}: Lc = {row['Lc']:.4f} m")

    # Check Tc detection
    no_tc = results_df[results_df['Tc'].isna()]
    if len(no_tc) > 0:
        print(f"\n⚠ {len(no_tc)} experiments with no Tc detected:")
        for idx, row in no_tc.iterrows():
            print(f"  - {row['experiment']}")

    print("="*60)

# Run QC
quality_control(results_df)
```

### Export Statistics

```python
def export_statistics(results_df, output_file="batch_statistics.txt"):
    """Export detailed statistics."""

    with open(output_file, 'w') as f:
        f.write("BATCH ANALYSIS STATISTICS\n")
        f.write("="*70 + "\n\n")

        f.write(f"Total experiments: {len(results_df)}\n\n")

        # For each parameter
        for param in ['Lc', 'gamma', 'beta']:
            f.write(f"{param}:\n")
            f.write(f"  Mean: {results_df[param].mean():.6e}\n")
            f.write(f"  Median: {results_df[param].median():.6e}\n")
            f.write(f"  Std: {results_df[param].std():.6e}\n")
            f.write(f"  Min: {results_df[param].min():.6e}\n")
            f.write(f"  Max: {results_df[param].max():.6e}\n")
            f.write(f"  CV: {(results_df[param].std()/results_df[param].mean())*100:.2f}%\n\n")

    print(f"✓ Statistics exported to {output_file}")

export_statistics(results_df)
```

## Visualization

### Comparison Plots

```python
import matplotlib.pyplot as plt

# Histogram of Lc values
plt.figure(figsize=(10, 6))
plt.hist(results_df['Lc'], bins=20, edgecolor='black', alpha=0.7)
plt.axvline(results_df['Lc'].mean(), color='r', linestyle='--',
            label=f"Mean: {results_df['Lc'].mean():.4f} m")
plt.xlabel('Convergence Length Lc (m)')
plt.ylabel('Frequency')
plt.title('Distribution of Lc Across Experiments')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('lc_distribution.png', dpi=300)

# Scatter plot: Lc vs gamma
plt.figure(figsize=(8, 6))
plt.scatter(results_df['Lc'], results_df['gamma'])
plt.xlabel('Lc (m)')
plt.ylabel('γ (s⁻¹)')
plt.title('Lc vs Gamma')
plt.grid(alpha=0.3)
plt.savefig('lc_vs_gamma.png', dpi=300)
```

## Troubleshooting

### Common Issues

**"Centerlines file not found"**
- Check CSV filename matches: `{exp_name}_centerlines.csv`
- Verify file is in experiment folder

**"No intersection found"**
- Masks may have insufficient overlap
- Check mask quality and plant position

**Script runs but no output**
- Check `OUTPUT_DIR` exists and is writable
- Verify experiment directory structure

### Debug Mode

Add verbose output:

```python
DEBUG = True

if DEBUG:
    print(f"Loading: {centerlines_file}")
    print(f"Data shape: {data.shape}")
    print(f"Frames: {data['frame'].nunique()}")
```

## Next Steps

- Create [custom workflows](/examples/workflows/) for specific analyses
- Implement [parallel processing](#parallel-processing) for large datasets
- Export results for statistical analysis in R or other tools
