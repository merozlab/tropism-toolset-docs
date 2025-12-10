---
title: Growth Analysis Workflow
description: Workflow for analyzing plant growth rates, elongation, and growth zone dynamics.
---

This guide details the **Growth Analysis Workflow**, used to quantify elongation rates, total growth, and the size of the growth zone. The workflow is implemented in the `growth_analysis.py` script.

## Overview

The growth analysis script (`growth_analysis.py`) performs the following for each experiment:
1.  **Converts units**: Transforms frame numbers to seconds based on frame duration.
2.  **Fits Convergence Length ($L_c$)**: Determines the characteristic length scale ($L_c$) and the starting position of the growth zone ($x_0$) from the steady-state profile.
3.  **Calculates Length Over Time**: Extracts the total length of the organ for every frame.
4.  **Fits Growth Rate**: Fits a linear model to the length-vs-time data to extract the growth rate ($dL/dt$).
5.  **Computes Metrics**: Calculates total absolute growth, percent growth, and growth zone length ($L_{gz} = L_{final} - x_0$).
6.  **Aggregates Statistics**: Generates summary statistics across all experiments.
7.  **Visualizes Results**: Creates bar plots and histograms comparing growth metrics across experiments.

## Usage

```bash
python examples/workflows/growth_analysis.py <directory> [options]
```

### Key Metrics

- **Growth Rate ($m/s$)**: The slope of the length vs. time curve.
- **Absolute Growth ($m$)**: Final length minus initial length.
- **Percent Growth (%)**: Relative increase in length.
- **Growth Zone Length ($L_{gz}$)**: The length of the active growth region, derived from the $L_c$ fit parameter $x_0$ (transition point).

## Example Workflow

### 1. Standard Analysis
Analyze all experiments in a folder with default settings (15 min/frame).

```bash
python examples/workflows/growth_analysis.py data/full_experiment
```

### 2. Custom Frame Duration
If your imaging interval was different (e.g., 20 minutes):

```bash
python examples/workflows/growth_analysis.py data/full_experiment --frame-duration 20
```

### 3. Smoothing Control
Adjust the Savitzky-Golay smoothing parameters if your centerline data is noisy.

```bash
python examples/workflows/growth_analysis.py data/full_experiment \
    --window-length 7 \
    --polyorder 3
```

## Outputs

The script creates a timestamped folder in `output/` containing:
- **`growth_analysis_results.csv`**: Detailed metrics for every plant.
- **`summary_statistics.txt`**: Aggregated mean/std for all metrics.
- **`growth_analysis_visualization.png`**: 4-panel figure showing growth rates, total growth, and distributions.
- **Individual plots**: Growth rate fits and $L_c$ fits for each plant.

## CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `directory` | str | Required | Path to the directory containing experiment subdirectories. |
| `--exclude` | list | `[]` | Experiment names to exclude. |
| `--frame-duration` | float | `15.0` | Frame duration in minutes. |
| `--smooth` | flag | `True` | Apply Savitzky-Golay smoothing to length data. |
| `--window-length` | int | `5` | Window length for smoothing. |
| `--polyorder` | int | `2` | Polynomial order for smoothing. |
| `--fit-params` | str | `None` | Path to JSON file containing custom initial guesses/bounds for $L_c$ fitting. |