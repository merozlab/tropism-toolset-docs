---
title: Tip Angle Analysis Workflow
description: Workflow for analyzing tip angle dynamics, visualizing reorientation, and fitting mathematical models.
---

This guide details the **Tip Angle Analysis Workflow**, which focuses on quantifying the dynamics of organ reorientation over time. This workflow is implemented in the `tip_angle_analysis.py` script.

## Overview

The tip angle analysis script (`tip_angle_analysis.py`) performs the following:
1.  **Tip Angle Extraction**: Calculates the average angle of the last N% (default 20%) of the centerline for every frame using a linear fit.
2.  **Time Series Construction**: Aggregates angle data across all experiments to compute the mean tip angle trajectory with standard error bands.
3.  **Visualization**:
    *   **Tip Percentage**: Visualizes exactly which part of the organ is considered the "tip" for angle calculation.
    *   **Mean Trajectory**: Plots the average reorientation curve across all samples.
4.  **Model Fitting**: Fits a combined **Exponential Rise + Damped Sinusoid** model to the averaged data to characterize the gravitropic response.
    *   **Model**: $\theta(t) = A(1 - e^{-kt}) + B(1 - e^{-mt})e^{-lt}\sin(\omega t + p)$
    *   Captures both the initial exponential response and any subsequent oscillations (overshoot/correction).

## Usage

```bash
python examples/workflows/tip_angle_analysis.py <directory> [options]
```

## Model Components

The workflow fits a sophisticated phenomenological model to capture complex reorientation dynamics:

1.  **Exponential Component** ($A(1 - e^{-kt})$): Represents the primary gravitropic response, driving the angle towards a setpoint.
2.  **Damped Sinusoid Component** ($B(1 - e^{-mt})e^{-lt}\sin(\omega t + p)$): Captures oscillations, overshoots, and corrections typical of proprioceptive feedback or hunting behaviors.

## Example Workflow

### 1. Basic Analysis
Analyze all experiments using the default top 20% of the organ as the tip.

```bash
python examples/workflows/tip_angle_analysis.py data/full_experiment
```

### 2. Adjust Tip Definition
Use the top 30% of the organ for a more robust (but potentially less sensitive) angle measurement.

```bash
python examples/workflows/tip_angle_analysis.py data/full_experiment --tip-percent 30
```

### 3. Time Limits
Limit the analysis to the first 48 hours to focus on the initial response.

```bash
python examples/workflows/tip_angle_analysis.py data/full_experiment --max-time 48
```

## Outputs

The script creates a timestamped folder in `output/` containing:
- **`tip_angle_mean_across_experiments.png`**: Plot of the mean tip angle vs. time with SEM bands.
- **`tip_angle_fitted_Average_Across_All_Experiments.png`**: Visualization of the model fit, decomposing it into exponential and oscillatory components, and showing residuals.
- **`tip_percentage_visualization.png`**: Visual check showing the tip region (red) vs base (blue) for a sample frame from each experiment.
- **`tip_angle_fit_parameters.csv`**: The fitted model parameters ($A, k, B, l, \omega, p, m, R^2$).

## CLI Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `directory` | str | Required | Path to the directory containing experiment subdirectories. |
| `--exclude` | list | `[]` | Experiment names to exclude. |
| `--tip-percent` | int | `20` | Percentage of centerline points to use for linear fit of the tip angle. |
| `--frame-duration` | float | `15.0` | Frame duration in minutes. |
| `--max-time` | float | `120.0` | Maximum time in hours to include in the analysis. |