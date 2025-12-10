---
title: Preprocessing Workflow
description: How to prepare and preprocess data for analysis using Tropism Toolset.
---

This guide details the preprocessing workflow for standardizing and validating plant centerline data before in-depth analysis. This is crucial for ensuring data quality and consistency across experiments.

### Features
- **Batch Processing**: Recursively finds and processes experiments within specified directories.
- **Data Standardization**: Applies geometric transformations (horizontal/vertical flips, reversing point order) to ensure all centerlines adhere to a consistent coordinate system (e.g., tip-to-base, left-to-right).
- **Quality Validation**: Integrates growth rate calculations as an initial check for data integrity, helping to identify potential tracking errors or abnormal growth patterns.
- **Export**: Optionally saves the modified and standardized centerline data to new CSV files, creating a clean dataset for subsequent analytical steps.

### Complete Script

The core logic for this workflow is encapsulated in the `preprocessing.py` script located in the `examples/workflows/` directory of the `tropism-toolset` package. A simplified version of the script is shown below to highlight its key components:

```python
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tropism_toolset import (
    fit_growth_rate,
    plot_centerline_data,
    display_centerline_endpoints,
    flip_centerlines_horizontal,
    flip_centerlines_vertical,
    reverse_centerline_order,
    get_lengths_from_centerlines,
    convert_centerline_units
)
# Assumes utils.py is in the same directory, containing discover_experiments
from utils import discover_experiments

def process_experiment(exp_name, centerlines_list, centerline_files, config, output_dir, save_modified=False, modified_parent_dir=None):
    """
    Process a single experiment: apply transformations and check growth rate.
    """
    results = []
    
    for plant_idx, centerlines in enumerate(centerlines_list):
        print(f"  Processing plant {plant_idx}...")
        
        # 1. Coordinate Transformations
        if config['flip_h']:
            centerlines = flip_centerlines_horizontal(centerlines)
        if config['flip_v']:
            centerlines = flip_centerlines_vertical(centerlines)
        if config['reverse']:
            centerlines = reverse_centerline_order(centerlines)
            
        # 2. Save Modified Data (Optional)
        if save_modified and modified_parent_dir:
            original_file = centerline_files[plant_idx]
            modified_exp_dir = modified_parent_dir / exp_name
            modified_exp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save transformed data
            out_path = modified_exp_dir / original_file.name
            centerlines.to_csv(out_path, index=False)
            print(f"  Saved modified to: {out_path}")

        # 3. Growth Rate Validation
        # Convert to physical units (seconds)
        frame_to_s = config["frame_duration"] * 60
        centerlines_phys = convert_centerline_units(
            centerlines, 
            px_to_m=None, # Assuming data is already in meters, if not, provide px_to_m
            frame_to_s=frame_to_s
        )
        
        # Calculate lengths
        length_df = get_lengths_from_centerlines(
            centerlines_phys,
            smooth=config["smooth"],
            window_length=config["window_length"], # Added window_length and polyorder
            polyorder=config["polyorder"]
        )

        # Fit growth rate
        growth_rate = fit_growth_rate(
            data=length_df,
            save_path=output_dir / f"{exp_name}_plant{plant_idx}_growth.png"
        )
        
        results.append({
            "experiment": exp_name,
            "plant": plant_idx,
            "growth_rate_m_s": growth_rate,
            "final_length": length_df["length (meters)"].iloc[-1]
        })
        
    return results

if __name__ == "__main__":
    # Simplified main block for demonstration
    parser = argparse.ArgumentParser(description="Batch Preprocessing")
    parser.add_argument("directory", help="Data directory")
    parser.add_argument("--save", action="store_true", help="Save modified files")
    parser.add_argument("--flip-horizontal", action="store_true", dest="flip_h")
    parser.add_argument("--flip-vertical", action="store_true", dest="flip_v")
    parser.add_argument("--reverse-order", action="store_true", dest="reverse")
    parser.add_argument("--frame-duration", type=float, default=15.0, help="Frame duration in minutes")
    parser.add_argument("--smooth", action="store_true", default=True, help="Apply Savitzky-Golay smoothing")
    parser.add_argument("--window-length", type=int, default=5, help="Savitzky-Golay window length")
    parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--exclude", nargs="*", default=[], help="Experiment names to exclude")
    parser.add_argument("--only", nargs="*", default=[], help="Only process these experiment names")
    
    args = parser.parse_args()
    
    # Example config (should be dynamically created in full script)
    config = {
        "flip_h": args.flip_h,
        "flip_v": args.flip_v,
        "reverse": args.reverse,
        "frame_duration": args.frame_duration,
        "smooth": args.smooth,
        "window_length": args.window_length,
        "polyorder": args.polyorder,
    }
    
    # This part of the main script would typically discover experiments and loop through them.
    # For a complete example, refer to `examples/workflows/preprocessing.py`
    print("Running simplified preprocessing script...")
    # Example: process_experiment(...) would be called here for each plant
```

### Usage

The `preprocessing.py` script is designed to be run from the command line, offering various options to customize the preprocessing steps.

```bash
# Basic validation of all experiments found in 'data/experiments'
python preprocessing.py data/experiments

# Standardize: Flip centerlines horizontally, reverse point order (base to tip),
# and SAVE the modified centerlines to new CSV files in a 'modified_data' folder.
python preprocessing.py data/experiments \
    --flip-horizontal \
    --reverse-order \
    --save

# Filter: Process only specific experiments while excluding others.
# This example processes all experiments except 'exp_03' and 'exp_09'.
python preprocessing.py data/experiments --exclude exp_03 exp_09

# Specify frame duration for growth rate calculations
python preprocessing.py data/experiments --frame-duration 20
```