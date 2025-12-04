---
title: Steady State Detection
description: Automatically detect when your system reaches steady state.
---

Learn how to identify when a plant organ has reached steady-state configuration after gravitropic stimulation.

## What is Steady State?

**Steady state** occurs when the plant organ has completed its gravitropic response and maintains a stable configuration. Key indicators:

- Length stops increasing (growth cessation)
- Angle profile remains constant
- Mask shape stabilizes

Identifying the steady-state time ($T_c$) is another possible way to calculate $\gamma$.

## Time Series Analysis

### From Length Data

Detect when length reaches a plateau:

```python
from constants import find_steady_state, get_arclengths
import pandas as pd
import numpy as np

# Load data and extract lengths
data = pd.read_csv("centerlines.csv")
px_to_m = 100

lengths = []
for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]
    arclengths = get_arclengths(frame_data) / px_to_m  # Convert to meters
    lengths.append(arclengths[-1])

lengths = np.array(lengths)

# Find steady state
Tc, lengths_smooth = find_steady_state(
    lengths,
    threshold_factor=0.15,
    min_steady_duration=20,
    show=True
)

if Tc is not None:
    print(f"Steady state begins at frame: {Tc}")
    minutes_per_frame = 15
    print(f"Time to steady state: {Tc * minutes_per_frame:.1f} minutes")
else:
    print("No steady state detected")
```

### Parameters

- **threshold_factor** (default: 0.15): Sensitivity for change detection
  - Lower values = stricter criterion (fewer false positives)
  - Higher values = looser criterion (earlier detection)

- **min_steady_duration** (default: 20): Minimum frames of stability
  - Higher values = more confident detection
  - Lower values = earlier detection (may be premature)

### Custom Threshold

```python
# Stricter criterion
Tc_strict, _ = find_steady_state(
    lengths,
    threshold_factor=0.10,
    min_steady_duration=30,
    show=True
)

# Looser criterion
Tc_loose, _ = find_steady_state(
    lengths,
    threshold_factor=0.20,
    min_steady_duration=10,
    show=True
)

print(f"Strict: frame {Tc_strict}")
print(f"Loose: frame {Tc_loose}")
```

## Mask-Based Steady State

Use binary masks to detect shape stability:

```python
from constants import find_steady_state_from_masks
from pathlib import Path

# Directory containing mask files
mask_dir = Path("data/experiment_masks")

# Find steady state from masks
Tc_mask, stability_scores = find_steady_state_from_masks(
    mask_dir,
    method="overlap",
    threshold_factor=0.15,
    min_steady_duration=20,
    show=True
)

if Tc_mask is not None:
    print(f"Mask-based steady state: frame {Tc_mask}")
else:
    print("No steady state detected from masks")
```

### Stability Metrics

**overlap** (default): Fraction of non-overlapping pixels between consecutive frames
- More stable = lower score
- Good for overall shape changes

**hausdorff**: Maximum distance between mask boundaries
- More stable = lower score
- Sensitive to local changes

**centroid_shift**: Distance centroid moves between frames
- More stable = lower score
- Good for translation detection

```python
# Compare methods
for method in ['overlap', 'hausdorff', 'centroid_shift']:
    Tc, scores = find_steady_state_from_masks(
        mask_dir,
        method=method,
        show=False
    )
    print(f"{method:15s}: Tc = {Tc}")
```

## Visual Inspection

Always visually verify automated detection:

```python
from constants import display_steady_state_analysis

# Visualize detection on length data
display_steady_state_analysis(
    data=lengths,
    data_smoothed=lengths_smooth,
    steady_start_orig=None,
    steady_start_smooth=Tc,
    title="Length-Based Steady State Detection",
    ylabel="Length (m)",
    time_unit="Frame"
)
```

For mask analysis:

```python
from constants import display_mask_stability_analysis

display_mask_stability_analysis(
    stability_scores=stability_scores,
    steady_start=Tc_mask,
    title="Mask Stability Analysis",
    ylabel="Overlap Score",
    time_unit="Frame",
    method="overlap"
)
```

## Manual Determination

For cases where automatic detection fails:

```python
import matplotlib.pyplot as plt

# Plot length over time
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(lengths, 'o-', markersize=4)
ax.set_xlabel('Frame', fontsize=12)
ax.set_ylabel('Length (m)', fontsize=12)
ax.set_title('Organ Length Over Time - Visual Inspection', fontsize=14)
ax.grid(alpha=0.3)

# Add candidate Tc lines
candidates = [100, 120, 140]
for Tc_candidate in candidates:
    ax.axvline(Tc_candidate, color='red', linestyle='--', alpha=0.5,
               label=f'Tc = {Tc_candidate}')

ax.legend()
plt.tight_layout()
plt.show()

# Choose based on visual inspection
Tc_manual = 120
print(f"Manually determined Tc: {Tc_manual}")
```

## Angle-Based Detection

Detect when tip angle stabilizes:

```python
from constants import get_angles

# Extract tip angle over time
tip_angles = []
for frame in sorted(data['frame'].unique()):
    frame_data = data[data['frame'] == frame]
    angles = get_angles(frame_data)
    tip_angles.append(angles[-1])

tip_angles = np.array(tip_angles)

# Find steady state in angle data
Tc_angle, angles_smooth = find_steady_state(
    tip_angles,
    threshold_factor=0.15,
    min_steady_duration=20,
    show=True
)

print(f"Angle-based Tc: {Tc_angle}")
```

## Combining Multiple Criteria

Use multiple detection methods and choose conservatively:

```python
# Collect all detections
detections = {}

# Length-based
Tc_length, _ = find_steady_state(lengths, show=False)
if Tc_length is not None:
    detections['length'] = Tc_length

# Angle-based
Tc_angle, _ = find_steady_state(tip_angles, show=False)
if Tc_angle is not None:
    detections['angle'] = Tc_angle

# Mask-based
Tc_mask, _ = find_steady_state_from_masks(mask_dir, show=False)
if Tc_mask is not None:
    detections['mask'] = Tc_mask

# Summary
print("Steady state detections:")
for method, Tc_val in detections.items():
    print(f"  {method:10s}: frame {Tc_val}")

# Use the latest (most conservative)
Tc_final = max(detections.values())
print(f"\nUsing conservative estimate: Tc = {Tc_final}")
```

## Verification

### Check Steady State Criteria

```python
# After steady state, values should be relatively constant
steady_region = lengths[Tc:] if Tc else lengths[-50:]

mean_val = steady_region.mean()
std_val = steady_region.std()
cv = (std_val / mean_val) * 100  # Coefficient of variation

print(f"Steady state statistics:")
print(f"  Mean: {mean_val:.6f} m")
print(f"  Std: {std_val:.6f} m")
print(f"  CV: {cv:.2f}%")

if cv < 2.0:
    print("✓ Good steady state (CV < 2%)")
elif cv < 5.0:
    print("⚠ Acceptable steady state (2% ≤ CV < 5%)")
else:
    print("✗ Poor steady state (CV ≥ 5%) - may need manual inspection")
```

### Create Validation Video

Generate a video showing the transition to steady state:

```python
from constants import create_video_with_colored_frames

# Tc = frame where steady state begins
create_video_with_colored_frames(
    mask_dir=mask_dir,
    output_file="steady_state_analysis.mp4",
    Tc=Tc,
    framerate=6
)

print(f"✓ Video created: steady_state_analysis.mp4")
print(f"  White frames: growth phase (0-{Tc-1})")
print(f"  Red frames: steady state ({Tc}+)")
```

## Handling Edge Cases

### No Steady State Detected

If the system hasn't reached steady state:

```python
if Tc is None:
    print("⚠ Warning: No steady state detected")
    print("Possible reasons:")
    print("  1. Experiment duration too short")
    print("  2. Plant still actively growing")
    print("  3. Threshold too strict")
    print("\nOptions:")
    print("  - Use looser threshold_factor (e.g., 0.20)")
    print("  - Reduce min_steady_duration")
    print("  - Continue experiment longer")
    print("  - Use manual determination")

    # Use last frame as approximation
    Tc_approx = len(lengths) - 20
    print(f"\nUsing approximation: Tc ≈ {Tc_approx}")
```

### Multiple Plateaus

If there are multiple stable regions:

```python
# Find all local plateaus
from scipy.signal import find_peaks

# Invert and find peaks in rate of change
dL = np.abs(np.diff(lengths_smooth))
peaks, properties = find_peaks(-dL, prominence=0.0001, width=20)

print(f"Found {len(peaks)} potential steady states:")
for i, peak in enumerate(peaks):
    print(f"  Plateau {i+1}: frame {peak}")

# Use the last one (final steady state)
if len(peaks) > 0:
    Tc_final = peaks[-1]
    print(f"\nUsing final plateau: Tc = {Tc_final}")
```

## Best Practices

1. **Always visualize**: Use `show=True` to inspect detection
2. **Multiple methods**: Compare length, angle, and mask-based detection
3. **Verify stability**: Check coefficient of variation in steady region
4. **Document choice**: Record why you chose a particular Tc
5. **Conservative**: When in doubt, choose later Tc (more conservative)

## Next Steps

- Use Tc to [calculate gamma (γ)](/guides/constants-extraction/#calculating-gamma-γ)
- Combine with [growth analysis](/guides/growth-analysis/) for complete temporal profile
- Apply to [batch processing](/advanced/batch-processing/) workflows
