---
title: Notebook Guides
description: Guide to the Jupyter notebooks included in the repository.
---

The repository includes several specialized Jupyter notebooks for different analysis tasks.

## Analysis Notebooks

### Moulia_constants.ipynb

**Purpose:** Extract physical constants using the Moulia proprioception model

**When to use:**
- Combined gravitropic and proprioceptive analysis
- Temporal dynamics of curvature
- Full parameter estimation (β, γ)

**Key outputs:**
- Proprioceptive sensitivity (γ)
- Gravitropic sensitivity (β)
- Model validation plots

---

### Goriely_constants.ipynb

**Purpose:** Geometric analysis using growth tensor framework

**When to use:**
- Geometric interpretation of growth
- Complex 3D growth patterns
- Theoretical geometric analysis

**Key outputs:**
- Growth tensor components
- Geometric parameters
- Curvature evolution

---

### Young's_modulus.ipynb

**Purpose:** Estimate mechanical properties from bending experiments

**When to use:**
- Mechanical property characterization
- Stiffness measurements
- Force-deformation analysis

**Key outputs:**
- Young's modulus (E)
- Bending stiffness
- Mechanical model fits

---

### pde.ipynb

**Purpose:** Partial differential equation modeling of growth

**When to use:**
- Continuous spatiotemporal modeling
- PDE-based growth description
- Advanced theoretical analysis

**Key outputs:**
- PDE solutions
- Spatiotemporal patterns
- Model parameters

---

### geometric_constants.ipynb

**Purpose:** Extract geometric parameters from plant organs

**When to use:**
- Basic geometric characterization
- Length, curvature, angle analysis
- Initial data exploration

**Key outputs:**
- Total lengths
- Curvature profiles
- Angle distributions

---

### Chauvet_constants.ipynb

**Purpose:** Implement the Chauvet gravitropism model

**When to use:**
- Growth rate analysis
- Angular velocity calculation
- Simple gravitropic sensitivity (β̃)

**Key outputs:**
- Growth rate (dL/dt)
- Angular velocity (dθ/dt)
- Chauvet β̃ parameter

---

### Bastien_constants.ipynb

**Purpose:** Analysis following Bastien et al. framework

**When to use:**
- Growth zone characterization
- Following Bastien methodology
- Comparative studies

**Key outputs:**
- Growth zone length
- Bastien model parameters
- Comparative metrics

---

## Growth Analysis Notebooks

### growth_rate.ipynb

**Purpose:** Interactive growth rate analysis

**When to use:**
- Detailed growth dynamics
- Single experiment deep dive
- Custom growth models

**Key features:**
- Interactive parameter tuning
- Multiple fitting options
- Visualization customization

---

### growth_parameters.ipynb

**Purpose:** Comprehensive growth parameter extraction

**When to use:**
- Multiple growth metrics
- Growth zone analysis
- Mask-based growth regions

**Key outputs:**
- Growth rates
- Growth zone dimensions
- Temporal growth patterns

---

## Notebook Workflow

### 1. Start with Geometric Constants

```bash
jupyter lab geometric_constants.ipynb
```

- Load your data
- Visualize centerlines
- Calculate basic geometric properties
- Identify issues with data quality

### 2. Choose Your Model

Depending on your research question:

**For simple analysis:**
→ `Chauvet_constants.ipynb`

**For comprehensive tropism study:**
→ `Moulia_constants.ipynb`

**For mechanical properties:**
→ `Young's_modulus.ipynb`

### 3. Deep Dive into Growth

```bash
jupyter lab growth_rate.ipynb
jupyter lab growth_parameters.ipynb
```

Analyze growth dynamics in detail.

### 4. Advanced Modeling

For theoretical work:
```bash
jupyter lab pde.ipynb
jupyter lab Goriely_constants.ipynb
```

## Customizing Notebooks

### Update File Paths

At the top of each notebook, update paths:

```python
# Update these paths for your data
data_file = "data/your_experiment_centerlines.csv"
mask_dir = Path("data/your_experiment_masks")
output_dir = Path("results")
```

### Modify Parameters

Common parameters to adjust:

```python
# Calibration
px_to_m = 100  # Your calibration factor

# Temporal
minutes_per_frame = 15  # Your frame interval
period = minutes_per_frame * 60

# Plant dimensions
R = 0.00145  # Plant radius in meters

# Fitting
crop_start = 0  # Points to crop from base
crop_end = 3    # Points to crop from tip
```

### Save Custom Versions

```bash
# Make a copy for your experiment
cp Moulia_constants.ipynb my_experiment_analysis.ipynb
```

## Best Practices

### 1. Run Cells Sequentially

Execute cells in order from top to bottom.

### 2. Restart Kernel When Needed

If variables get messy:
- Kernel → Restart & Clear Output
- Re-run from the beginning

### 3. Save Outputs

Export plots and results:

```python
# Save figure
fig.savefig('results/my_plot.png', dpi=300, bbox_inches='tight')

# Save data
results_df.to_csv('results/my_results.csv', index=False)
```

### 4. Document Changes

Add markdown cells to explain your modifications:

```markdown
## My Analysis Notes

Changed crop_end from 3 to 5 because of noisy tip region.
Using custom Tc = 125 based on manual inspection.
```

## Troubleshooting

### Import Errors

If `from constants import ...` fails:

```python
# Add this at the top of the notebook
import sys
sys.path.append('/path/to/constants/src')
```

Or ensure your Jupyter kernel is using the correct environment (see [Installation Guide](/guides/installation/#jupyter-setup)).

### Kernel Crashes

For large datasets, increase Jupyter memory:

```bash
jupyter lab --NotebookApp.iopub_data_rate_limit=1e10
```

### Missing Dependencies

Install in the notebook:

```python
!pip install missing-package
```

## Creating New Notebooks

Start with a template:

```python
# Notebook header
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import toolkit
from constants import *

# Configuration
px_to_m = 100
period = 15 * 60  # seconds
data_file = "data/experiment.csv"

# Your analysis
# ...
```

## See Also

- [Complete Workflows](/examples/workflows/) - Script-based analyses
- [API Reference](/reference/geometric-calculations/) - Function documentation
- [Batch Processing](/advanced/batch-processing/) - Automating multiple experiments
