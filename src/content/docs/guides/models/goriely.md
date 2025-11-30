---
title: Goriely Model
description: Geometric growth tensor approach to tropisms.
---

The Goriely model uses differential geometry and growth tensor analysis to describe plant tropisms.

## Overview

This model takes a geometric approach based on:
- Growth tensors
- Differential geometry of curves
- Continuous growth fields

## Implementation

The primary analysis notebook for this model is `Goriely_constants.ipynb` in the repository.

### Theoretical Framework

The model describes growth-induced curvature changes using geometric quantities and growth tensors.

## Using the Notebook

### Open the Notebook

```bash
cd /path/to/constants
poetry shell
jupyter lab Goriely_constants.ipynb
```

### Analysis Steps

1. Load centerline data in arc-length coordinates
2. Calculate geometric quantities (curvature, torsion)
3. Apply the Goriely framework
4. Extract model parameters

## When to Use This Model

The Goriely model is particularly useful when:
- You have 3D growth data
- Geometric interpretations are preferred
- Studying complex growth patterns
- Analyzing torsion in addition to planar curvature

## References

- Goriely, A., & Tabor, M. (2013). Biomechanical models of hyphal growth in actinomycetes. *Journal of Theoretical Biology*, 222, 211-218.
- Goriely, A. (2017). *The Mathematics and Mechanics of Biological Growth*. Springer.

## See Also

- [Moulia Model](/guides/models/moulia/) - Proprioception-based framework
- [Chauvet Model](/guides/models/chauvet/) - Inclination sensing model
- [Geometric Analysis](/guides/geometric-analysis/) - Computing geometric quantities
