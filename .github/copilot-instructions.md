# Copilot Instructions

## Project Overview

Documentation site for the **Tropism Toolset** — a Python library for analyzing plant mechanics and tropisms. Built with Astro 5.6+ and Starlight theme. The actual Python library lives in `../tropism-toolset/src/`, this repo is **documentation only**.

**Core domain**: Plant gravitropism analysis — extracting physical constants (Young's modulus, convergence length, proprioceptive/gravitropic sensitivity) from time-lapse image centerline data.

## Development

```bash
npm run dev       # Dev server at localhost:4321
npm run build     # Production build to ./dist/
npm run preview   # Preview production build
```

No test/lint scripts configured. Astro hot-reloads automatically during dev.

## Content Architecture

### File → Route Mapping

Content in `src/content/docs/` maps directly to routes:
- `src/content/docs/index.mdx` → `/`
- `src/content/docs/guides/installation.mdx` → `/guides/installation/`

**File organization principle**: Hierarchical sections (Getting Started, User Guides, Mathematical Models, API Reference, Examples, Advanced) defined in `astro.config.mjs` lines 23-73.

### Documentation Sections

1. **Getting Started**: Installation, quickstart, core concepts (tropisms, convergence length, coordinate systems)
2. **User Guides**: Data handling, geometric analysis, constants extraction, growth analysis, steady state, visualization
3. **Mathematical Models**: Chauvet/Moulia/Goriely models (collapsed by default in sidebar)
4. **API Reference**: Auto-generated from `/reference/` directory using Starlight's `autogenerate`
5. **Examples**: Workflows, Jupyter notebooks
6. **Advanced**: Batch processing, customization

### Content Frontmatter

Required fields:
```yaml
---
title: Page Title
description: Short description for SEO
---
```

Optional:
- `template: splash` for hero pages (see `index.mdx`)

## Mathematical Notation

**Critical**: LaTeX math rendering via remark-math + rehype-katex (configured in `astro.config.mjs`).

**Inline math**: `$L_c = \gamma/\beta$`

**Block math**: 
```
$$
\theta(s) = B_\ell + A(1 - e^{-(s-s_0)/L_c})
$$
```

**Custom Equation Component**: For bordered equations with labels:
```astro
import Equation from '../../../components/Equation.astro';

<Equation label="Eq. 1">
$$
\gamma = \frac{1}{T_c \cdot \Delta t}
$$
</Equation>
```

Component adds styling: border, padding, centered layout. Located at `src/components/Equation.astro`.

## Starlight Components

Available from `@astrojs/starlight/components`:
- `<Card>` and `<CardGrid>` — feature showcases (see `index.mdx`)
- `<Tabs>` and `<TabItem>` — tabbed content (see `installation.mdx`)
- Hero sections with CTA buttons (splash template)

## Code Examples Convention

**Import pattern**: Always use `from constants import function_name`

Example:
```python
from constants import get_angles, fit_Lc, get_gamma, get_beta
import pandas as pd

# Load data
data = pd.read_csv("centerlines.csv")

# Analysis
angles = get_angles(data)
Lc = fit_Lc(arclengths, angles)
gamma = get_gamma(Tc=120, period=900)
```

**Units**: Always specify in parameter descriptions:
- Length: m (meters)
- Time: s (seconds)
- Angles: rad (radians)
- Sensitivity: γ in s⁻¹, β in m⁻¹
- Pressure: Pa (Pascals)

## Domain-Specific Patterns

### Coordinate Systems
Two systems referenced throughout:
1. **Cartesian (x, y)**: Raw pixel coordinates from images
2. **Arc-length (s, θ)**: Natural coordinates along plant centerline

Transform: `s, theta = x_y_to_s_theta(x_coords, y_coords, px_to_m)`

### Key Parameters
- **Lc (convergence length)**: Characteristic spatial scale, units: m
- **γ (gamma)**: Proprioceptive sensitivity, units: s⁻¹
- **β (beta)**: Gravitropic sensitivity, units: m⁻¹, relationship: β = γ/Lc
- **Tc**: Time to steady state, units: frames

### Data Structure
Standard CSV format for centerline data:
```csv
frame,x,y
0,245.0,1088.0
0,246.2,1076.3
```

Ordered from base to tip, sequential frames.

## Styling

**TailwindCSS 4.1+**: Configured via Vite plugin (`astro.config.mjs` lines 76-78), not PostCSS.

**Custom CSS**: `src/styles/custom.css` contains KaTeX fixes:
- Hide duplicate `.katex-html` rendering
- Reset margins for math elements

**Starlight CSS variables**: Use `var(--sl-color-*)` for theme-aware colors (see `Equation.astro`).

## Configuration Details

**`astro.config.mjs`**:
- Markdown: remark-math + rehype-katex plugins (lines 10-14)
- Starlight title: "Tropism Toolset"
- GitHub social link: https://github.com/merozlab
- Custom CSS: `./src/styles/custom.css`

**`src/content.config.ts`**:
- Uses Starlight's `docsLoader` and `docsSchema` — no custom schema needed

**`package.json`**:
- No workspace configuration (single project)
- Key deps: `@astrojs/starlight`, `astro`, `katex`, `rehype-katex`, `remark-math`, `sharp`, `tailwindcss`

## Common Tasks

### Adding a new guide page
1. Create `.md` or `.mdx` in `src/content/docs/guides/`
2. Add frontmatter with `title` and `description`
3. Reference in `astro.config.mjs` sidebar if needed (or rely on autogenerate)

### Adding math equations
Use inline `$...$` or block `$$...$$` syntax. Wrap in `<Equation>` component for styled presentation.

### Adding API reference
Place file in `src/content/docs/reference/` — automatically picked up by `autogenerate: { directory: 'reference' }`.

### Using Starlight components
Import at top of MDX: `import { Card, CardGrid } from '@astrojs/starlight/components';`

## Gotchas

1. **Math rendering**: KaTeX requires proper escaping in Markdown. Test locally before committing.
2. **Import paths**: Astro components use relative paths with file extensions (`.astro`, `.mdx`).
3. **Sidebar order**: Manually specified in `astro.config.mjs`, not filesystem-based.
4. **Collapsed sections**: Mathematical Models/API Reference/Examples/Advanced sections default to collapsed.
5. **Python 3.13 requirement**: Documented in installation — relevant for code examples.

## Related Repositories

- **Python library**: `../tropism-toolset/` — contains actual implementation
- **Python source**: `../tropism-toolset/src/` — where API functions are defined

When documenting API functions, refer to the actual Python implementation in sibling directory for accuracy.
