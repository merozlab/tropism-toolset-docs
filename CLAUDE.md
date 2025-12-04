# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a documentation site for the **Tropism Toolset**, a Python toolkit for analyzing plant mechanics and tropisms. Built with Astro and Starlight, it documents tools for extracting physical constants (Young's modulus, convergence length, proprioceptive/gravitropic sensitivity) from plant image data. The actual code sits in `../tropism-toolset` and speficically in `../tropism-toolset/src`.

## Development Commands

```bash
# Install dependencies
npm install

# Start development server (runs at localhost:4321)
npm run dev

# Build for production (outputs to ./dist/)
npm run build

# Preview production build locally
npm run preview

# Run Astro CLI commands
npm run astro -- [command]
```

## Architecture

### Framework Stack
- **Astro 5.6+**: Static site generator with content collections
- **Starlight**: Documentation theme and framework
- **TailwindCSS 4.1+**: Styling (configured via Vite plugin)
- **KaTeX**: Math rendering (via rehype-katex and remark-math)

### Content Structure

All documentation lives in `src/content/docs/` as Markdown/MDX files. The file structure maps directly to routes:
- `src/content/docs/index.mdx` → homepage
- `src/content/docs/guides/installation.md` → `/guides/installation/`
- `src/content/docs/reference/constants.md` → `/reference/constants/`

The content is organized into sections defined in `astro.config.mjs:23-73`:
- Getting Started (installation, quickstart, concepts)
- User Guides (data handling, geometric analysis, constants extraction, growth analysis, steady state, visualization)
- Mathematical Models (Chauvet, Moulia, Goriely models)
- API Reference (auto-generated from `/reference/` directory)
- Examples (workflows, notebooks)
- Advanced (batch processing, customization)

### Configuration Files

**astro.config.mjs**: Main configuration
- Markdown processing: remark-math + rehype-katex for LaTeX math rendering (lines 10-14)
- Starlight integration with sidebar structure (lines 17-74)
- TailwindCSS Vite plugin (lines 76-78)

**src/content.config.ts**: Content collections setup
- Uses Starlight's docsLoader and docsSchema for content validation

**package.json**: Dependencies and scripts
- No lint/test scripts defined currently
- Key dependencies: `@astrojs/starlight`, `astro`, `katex`, `rehype-katex`, `remark-math`, `sharp`, `tailwindcss`

### Mathematical Notation

The site extensively uses LaTeX math notation rendered by KaTeX:
- Inline math: `$L_c = \gamma/\beta$`
- Block math: `$$\theta(s) = ...$$`
- Special consideration: all mathematical formulas must be properly escaped for Markdown

## Domain-Specific Context

### Plant Tropism Analysis
The documentation covers:
- **Gravitropism**: Plant growth response to gravity
- **Proprioception**: Plant's ability to sense its own curvature
- **Key parameters**: Convergence length (Lc), gamma (γ), beta (β), Young's modulus (E)

### Coordinate Systems
Two primary coordinate systems are referenced:
1. Cartesian (x, y): Raw pixel coordinates from images
2. Arc-length (s, θ): Natural coordinates along plant centerline

### Data Flow
Typical analysis workflow documented:
1. Load centerline CSV data (frame, x, y coordinates)
2. Transform to arc-length coordinates
3. Calculate angles and curvatures
4. Fit mathematical models
5. Extract physical constants

## Working with Content

When creating or editing documentation:
- Use frontmatter for metadata: `title`, `description`
- Mathematical notation must use KaTeX-compatible LaTeX
- Code examples should use Python with the `constants` module import pattern
- Reference API functions in the format: `from tropism_toolset import function_name`
- Include units in parameter descriptions (m, s⁻¹, m⁻¹, Pa, rad)

## Starlight Features Available

- `<Card>` and `<CardGrid>` components from `@astrojs/starlight/components`
- Hero sections with taglines and CTAs (splash template)
- Collapsible sidebar sections
- Auto-generated API reference (via `autogenerate: { directory: 'reference' }`)
- GitHub social link integration
