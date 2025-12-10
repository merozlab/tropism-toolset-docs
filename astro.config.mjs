// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import tailwindcss from "@tailwindcss/vite";
import vercel from "@astrojs/vercel";

// https://astro.build/config
export default defineConfig({
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  integrations: [
    starlight({
      title: "Tropism Toolset",
      description: "Python toolkit for analyzing plant mechanics and tropisms",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/merozlab/tropism-toolset",
        },
      ],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Installation", slug: "guides/installation" },
            { label: "Quick Start", slug: "guides/quickstart" },
            { label: "Design Principles", slug: "guides/design-principles" },
            { label: "Core Concepts", slug: "guides/concepts" },
          ],
        },
        {
          label: "Mathematical Models",
          collapsed: true,
          items: [
            {
              label: "The AC and ACĖ Models",
              slug: "guides/models/the-ac-and-ace-models",
            },
          ],
        },
        {
          label: "User Guides",
          items: [
            { label: "Preprocessing Workflow", slug: "guides/data-handling" },
            { label: "Beta & Gamma Analysis Workflow", slug: "guides/finding-beta-gamma" },
            { label: "Growth Analysis Workflow", slug: "guides/growth-analysis" },
            { label: "Tip Angle Analysis Workflow", slug: "guides/steady-state" },
          ],
        },
        {
          label: "Core Tools",
          items: [
            { label: "Fitting", slug: "guides/fitting" },
            { label: "Visualization", slug: "guides/visualization" },
          ],
        },
        {
          label: "API Reference",
          collapsed: true,
          autogenerate: { directory: "reference" },
        },
        {
          label: "Examples",
          collapsed: true,
          items: [
            { label: "Complete Workflows", slug: "examples/workflows" },
          ],
        },
        {
          label: "Advanced",
          collapsed: true,
          items: [
            { label: "Batch Processing", slug: "advanced/batch-processing" },
            { label: "Customization", slug: "advanced/customization" },
          ],
        },
      ],
      customCss: [
        "./src/styles/custom.css",
        // you can add more files if you like
      ],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
  output: "static",
  adapter: vercel(),
});
