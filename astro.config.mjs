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
            { label: "Working with Data", slug: "guides/data-handling" },
            { label: "Finding β̃ and γ̃", slug: "guides/finding-beta-gamma" },
            { label: "Fitting", slug: "guides/fitting" },
            { label: "Growth Analysis", slug: "guides/growth-analysis" },
            { label: "Steady State Detection", slug: "guides/steady-state" },
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
            { label: "Notebook Guides", slug: "examples/notebooks" },
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
