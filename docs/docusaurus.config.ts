import { themes as prismThemes } from "prism-react-renderer";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: "IGN LiDAR HD Processing Library",
  tagline: "Transform IGN LiDAR data into ML-ready datasets with v4.0.1",
  favicon: "img/favicon.ico",

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: "https://sducournau.github.io",
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: "/IGN_LIDAR_HD_DATASET/",

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: "sducournau", // Usually your GitHub org/user name.
  projectName: "IGN_LIDAR_HD_DATASET", // Usually your repo name.

  onBrokenLinks: "warn",

  // GitHub Pages deployment config
  trailingSlash: false,
  deploymentBranch: "gh-pages",

  // Markdown configuration
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
    mermaid: true,
  },

  // Internationalization (English only)
  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      {
        docs: {
          routeBasePath: "/", // Serve the docs at the root
          sidebarPath: "./sidebars.ts",
          editUrl:
            "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/website/",
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: {
          routeBasePath: "/blog", // Serve the blog at /blog
          showReadingTime: true,
          feedOptions: {
            type: ["rss", "atom"],
            xslt: true,
          },
          editUrl:
            "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/tree/main/website/",
          // Useful options to enforce blogging best practices
          onInlineTags: "warn",
          onInlineAuthors: "warn",
          onUntruncatedBlogPosts: "warn",
        },
        theme: {
          customCss: "./src/css/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    "@docusaurus/theme-mermaid",
    [
      "@easyops-cn/docusaurus-search-local",
      {
        // Options for local search
        hashed: true,
        language: ["en"],
        indexDocs: true,
        indexBlog: true,
        indexPages: false,
        docsRouteBasePath: "/",
        blogRouteBasePath: "/blog",
        // Search result limits
        searchResultLimits: 8,
        searchResultContextMaxLength: 50,
        // Exclude patterns
        ignoreFiles: [
          /archive\//,
          /legacy\//,
          /migration\//,
          /^docs\/docs\/docs\//,
        ],
      },
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: "img/lidar-visualization-thumbnail.jpg",
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "IGN LiDAR HD",
      logo: {
        alt: "LOD3 Building Model",
        src: "img/lod3.png",
      },
      items: [
        {
          type: "docSidebar",
          sidebarId: "tutorialSidebar",
          position: "left",
          label: "Documentation",
        },
        { to: "/blog", label: "📢 First Release!", position: "left" },
        {
          href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      style: "dark",
      links: [
        {
          title: "Docs",
          items: [
            {
              label: "Getting Started",
              to: "/",
            },
            {
              label: "Installation",
              to: "/installation/quick-start",
            },
            {
              label: "Basic Usage",
              to: "/guides/basic-usage",
            },
          ],
        },
        {
          title: "Community",
          items: [
            {
              label: "GitHub",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
            },
            {
              label: "Issues",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET/issues",
            },
          ],
        },
        {
          title: "More",
          items: [
            {
              label: "Blog",
              to: "/blog",
            },
            {
              label: "GitHub",
              href: "https://github.com/sducournau/IGN_LIDAR_HD_DATASET",
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} IGN LiDAR HD. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ["python", "bash"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
