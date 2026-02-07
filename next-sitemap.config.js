/** @type {import('next-sitemap').IConfig} */
module.exports = {
  siteUrl: process.env.NEXT_PUBLIC_SITE_URL || "https://agent-catalog-fr.vercel.app",
  generateRobotsTxt: true,
  changefreq: "weekly",
  priority: 0.7,
  sitemapSize: 5000,
  exclude: ["/api/*"],
  robotsTxtOptions: {
    policies: [
      {
        userAgent: "*",
        allow: "/",
        disallow: ["/api/"],
      },
    ],
    additionalSitemaps: [
      "https://agent-catalog-fr.vercel.app/sitemap.xml",
    ],
  },
  transform: async (config, path) => {
    // High priority: homepage, catalogue, pricing
    if (path === "/" || path === "/catalogue" || path === "/pricing") {
      return { loc: path, changefreq: "daily", priority: 1.0, lastmod: new Date().toISOString() };
    }
    // High priority: individual use cases (main content)
    if (path.startsWith("/use-case/")) {
      return { loc: path, changefreq: "weekly", priority: 0.9, lastmod: new Date().toISOString() };
    }
    // Medium-high: guides and comparisons (SEO content)
    if (path.startsWith("/guide/") || path.startsWith("/comparatif/")) {
      return { loc: path, changefreq: "weekly", priority: 0.8, lastmod: new Date().toISOString() };
    }
    // Medium: index pages, calculator
    if (path === "/guide" || path === "/comparatif" || path === "/calculateur-roi") {
      return { loc: path, changefreq: "weekly", priority: 0.8, lastmod: new Date().toISOString() };
    }
    // Medium: sector and metier pages
    if (path.startsWith("/secteur/") || path.startsWith("/metier/")) {
      return { loc: path, changefreq: "weekly", priority: 0.7, lastmod: new Date().toISOString() };
    }
    // Default
    return { loc: path, changefreq: config.changefreq, priority: 0.5, lastmod: new Date().toISOString() };
  },
};
