# PLAN.md — Plan d'Exécution AgentCatalog FR

## Itération 1 (DONE)
- [x] Bootstrap Next.js + Tailwind + shadcn
- [x] 10 use cases FR, 41 pages statiques
- [x] 36/36 Playwright tests, build OK
- [x] Déployé Vercel + GitHub

## Itération 2 — Search + Newsletter + SEO (DONE)
- [x] Fuse.js search + synonymes FR (40+ synonymes, 9 catégories)
- [x] URL sync filtres - URL (partage de liens filtrés)
- [x] Zero-results suggestions + compteur résultats
- [x] Newsletter Brevo : signup (3 variantes), welcome email, daily cron
- [x] JSON-LD HowTo schema
- [x] next-sitemap + robots.txt
- [x] OG images dynamiques (home + use cases)

## Itération 3 — QMD + Copywriting + Content (DONE)
- [x] Search fix (per-word Fuse.js + synonym merge)
- [x] URL state sync fix (useEffect for external navigation)
- [x] Custom request form (/demande) + API + admin/user emails
- [x] SaaS copywriting rewrite (hero, CTAs, pricing, footer)
- [x] Pricing page (3 tiers: Gratuit/Pro 29EUR/Equipe 99EUR)
- [x] 10 new use cases (20 total)
- [x] Daily generation script (scripts/)
- [x] Enterprise sections on all use cases
- [x] n8n workflows on all use cases

## Itération 4 — Conversion Optimization (DONE)
- [x] Global audit (SEO 74/100, conversion, technical)
- [x] Exit-intent popup (homepage + catalogue)
- [x] Social proof trust signals (RGPD, open-source, daily updates, FR companies)
- [x] Sticky CTA bar on use case pages
- [x] Mid-page "need help?" CTA on use case pages
- [x] Sidebar CTA card on use case pages
- [x] FAQ JSON-LD schema on pricing page
- [x] Organization JSON-LD schema in layout
- [x] BreadcrumbList JSON-LD schema on use case pages
- [x] Metadata on catalogue and demande pages
- [x] Pricing comparison table
- [x] Guarantee badges on pricing
- [x] Newsletter between catalogue results (after 8 cards)
- [x] Custom request CTA + newsletter on sector/metier pages
- [x] "Recently added" section on homepage
- [x] Footer improvements (RGPD, contact, intermediate difficulty)

## Itération 5 — Next Steps (PLANNED)
- [ ] SEO content pages (blog-style "guides") for long-tail keywords
- [ ] Email welcome sequence (5 emails over 7 days via Brevo automation)
- [ ] A/B test framework for CTA copy variants
- [ ] Analytics integration (Plausible or Umami — RGPD-friendly)
- [ ] Performance optimization (Lighthouse 95+)
- [ ] More use cases (target: 50+ by end of month)
- [ ] User testimonials/case studies (when available)

---
*Mis à jour : 2026-02-07*
