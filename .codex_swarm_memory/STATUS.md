# STATUS.md — État Courant

## État actuel
- **Phase** : Itération 4 — Conversion Optimization (Cold Start)
- **Use cases** : 20 workflows documentés
- **Site live** : https://agent-catalog-fr.vercel.app
- **GitHub** : https://github.com/adrien9192/agent-catalog-fr
- **Objectif** : Maximiser visitor → prospect → free → paid

## Résumé Itérations

| Itération | Statut | Livraisons clés |
|-----------|--------|----------------|
| 1. MVP | DONE | 10 use cases, 41 pages, Playwright 36/36, Vercel |
| 2. Search+Newsletter+SEO | DONE | Fuse.js, Brevo, JSON-LD, OG images, sitemap |
| 3. QMD + Copywriting + Content | DONE | Search fix, pricing, copywriting rewrite, custom request form, 10 new use cases, daily script |
| 4. Conversion Optimization | EN COURS | Social proof, exit-intent, CTAs, SEO schemas, pricing UX |

## Audit Global (2026-02-07)

### SEO Score: 74/100
- Missing: metadata on catalogue/demande pages
- Missing: OG images for sector/metier/pricing pages
- Missing: FAQ schema on pricing, Organization schema, BreadcrumbList schema
- Good: sitemap, robots.txt, JSON-LD HowTo, internal linking, H1/H2 hierarchy

### Conversion Score: Estimated 2-5% (needs 8%+ target)
- Missing: social proof (0 testimonials, 0 logos, 0 case studies)
- Missing: exit-intent popup for newsletter capture
- Missing: strong CTAs on use case pages (no email gate, no upgrade path)
- Missing: trust signals (no legal links, no compliance badges)
- Good: search/filter UX, custom request form, pricing structure

### Technical Score: Good
- No console.log in production (only console.error in API routes - correct)
- TypeScript strict mode passing
- No security issues detected
- Brevo API key properly used via env vars

## Conversion Funnel Plan

| Stage | Current | Target | Actions |
|-------|---------|--------|---------|
| Visitor → Newsletter | ~2% | 8% | Exit-intent popup, better placement, lead magnet copy |
| Newsletter → Catalogue | ~30% | 60% | Personalized daily emails, segmentation |
| Catalogue → Use Case | ~50% | 80% | Already good, add personalization |
| Use Case → Lead | ~0% | 20% | Sticky CTA, inline newsletter, "Get this workflow" |
| Lead → Pricing | TBD | 30% | Targeted emails, clear upgrade path |
| Pricing → Paid | TBD | 5% | Guarantee, annual toggle, comparison table |

---
*Mis à jour : 2026-02-07*
