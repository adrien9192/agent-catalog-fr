# AUDIT.md — Audit Global (2026-02-07)

## SEO Audit

### Present (Good)
- Global metadata with title template + description
- metadataBase set to production URL
- HTML lang="fr", OpenGraph locale fr_FR
- JSON-LD HowTo schema on all use case pages
- Sitemap auto-generated (52 URLs)
- Robots.txt blocking /api/
- OG images for home + all 20 use cases
- Internal linking excellent (breadcrumbs, related use cases, footer links)
- H1/H2 hierarchy correct on most pages

### Missing (Fix)
1. **Catalogue page** — NO metadata export (critical page)
2. **Demande page** — NO metadata export
3. **Pricing page FAQ** — No FAQPage schema
4. **Organization schema** — Not present
5. **BreadcrumbList schema** — Present in UI but not in structured data
6. **OG images** missing for sector/metier/pricing/demande pages
7. **Twitter creator/site** tags not set
8. **Canonical URL** on catalogue (filter params create duplicates)

## Conversion/UX Audit

### Homepage
- Good: clear value prop, multi-CTA strategy, 3-step process
- Fix: update stats (now 20+ not 10+), add testimonials/social proof, add urgency

### Catalogue
- Good: search with synonyms, filters, zero-state with custom request CTA
- Fix: add newsletter between results, improve filter UX on mobile

### Use Case Pages
- Good: comprehensive content, enterprise section, n8n workflows
- Fix: NO lead capture CTAs, no sticky CTA bar, newsletter only at very bottom

### Pricing
- Good: 3 tiers, FAQ section, clear structure
- Fix: no guarantee badge, no annual toggle, no comparison table, no urgency

### Newsletter
- Good: hero, inline, footer variants, welcome email works
- Fix: weak copy ("Recevoir mon workflow"), no incentive, appears too late on pages

### Header
- Good: sticky, responsive, mobile menu
- Fix: CTA button redundant with nav link, no search in header

### Footer
- Good: organized links, newsletter
- Fix: no legal links, no social links, no contact info, weak resources section

## Conversion Research Key Findings

### Cold-Start SaaS Best Practices 2026
1. Lead magnets: free templates, checklists, ROI calculators
2. Social proof when zero customers: "Built for X companies", methodology credibility
3. Exit-intent captures 10-15% of abandoning visitors
4. Email nurture: 5-email welcome sequence converts 3x better than single welcome
5. Trust signals: RGPD compliance badge, "Réponse sous 48h" guarantees
6. Pricing: annual discount toggle increases ARPU 15-20%
7. FAQ schema on pricing improves CTR by 30% in search results

### Priority Implementation (Impact * Effort)
1. Exit-intent popup (HIGH impact, LOW effort)
2. Social proof section on homepage (HIGH impact, LOW effort)
3. Sticky CTA on use case pages (HIGH impact, LOW effort)
4. Metadata on catalogue/demande (HIGH impact, LOW effort)
5. FAQ schema on pricing (MED impact, LOW effort)
6. Annual billing toggle (MED impact, MED effort)
7. Comparison table on pricing (MED impact, MED effort)

---
*Créé : 2026-02-07*
