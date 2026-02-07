# PLAN.md — Plan d'Exécution AgentCatalog FR — Itération 2

## Itération 1 (DONE)
- [x] Bootstrap Next.js + Tailwind + shadcn
- [x] 10 use cases FR, 41 pages statiques
- [x] 36/36 Playwright tests, build OK
- [x] Déployé Vercel + GitHub

## Itération 2 — Search + Newsletter + Tutorials + SEO

### 2A. Recherche FR intelligente
- [ ] Installer Fuse.js
- [ ] Créer table de synonymes FR (SAV/support, DSI/IT, compta/finance, etc.)
- [ ] Implémenter index Fuse.js avec poids par champ
- [ ] Améliorer UX : highlight résultats, compteur, zero-results avec suggestions
- [ ] Synchroniser filtres ↔ URL (partage de liens filtrés)

### 2B. Newsletter Brevo
- [ ] Créer API route `/api/newsletter/subscribe` (POST)
- [ ] Implémenter upsert contact Brevo + gestion consentement
- [ ] Créer composant NewsletterSignup (hero + footer + pages use case)
- [ ] Créer API route `/api/cron/daily-email` (Vercel Cron)
- [ ] Implémenter logique round-robin "use case du jour"
- [ ] Configurer vercel.json avec cron schedule

### 2C. Tutoriels mis à jour
- [ ] Mettre à jour références modèles (Claude Sonnet 4.5, GPT-4.1)
- [ ] Ajouter workflows n8n concrets pour chaque use case
- [ ] Ajouter sections enterprise : PII, audit logs, monitoring
- [ ] Standardiser template tutoriel par niveau de difficulté

### 2D. SEO
- [ ] Ajouter JSON-LD HowTo schema sur pages use case
- [ ] Installer + configurer next-sitemap
- [ ] Ajouter OG image dynamique (next/og)
- [ ] Ajouter section "Cas d'usage similaires" (internal linking)

### 2E. Tests + QA
- [ ] Tests Playwright : recherche fuzzy
- [ ] Tests Playwright : inscription newsletter
- [ ] Validation build production
- [ ] Push + redeploy Vercel

---
*Mis à jour : 2026-02-07*
