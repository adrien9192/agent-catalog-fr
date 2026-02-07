# PLAN.md — Plan d'Exécution AgentCatalog FR

## Itération 1 (DONE)
- [x] Bootstrap Next.js + Tailwind + shadcn
- [x] 10 use cases FR, 41 pages statiques
- [x] 36/36 Playwright tests, build OK
- [x] Déployé Vercel + GitHub

## Itération 2 — Search + Newsletter + SEO (DONE)
- [x] Fuse.js search + synonymes FR (40+ synonymes, 9 catégories)
- [x] URL sync filtres ↔ URL (partage de liens filtrés)
- [x] Zero-results suggestions + compteur résultats
- [x] Newsletter Brevo : signup (3 variantes), welcome email, daily cron
- [x] List ID fix (2→3), verified sender, welcome transactionnel
- [x] Daily cron working (campaign #14 sent, 2 delivered)
- [x] JSON-LD HowTo schema
- [x] next-sitemap + robots.txt
- [x] OG images dynamiques (home + use cases)
- [x] Metadata fix (metadataBase, twitter card)
- [x] Header active state + navigation simplifiée
- [x] Modèles mis à jour (Claude Sonnet 4.5, GPT-4.1)
- [x] n8n/Make.com en alternatives low-cost (10 use cases)

## Itération 3 — QMD Améliorations (EN COURS)

### Audit des 22 questions — Matrice de conformité

| # | Question | Statut | Gap |
|---|----------|--------|-----|
| 1 | Storage use cases | ✅ DONE | — |
| 2 | Catalogue SSR/SSG | ✅ DONE | Client + Suspense |
| 3 | Search implementation | ✅ DONE | Fuse.js + synonymes |
| 4 | FR synonymes | ✅ DONE | 40+ synonymes, 9 catégories |
| 5 | Approche search + poids | ✅ DONE | Fuse weighted (title:3 → roi:0.5) |
| 6 | Search UX | ⚠️ PARTIAL | Highlights manquants |
| 7 | Filtres + search interplay | ✅ DONE | AND logic combiné |
| 8 | Single vs double opt-in | ✅ DONE | Single opt-in MVP |
| 9 | Placements newsletter | ✅ DONE | 4 emplacements |
| 10 | Segmentation Brevo | ✅ DONE | SOURCE, SECTORS, FUNCTIONS, OPT_IN_DATE |
| 11 | Daily email round-robin | ✅ DONE | Deterministic day-of-year % |
| 12 | Architecture état | ✅ DONE | Stateless, no DB |
| 13 | Séquence onboarding | ⚠️ PARTIAL | 1 welcome, pas 5-7 emails |
| 14 | Template tutoriel par difficulté | ⚠️ PARTIAL | Structure OK, pas de diff visuelle |
| 15 | Références modèles | ✅ DONE | Claude 4.5, GPT-4.1 |
| 16 | Workflows n8n/Make concrets | ❌ MISSING | Mentionnés en stack, pas de JSON/click-path |
| 17 | Sections enterprise | ⚠️ PARTIAL | RGPD (CV), audit (fraude), pas systématique |
| 18 | SEO (OG/sitemap/schema) | ✅ DONE | Tout en place |
| 19 | Copywriting FR | ✅ DONE | Bon positionnement |
| 20 | Layout (reading width/code) | ✅ DONE | 70ch, dark code blocks |
| 21 | Responsive no h-scroll | ✅ DONE | Testé 360-1440px |
| 22 | Playwright tests complets | ⚠️ PARTIAL | Navigation OK, search/filters/signup manquants |

### 3A. Search highlights (Q6)
- [ ] Highlight termes matchés dans les résultats de recherche
- [ ] Afficher le score de pertinence visuellement

### 3B. Workflows n8n/Make concrets (Q16)
- [ ] Ajouter un workflow n8n JSON exportable pour les 3 use cases Facile
- [ ] Ajouter un parcours Make pas-à-pas pour les 4 use cases Moyen
- [ ] Ajouter un workflow hybrid code+n8n pour les 2 use cases Expert

### 3C. Sections enterprise systématiques (Q17)
- [ ] Ajouter section "Enterprise" à CHAQUE use case (PII, audit, HITL, monitoring)
- [ ] Mettre à jour types.ts si nécessaire

### 3D. Template visuel par difficulté (Q14)
- [ ] Badges visuels différenciés + indicateurs de temps
- [ ] Sections optionnelles selon le niveau

### 3E. Séquence onboarding 5 emails (Q13)
- [ ] Créer 5 templates Brevo transactionnels pour onboarding
- [ ] Implémenter logique de séquence (j+0, j+1, j+2, j+3, j+7)

### 3F. Playwright tests search/filters/signup (Q22)
- [ ] Test search fuzzy (SAV → support, compta → finance)
- [ ] Test filtres combinés (fonction + difficulté)
- [ ] Test inscription newsletter (form submission + response)
- [ ] Test responsive screenshots automatisés

### 3G. Images visuelles sur le site
- [ ] Hero illustrations/icônes pour chaque section
- [ ] Icônes visuelles pour les cards use case
- [ ] Illustrations pour les pages secteur/métier

---
*Mis à jour : 2026-02-07*
