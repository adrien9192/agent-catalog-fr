# PLAN.md — Plan d'Exécution AgentCatalog FR

## Phase A : Fondations
- [x] Créer CLAUDE.md
- [x] Créer .codex_swarm_memory/ (SPEC, PLAN, DECISIONS, STATUS)
- [x] Importer subagents GitHub → .claude/agents/

## Phase B : Bootstrap Technique
- [x] Init Next.js 16 App Router + TypeScript
- [x] Configurer Tailwind CSS + design tokens (globals.css)
- [x] Installer shadcn/ui (Button, Card, Badge, Input, Sheet, Dialog, Tabs, Separator, ScrollArea)
- [x] Configurer next/font (Inter) + next/image
- [x] Configurer ESLint + Prettier
- [x] Installer Playwright + config initiale
- [x] Installer Framer Motion

## Phase C : Design System + Layout
- [x] Créer layout racine (header, footer, mobile nav)
- [x] Implémenter dotted grid background
- [x] Implémenter hero section (H1 + sous-titre + CTA pills)
- [x] Implémenter dark prompt bar (recherche IA-style)
- [x] Composants réutilisables : UseCaseCard, DifficultyBadge, FilterBar
- [x] Responsive : valider 360 → 1440px, zéro scroll horizontal

## Phase D : Pages + Routing
- [x] Page Accueil (/) : hero + prompt bar + use cases vedettes
- [x] Page Catalogue (/catalogue) : grille + filtres (secteur, métier, difficulté, stack)
- [x] Page Use Case (/use-case/[slug]) : détail complet (5 sections)
- [x] Pages Secteur (/secteur/[slug]) : agrégation par secteur
- [x] Pages Métier (/metier/[slug]) : agrégation par métier

## Phase E : Contenu MVP
- [x] Écrire les 10 use cases structurés (données TypeScript)
- [x] Remplir chaque use case : présentation, stack, tutoriel, metadata
- [x] Valider couverture : 7 fonctions, 3 niveaux

## Phase F : SEO + Meta
- [x] Metadata FR par page (title, description)
- [ ] JSON-LD structured data pour use cases (post-MVP)
- [ ] Sitemap.xml (post-MVP)
- [ ] Open Graph images (post-MVP)

## Phase G : Tests + QA
- [x] Tests Playwright : navigation principale (36/36 pass)
- [x] Tests Playwright : responsive screenshots (360, 390, 768, 1024, 1280, 1440)
- [x] Tests Playwright : zéro scroll horizontal (all breakpoints pass)
- [x] Validation build production (next build — 41 pages, 0 erreurs)
- [ ] Validation Lighthouse (post-deploy)

## Phase H : Déploiement
- [x] Provisioning Pack (scripts + docs)
- [x] Git init + premier commit
- [x] Push GitHub (gh CLI authenticated)
- [ ] Deploy Vercel (Vercel CLI non authentifié — instructions fournies)

---
*Mis à jour : 2025-02-07*
