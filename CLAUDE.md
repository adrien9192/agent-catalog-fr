# CLAUDE.md — Règles Projet "AgentCatalog FR"

## Identité
- Site FR cataloguant des cas d'usage d'Agents IA implantables en entreprise.
- Langue : français uniquement (UI, contenu, metadata, alt texts).

## Stack (non-négociable)
- Next.js 14+ (App Router) + TypeScript strict
- Tailwind CSS + shadcn/ui (Radix UI)
- next/font (Inter) + next/image
- ESLint + Prettier
- Playwright (tests responsive + screenshots)
- Framer Motion (micro-animations uniquement, optionnel)

## Design Tokens
- Toutes les couleurs via variables CSS shadcn + Tailwind (`globals.css`).
- Changer la charte = modifier UN fichier (`globals.css`) sans casser le responsive.
- Style : "Modern SaaS 2025" — dotted grid bg, hero fort, CTA pills, dark prompt bar.

## Mobile-First (non-négociable)
- CSS mobile-first, breakpoints: 360, 390, 768, 1024, 1280, 1440.
- Zéro scroll horizontal. Texte base >= 16px mobile. Largeur lecture 65-75ch desktop.
- Images/embeds jamais en overflow.

## Architecture Contenu
- Chaque use case : présentation, stack recommandée, tutoriel, difficulté, tags.
- Données use cases en fichiers TypeScript (pas de DB pour le MVP).
- Routing : /catalogue, /use-case/[slug], /secteur/[slug], /metier/[slug], /blog (optionnel).

## Conventions
- Noms de fichiers : kebab-case.
- Composants : PascalCase.
- Pas de `any` TypeScript. Interfaces explicites.
- Commits conventionnels (feat:, fix:, docs:, chore:).

## Workflow
- ARCHITECT → BUILDER → REVIEWER (boucle).
- Max 3 tentatives par blocage avant escalade.
- Mémoire projet : `.codex_swarm_memory/` (SPEC, PLAN, DECISIONS, STATUS).

## Déploiement
- GitHub + Vercel (free tier). Pas de DB sauf nécessité prouvée.
- Ne jamais prétendre avoir créé des comptes.
