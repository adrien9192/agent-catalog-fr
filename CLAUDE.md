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

## Optimisation des modèles (non-négociable)
Toujours spécifier le modèle adapté lors du lancement de sous-agents (Task tool) :
- **Haiku** (`model: "haiku"`) : recherches, lectures de fichiers, explorations, tâches simples et rapides.
- **Sonnet** (`model: "sonnet"`) : écriture de code standard, modifications modérées, recherches complexes.
- **Opus** (`model: "opus"`) : architecture, raisonnement complexe, rédaction de contenu de haute qualité, stratégie.
- Ne JAMAIS laisser le modèle par défaut (hérité = Opus) pour des tâches simples. Optimiser coût et latence.

## Agent Teams (tmux split-pane mode)
Agent teams est activé (`CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`, `teammateMode: "tmux"`).

### Quand utiliser les agent teams
- Écriture de 5+ use cases ou guides en parallèle (1 teammate par use case).
- Refactoring multi-fichiers indépendants (ex: rollout n8n tutorial sur 60 pages).
- Recherche + implémentation en parallèle (1 researcher + 1 builder).
- Audit multi-aspect (SEO, perf, UX, code quality — chacun son teammate).
- Debug avec hypothèses concurrentes.

### Quand NE PAS utiliser les agent teams
- Tâches séquentielles (édition du même fichier).
- Tâches simples (< 3 étapes).
- Modifications dépendantes les unes des autres.
→ Utiliser des subagents (Task tool) ou le lead seul dans ces cas.

### Organisation par défaut
- **Lead** : coordonne, crée les tasks, synthétise, fait le build final et push.
- **Teammates** : exécutent les tâches assignées. Modèle Sonnet par défaut sauf tâche complexe.
- Lead ne code PAS lui-même quand des teammates sont actifs (mode delegate).
- Chaque teammate possède ses propres fichiers — jamais 2 teammates sur le même fichier.

### Règles d'attribution des modèles pour les teammates
- **Haiku** : recherche web, lecture de fichiers, exploration codebase.
- **Sonnet** : écriture de code, rédaction de contenu, modifications.
- **Opus** : architecture complexe, stratégie, revue critique.

### Tâches typiques pour ce projet
| Tâche | Nb teammates | Modèle |
|---|---|---|
| Écrire 5 use cases | 5 (1 par UC) | Sonnet |
| Écrire 4 guides SEO | 4 (1 par guide) | Sonnet |
| Rollout n8n tutorial | 3-4 (batch de fichiers) | Sonnet |
| Audit global | 3 (SEO, perf, UX) | Haiku/Sonnet |
| Refactoring composants | 2-3 (par scope) | Sonnet |

### Bonnes pratiques
- Toujours donner un prompt détaillé au spawn (types, interfaces, exemples, contraintes).
- 5-6 tasks par teammate maximum pour garder la productivité.
- Le lead attend que TOUS les teammates finissent avant de build/push.
- Le lead fait le cleanup (`Clean up the team`) quand tout est terminé.
- Vérifier qu'aucun teammate n'édite le même fichier qu'un autre.

## Déploiement
- GitHub + Vercel (free tier). Pas de DB sauf nécessité prouvée.
- Ne jamais prétendre avoir créé des comptes.
