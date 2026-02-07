# DECISIONS.md — Journal de Décisions

> Contraintes, choix techniques, commit pins, "do not repeat".

## D001 — Pas de base de données pour le MVP
- **Décision** : Les use cases sont stockés en fichiers TypeScript (`src/data/use-cases/`).
- **Raison** : Pas de coût, pas de latence, déploiement statique pur sur Vercel free tier.
- **Réversible** : Oui, migration vers DB possible ultérieurement.

## D002 — Subagents source
- **Repo** : https://github.com/VoltAgent/awesome-claude-code-subagents
- **Commit** : `eef78c7b5696096a39b0d3db753095d280c46958` (2026-02-06)
- **Agents importés** : architect-reviewer, code-reviewer, qa-expert, build-engineer, nextjs-developer, fullstack-developer, seo-specialist
- **Audit** : Aucune commande dangereuse détectée (grep rm -rf, drop, delete, destroy, force push, --hard = 0 résultats)

## D003 — Design tokens centralisés
- **Décision** : Toutes les couleurs dans `src/app/globals.css` via variables CSS HSL (standard shadcn).
- **DO NOT REPEAT** : Ne jamais coder des couleurs en dur dans les composants.

## D004 — Contenu FR uniquement
- **Décision** : Pas d'i18n, pas de contenu anglais. Tout en français.
- **DO NOT REPEAT** : Ne pas ajouter de traductions ou de switcher de langue.

---
*Créé le 2025-02-07*
