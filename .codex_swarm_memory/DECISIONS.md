# DECISIONS.md — Journal de Décisions

> Contraintes, choix techniques, commit pins, "do not repeat".

## D001 — Pas de base de données pour le MVP
- **Décision** : Les use cases sont stockés en fichiers TypeScript (`src/data/use-cases.ts`).
- **Raison** : Pas de coût, pas de latence, déploiement statique pur sur Vercel free tier.
- **Réversible** : Oui, migration vers DB possible ultérieurement.

## D002 — Subagents source
- **Repo** : https://github.com/VoltAgent/awesome-claude-code-subagents
- **Commit** : `eef78c7b5696096a39b0d3db753095d280c46958` (2026-02-06)
- **Agents importés** : architect-reviewer, code-reviewer, qa-expert, build-engineer, nextjs-developer, fullstack-developer, seo-specialist
- **Audit** : Aucune commande dangereuse détectée

## D003 — Design tokens centralisés
- **Décision** : Toutes les couleurs dans `src/app/globals.css` via variables CSS OKLCH.
- **DO NOT REPEAT** : Ne jamais coder des couleurs en dur dans les composants.

## D004 — Contenu FR uniquement
- **Décision** : Pas d'i18n, pas de contenu anglais. Tout en français.
- **DO NOT REPEAT** : Ne pas ajouter de traductions ou de switcher de langue.

---

## D005 — Moteur de recherche FR (Itération 2)

### Options évaluées
| Option | Coût | Complexité | Robustesse | Qualité UX | SEO |
|--------|------|-----------|------------|-----------|-----|
| 1. `.includes()` actuel | 0 | trivial | faible (pas de fuzzy, pas de synonymes) | 2/5 | 0 |
| 2. **Fuse.js weighted fuzzy** | 0 | faible | bonne (typos, accents, poids par champ) | 4/5 | 0 |
| 3. MiniSearch (inverted index) | 0 | moyen | très bonne (tokenisation, stemming) | 4/5 | 0 |
| 4. Algolia DocSearch | gratuit (OSS) | moyen | excellente | 5/5 | 0 |
| 5. Embeddings sémantiques (OpenAI) | ~$0.01/req | élevé | excellente | 5/5 | 0 |

### Choix : **Option 2 — Fuse.js** + table de synonymes FR
- **Raisons** : zéro coût, ~5KB bundle, fuzzy matching natif, poids configurables par champ, pas de backend.
- **Enrichissement** : table de synonymes FR manuels (SAV→Support, DSI→IT, compta→Finance, etc.)
- **Rejetés** :
  - MiniSearch : plus lourd, stemming FR pas natif, surdimensionné pour 10-50 use cases
  - Algolia : dependency externe, compte à configurer, surdimensionné MVP
  - Embeddings : coût API, latence, infrastructure serveur requise

## D006 — Newsletter Brevo (Itération 2)

### Architecture état daily "use case du jour"
| Option | Coût | Complexité | Fiabilité | Maintenance |
|--------|------|-----------|-----------|-------------|
| 1. **Brevo contact attributes** (last_sent_index) | 0 | faible | bonne | faible |
| 2. Vercel KV (Redis) | gratuit (300 req/j) | moyen | bonne | moyen |
| 3. Supabase (Postgres) | gratuit | moyen | excellente | moyen |
| 4. Fichier JSON dans le repo | 0 | trivial | fragile | élevé |
| 5. Vercel Cron + Brevo campaign API | 0 | moyen | bonne | faible |

### Choix : **Option 5 — Vercel Cron + Brevo Campaigns API**
- Un cron Vercel (`/api/cron/daily-email`) tourne à 7h chaque jour.
- Calcule l'index du jour (`dayOfYear % totalUseCases`) → round-robin sans doublon.
- Envoie un email transactionnel via Brevo API à la liste "newsletter".
- Pas de DB, pas de KV. L'état est déterministe (date → index).
- **Rejetés** :
  - Brevo attrs : polling complexe, limité à 200 attrs
  - Vercel KV/Supabase : infrastructure supplémentaire inutile pour du round-robin
  - Fichier JSON : nécessite un commit+deploy pour chaque changement d'état

### Opt-in : Single opt-in pour le MVP
- Raison : plus simple, conversion plus élevée, suffisant pour un site B2B FR.
- Double opt-in envisageable en V2 si volume > 1000 abonnés.

## D007 — Tutoriels n8n/Make (Itération 2)
- **Décision** : Chaque tutoriel inclut un workflow n8n concret (JSON exportable si applicable) ou un parcours Make pas-à-pas.
- **Modèles IA mis à jour** : Claude Sonnet 4.5, GPT-4.1, Mistral Large 2 (références 2025-2026).
- **Template standardisé** : Prérequis → Configuration → Workflow nodes → Tests → Monitoring.

## D008 — SEO Améliorations (Itération 2)
- **JSON-LD** : `HowTo` schema pour chaque use case (indexation Google).
- **Sitemap** : `next-sitemap` auto-généré au build.
- **OG Images** : Template dynamique via `next/og` (ImageResponse API).
- **Internal linking** : "Cas d'usage similaires" en bas de chaque page use case.

---
*Mis à jour : 2026-02-07*
