# Guide de Déploiement — AgentCatalog FR

## GitHub (FAIT)
- Repo : https://github.com/adrien9192/agent-catalog-fr
- Branche : `main`
- Statut : poussé et à jour

## Vercel — Déploiement en 3 étapes

### Option A : Via le dashboard Vercel (recommandé)

1. Aller sur https://vercel.com/new
2. Cliquer "Import Git Repository"
3. Sélectionner `adrien9192/agent-catalog-fr`
4. Vérifier la configuration :
   - **Framework Preset** : Next.js (auto-détecté)
   - **Root Directory** : `.` (par défaut)
   - **Build Command** : `next build` (par défaut)
   - **Output Directory** : `.next` (par défaut)
5. Cliquer "Deploy"
6. URL de production : `https://agent-catalog-fr.vercel.app`

### Option B : Via Vercel CLI

```bash
# 1. Se connecter
npx vercel login

# 2. Déployer (depuis le dossier du projet)
npx vercel --prod

# Répondre aux questions :
# - Set up and deploy? → Y
# - Which scope? → (votre compte)
# - Link to existing project? → N
# - Project name? → agent-catalog-fr
# - Directory? → ./
# - Override settings? → N
```

### Option C : Script automatique

```bash
# Si vous êtes déjà connecté à Vercel CLI :
npx vercel --prod --yes
```

## Variables d'environnement

Aucune variable d'environnement requise pour le MVP.
Le site est entièrement statique (pas de DB, pas d'API keys).

## Domaine personnalisé (optionnel)

```bash
# Via CLI
npx vercel domains add mondomaine.com

# Ou via le dashboard Vercel :
# Settings → Domains → Add Domain
```

## Coût

- **Vercel Free Tier** : 100 GB bandwidth/mois, déploiements illimités
- **GitHub Free** : repos publics illimités
- **Total : 0€/mois**
