Tu es un expert en Agents IA et automatisation d'entreprise. Ta mission est d'ajouter 3 à 5 nouveaux cas d'usage au fichier `src/data/use-cases.ts` du projet AgentCatalog.

## Étape 1 : Recherche

Recherche sur le web les cas d'usage d'Agents IA les plus tendance et demandés en 2026. Concentre-toi sur :
- Les workflows IA que les entreprises recherchent activement
- Les niches sous-représentées dans notre catalogue
- Les innovations récentes en matière d'agents IA

## Étape 2 : Vérification des doublons

Lis le fichier `src/data/use-cases.ts` pour voir les cas d'usage existants. NE PAS ajouter de doublons. Vérifie les slugs existants.

## Étape 3 : Rédaction

Pour chaque nouveau cas d'usage, respecte EXACTEMENT cette structure TypeScript (tous les champs sont obligatoires) :

```typescript
{
    slug: "agent-nom-du-cas",
    title: "Titre en Français",
    subtitle: "Sous-titre descriptif",
    problem: "Description du problème résolu (2-3 phrases)",
    value: "Proposition de valeur (2-3 phrases)",
    inputs: ["Input 1", "Input 2", "Input 3", "Input 4"],
    outputs: ["Output 1", "Output 2", "Output 3", "Output 4", "Output 5"],
    risks: ["Risque 1", "Risque 2", "Risque 3"],
    roiIndicatif: "ROI avec chiffres spécifiques",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Database", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `Diagramme ASCII montrant le flux`,
    tutorial: [
      {
        title: "Titre de l'étape",
        content: "Description de l'étape",
        codeSnippets: [
          {
            language: "python",
            code: `code Python fonctionnel`,
            filename: "fichier.py",
          },
        ],
      },
      // 3-4 étapes au total
    ],
    enterprise: {
      piiHandling: "Gestion PII/RGPD",
      auditLog: "Ce qui est tracé",
      humanInTheLoop: "Quand la validation humaine est requise",
      monitoring: "Métriques clés à suivre",
    },
    n8nWorkflow: {
      description: "Description du workflow n8n",
      nodes: ["Node 1", "Node 2", "Node 3"],
      triggerType: "Type de trigger",
    },
    estimatedTime: "X-Yh",
    difficulty: "Facile" ou "Moyen" ou "Expert",
    sectors: ["Secteur1", "Secteur2"],
    metiers: ["Métier1", "Métier2"],
    functions: ["Fonction"],
    metaTitle: "Titre SEO optimisé",
    metaDescription: "Description SEO (max 160 caractères)",
    createdAt: "YYYY-MM-DD", // date du jour
    updatedAt: "YYYY-MM-DD", // date du jour
  },
```

## Règles impératives

1. **Tout en français** sauf les noms de technologies
2. **Code Python fonctionnel** dans les tutoriels avec `claude-sonnet-4-5-20250514` comme modèle
3. **Pas de doublons** avec les cas existants
4. **ROI chiffré** et réaliste
5. **3-4 étapes de tutoriel** avec code pour chaque
6. **Diagramme ASCII** cohérent avec le flux
7. **SEO** : metaTitle et metaDescription optimisés pour le référencement français
8. **Secteurs** parmi : Banque, Assurance, E-commerce, B2B SaaS, Services, Industrie, Retail, Santé, Telecom, Média, Audit, Distribution, Tous secteurs
9. **Fonctions** parmi : Support, Sales, RH, Marketing, Finance, IT, Supply Chain, Legal, Operations, Product

## Étape 4 : Insertion

Édite le fichier `src/data/use-cases.ts` pour ajouter les nouveaux cas d'usage AVANT le `];` final.

## Étape 5 : Validation

Lance `npx tsc --noEmit` pour vérifier qu'il n'y a pas d'erreurs TypeScript.

## Étape 6 : Build et Deploy

```bash
npm run build
git add src/data/use-cases.ts public/sitemap-0.xml
git commit -m "feat: add X new use cases (YYYY-MM-DD)"
git push origin main
```
