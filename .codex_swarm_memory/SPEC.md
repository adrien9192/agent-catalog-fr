# SPEC.md — Spécification AgentCatalog FR

> Append-only. Ne jamais supprimer d'entrées.

## 1. Vision Produit
Site web français recensant des cas d'usage métiers d'Agents IA implantables en entreprise.
Cible : décideurs, tech leads, consultants IA francophones.

## 2. Modèle de Données — Use Case

```typescript
interface UseCase {
  slug: string;
  title: string;
  subtitle: string;
  // Présentation
  problem: string;
  value: string;
  inputs: string[];
  outputs: string[];
  risks: string[];
  roiIndicatif: string;
  // Stack
  recommendedStack: StackItem[];
  lowCostAlternatives: StackItem[];
  architectureDiagram: string; // ASCII ou SVG inline
  // Tutoriel
  tutorial: TutorialSection[];
  // Metadata
  difficulty: 'Facile' | 'Moyen' | 'Expert';
  sector: string[];       // ex: "Banque", "Retail", "Santé"
  metier: string[];        // ex: "Support Client", "RH"
  functions: string[];     // ex: "Support", "RH", "Finance", "IT", "Sales"
  // SEO
  metaTitle: string;
  metaDescription: string;
  // Dates
  createdAt: string;
  updatedAt: string;
}

interface StackItem {
  name: string;
  category: 'LLM' | 'Orchestration' | 'Database' | 'Hosting' | 'Monitoring' | 'Other';
  url?: string;
  isFree?: boolean;
}

interface TutorialSection {
  title: string;
  content: string;       // Markdown
  codeSnippets?: CodeSnippet[];
}

interface CodeSnippet {
  language: string;
  code: string;
  filename?: string;
}
```

## 3. Navigation / Routes

| Route | Description |
|-------|-------------|
| `/` | Accueil : hero + prompt bar + CTA + use cases vedettes |
| `/catalogue` | Catalogue filtrable (secteur, métier, difficulté, stack) |
| `/use-case/[slug]` | Détail complet d'un use case |
| `/secteur/[slug]` | Page secteur regroupant les use cases |
| `/metier/[slug]` | Page métier regroupant les use cases |
| `/blog` | Blog (optionnel MVP) |

## 4. MVP — 10 Use Cases

| # | Titre | Fonction | Difficulté |
|---|-------|----------|-----------|
| 1 | Agent de Triage Support Client | Support | Facile |
| 2 | Agent de Qualification de Leads | Sales | Moyen |
| 3 | Agent d'Analyse de CVs et Pré-sélection | RH | Moyen |
| 4 | Agent de Veille Concurrentielle | Marketing | Moyen |
| 5 | Agent de Génération de Rapports Financiers | Finance | Expert |
| 6 | Agent de Gestion des Incidents IT | IT | Moyen |
| 7 | Agent de Rédaction de Contenu Marketing | Marketing | Facile |
| 8 | Agent d'Onboarding Collaborateurs | RH | Facile |
| 9 | Agent de Détection de Fraude | Finance | Expert |
| 10 | Agent d'Automatisation des Achats | Supply Chain | Moyen |

Fonctions couvertes : Support, Sales, RH, Marketing, Finance, IT, Supply Chain (7 fonctions > 5 minimum).
Niveaux : 3 Facile, 4 Moyen, 2 Expert ✓

## 5. SEO
- Metadata FR sur chaque page (title, description, og:image).
- Structured data JSON-LD pour les use cases.
- Sitemap.xml auto-généré.
- Canonical URLs.

## 6. Performance
- Score Lighthouse cible : 90+ sur toutes les métriques.
- Images optimisées via next/image.
- Fonts optimisées via next/font.

---
*Créé le 2025-02-07*
