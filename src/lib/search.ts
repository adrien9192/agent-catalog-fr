import Fuse from "fuse.js";
import type { UseCase } from "@/data/types";

// Table de synonymes FR → terme canonique indexé
const FR_SYNONYMS: Record<string, string[]> = {
  support: ["SAV", "service client", "helpdesk", "assistance", "ticket", "réclamation", "FAQ", "self-service"],
  sales: ["vente", "commercial", "prospection", "relance", "CRM", "lead", "prospect", "qualification", "rendez-vous", "appel"],
  rh: ["ressources humaines", "recrutement", "embauche", "onboarding", "CV", "candidat", "talent", "engagement", "collaborateur"],
  marketing: ["contenu", "SEO", "rédaction", "campagne", "emailing", "réseaux sociaux", "brand", "avis", "traduction", "localisation"],
  finance: ["comptabilité", "compta", "facture", "rapport financier", "budget", "trésorerie", "audit", "notes de frais", "crédit", "scoring", "risque crédit"],
  it: ["DSI", "informatique", "incident", "DevOps", "monitoring", "infrastructure", "système", "maintenance", "panne", "prédictive"],
  "supply chain": ["achats", "approvisionnement", "fournisseur", "logistique", "stock", "commande"],
  fraude: ["anti-fraude", "détection", "conformité", "KYC", "LCB-FT", "risque"],
  veille: ["concurrentiel", "benchmark", "intelligence économique", "surveillance", "marché"],
  email: ["courrier", "mail", "courriel", "messagerie", "tri", "automatisation email"],
  rapport: ["reporting", "dashboard", "tableau de bord", "KPI", "indicateur"],
};

/** Get synonym-expanded search terms as an array of individual terms */
export function getSearchTerms(query: string): string[] {
  const q = query.toLowerCase().trim();
  if (!q) return [];

  const terms = new Set<string>();
  // Add original words
  q.split(/\s+/).forEach((w) => terms.add(w));

  for (const [canonical, synonyms] of Object.entries(FR_SYNONYMS)) {
    // If query matches a synonym, add the canonical term
    for (const syn of synonyms) {
      if (q.includes(syn.toLowerCase())) {
        terms.add(canonical);
        break;
      }
    }
    // If query matches canonical, add synonyms as individual words
    if (q.includes(canonical)) {
      for (const syn of synonyms) {
        syn.toLowerCase().split(/\s+/).forEach((w) => terms.add(w));
      }
    }
  }

  return [...terms];
}

/** Backward compat: return expanded query as string (used for highlights) */
export function expandQuery(query: string): string {
  return query.trim();
}

/** Build a Fuse index with weighted fields */
export function createSearchIndex(useCases: UseCase[]): Fuse<UseCase> {
  return new Fuse(useCases, {
    keys: [
      { name: "title", weight: 3 },
      { name: "subtitle", weight: 2 },
      { name: "problem", weight: 1.5 },
      { name: "value", weight: 1.5 },
      { name: "functions", weight: 2 },
      { name: "sectors", weight: 1 },
      { name: "metiers", weight: 1 },
      { name: "roiIndicatif", weight: 0.5 },
    ],
    threshold: 0.4,
    distance: 200,
    includeScore: true,
    includeMatches: true,
    ignoreLocation: true,
    minMatchCharLength: 2,
    findAllMatches: true,
  });
}

/** Search use cases with per-word matching and synonym expansion */
export function searchUseCases(
  fuseIndex: Fuse<UseCase>,
  query: string
): UseCase[] {
  const q = query.trim();
  if (!q) return [];

  const terms = getSearchTerms(q);
  if (terms.length === 0) return [];

  // Search each term individually and collect scored results
  const scoreMap = new Map<string, { item: UseCase; score: number; hits: number }>();

  for (const term of terms) {
    const results = fuseIndex.search(term);
    for (const r of results) {
      const existing = scoreMap.get(r.item.slug);
      if (existing) {
        // Better (lower) score wins, count hits for ranking
        existing.score = Math.min(existing.score, r.score ?? 1);
        existing.hits += 1;
      } else {
        scoreMap.set(r.item.slug, {
          item: r.item,
          score: r.score ?? 1,
          hits: 1,
        });
      }
    }
  }

  // Sort by hits (desc) then score (asc = better match)
  return [...scoreMap.values()]
    .sort((a, b) => b.hits - a.hits || a.score - b.score)
    .map((r) => r.item);
}
