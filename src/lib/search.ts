import Fuse from "fuse.js";
import type { UseCase } from "@/data/types";

// Table de synonymes FR → terme canonique indexé
const FR_SYNONYMS: Record<string, string[]> = {
  support: ["SAV", "service client", "helpdesk", "assistance", "ticket", "réclamation"],
  sales: ["vente", "commercial", "prospection", "relance", "CRM", "lead", "prospect"],
  rh: ["ressources humaines", "recrutement", "embauche", "onboarding", "CV", "candidat", "talent"],
  marketing: ["contenu", "SEO", "rédaction", "campagne", "emailing", "réseaux sociaux", "brand"],
  finance: ["comptabilité", "compta", "facture", "rapport financier", "budget", "trésorerie", "audit"],
  it: ["DSI", "informatique", "incident", "DevOps", "monitoring", "infrastructure", "système"],
  "supply chain": ["achats", "approvisionnement", "fournisseur", "logistique", "stock", "commande"],
  fraude: ["anti-fraude", "détection", "conformité", "KYC", "LCB-FT", "risque"],
  veille: ["concurrentiel", "benchmark", "intelligence économique", "surveillance", "marché"],
};

/** Expand a query with synonyms: if user types "SAV", also search "support" */
export function expandQuery(query: string): string {
  const q = query.toLowerCase().trim();
  const parts = [q];

  for (const [canonical, synonyms] of Object.entries(FR_SYNONYMS)) {
    for (const syn of synonyms) {
      if (q.includes(syn.toLowerCase())) {
        parts.push(canonical);
        break;
      }
    }
    if (q.includes(canonical)) {
      parts.push(...synonyms.map((s) => s.toLowerCase()));
    }
  }

  return [...new Set(parts)].join(" ");
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
    useExtendedSearch: false,
    findAllMatches: true,
  });
}
