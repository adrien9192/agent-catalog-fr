import Link from "next/link";
import type { Metadata } from "next";
import { useCases } from "@/data/use-cases";
import { guides } from "@/data/guides";
import { sectors } from "@/data/sectors";
import { metiers } from "@/data/metiers";
import { comparisons } from "@/data/comparisons";

export const metadata: Metadata = {
  title: "Plan du site — AgentCatalog",
  description:
    "Plan du site complet d'AgentCatalog. Accédez à tous nos workflows IA, guides, comparatifs et pages par secteur et fonction.",
};

export default function PlanDuSitePage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <h1 className="text-3xl font-bold mb-10">Plan du site</h1>

      <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-3">
        {/* Pages principales */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Pages principales</h2>
          <ul className="space-y-1.5 text-sm">
            <li><Link href="/" className="text-muted-foreground hover:text-foreground transition-colors">Accueil</Link></li>
            <li><Link href="/catalogue" className="text-muted-foreground hover:text-foreground transition-colors">Catalogue des workflows</Link></li>
            <li><Link href="/guide" className="text-muted-foreground hover:text-foreground transition-colors">Guides pratiques</Link></li>
            <li><Link href="/comparatif" className="text-muted-foreground hover:text-foreground transition-colors">Comparatifs</Link></li>
            <li><Link href="/calculateur-roi" className="text-muted-foreground hover:text-foreground transition-colors">Calculateur ROI</Link></li>
            <li><Link href="/pricing" className="text-muted-foreground hover:text-foreground transition-colors">Tarifs</Link></li>
            <li><Link href="/demande" className="text-muted-foreground hover:text-foreground transition-colors">Demande sur mesure</Link></li>
          </ul>
        </div>

        {/* Comparatifs */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Comparatifs ({comparisons.length})</h2>
          <ul className="space-y-1.5 text-sm">
            {comparisons.map((c) => (
              <li key={c.slug}>
                <Link href={`/comparatif/${c.slug}`} className="text-muted-foreground hover:text-foreground transition-colors">
                  {c.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Secteurs */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Par secteur ({sectors.length})</h2>
          <ul className="space-y-1.5 text-sm">
            {sectors.map((s) => (
              <li key={s.slug}>
                <Link href={`/secteur/${s.slug}`} className="text-muted-foreground hover:text-foreground transition-colors">
                  {s.icon} {s.name}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Métiers */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Par métier ({metiers.length})</h2>
          <ul className="space-y-1.5 text-sm">
            {metiers.map((m) => (
              <li key={m.slug}>
                <Link href={`/metier/${m.slug}`} className="text-muted-foreground hover:text-foreground transition-colors">
                  {m.icon} {m.name}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Guides */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Guides ({guides.length})</h2>
          <ul className="space-y-1.5 text-sm">
            {guides.map((g) => (
              <li key={g.slug}>
                <Link href={`/guide/${g.slug}`} className="text-muted-foreground hover:text-foreground transition-colors">
                  {g.title}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        {/* Workflows (first 20 + link to all) */}
        <div>
          <h2 className="text-lg font-semibold mb-3">Workflows ({useCases.length})</h2>
          <ul className="space-y-1.5 text-sm">
            {useCases.slice(0, 20).map((uc) => (
              <li key={uc.slug}>
                <Link href={`/use-case/${uc.slug}`} className="text-muted-foreground hover:text-foreground transition-colors">
                  {uc.title}
                </Link>
              </li>
            ))}
            {useCases.length > 20 && (
              <li>
                <Link href="/catalogue" className="text-primary font-medium hover:underline">
                  ... et {useCases.length - 20} autres workflows &rarr;
                </Link>
              </li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
}
