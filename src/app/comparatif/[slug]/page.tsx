import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { comparisons } from "@/data/comparisons";
import { useCases } from "@/data/use-cases";

export function generateStaticParams() {
  return comparisons.map((c) => ({ slug: c.slug }));
}

export function generateMetadata({
  params,
}: {
  params: { slug: string };
}): Metadata {
  const comparison = comparisons.find((c) => c.slug === params.slug);
  if (!comparison) return {};

  return {
    title: comparison.metaTitle,
    description: comparison.metaDescription,
    openGraph: {
      title: comparison.metaTitle,
      description: comparison.metaDescription,
    },
  };
}

function BreadcrumbJsonLd({ comparison }: { comparison: (typeof comparisons)[0] }) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: [
      {
        "@type": "ListItem",
        position: 1,
        name: "Accueil",
        item: "https://agent-catalog-fr.vercel.app",
      },
      {
        "@type": "ListItem",
        position: 2,
        name: "Comparatifs",
        item: "https://agent-catalog-fr.vercel.app/comparatif",
      },
      {
        "@type": "ListItem",
        position: 3,
        name: comparison.title,
        item: `https://agent-catalog-fr.vercel.app/comparatif/${comparison.slug}`,
      },
    ],
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function ComparatifPage({
  params,
}: {
  params: { slug: string };
}) {
  const comparison = comparisons.find((c) => c.slug === params.slug);
  if (!comparison) notFound();

  const related = comparison.relatedUseCases
    .map((slug) => useCases.find((uc) => uc.slug === slug))
    .filter(Boolean);

  return (
    <div className="mx-auto max-w-4xl px-4 py-16 sm:px-6 lg:px-8">
      <BreadcrumbJsonLd comparison={comparison} />

      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-muted-foreground mb-8">
        <Link href="/" className="hover:text-foreground transition-colors">
          Accueil
        </Link>
        <span>/</span>
        <Link href="/comparatif" className="hover:text-foreground transition-colors">
          Comparatifs
        </Link>
        <span>/</span>
        <span className="text-foreground">{comparison.title.slice(0, 40)}...</span>
      </nav>

      {/* Header */}
      <div className="mb-10">
        <div className="flex flex-wrap gap-2 mb-4">
          {comparison.options.map((o) => (
            <Badge key={o.name} variant="secondary" className="text-xs">
              {o.name}
            </Badge>
          ))}
        </div>
        <h1 className="text-2xl font-bold sm:text-3xl lg:text-4xl leading-tight">
          {comparison.title}
        </h1>
        <p className="mt-4 text-lg text-muted-foreground leading-relaxed">
          {comparison.intro}
        </p>
      </div>

      {/* Options overview */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 mb-12">
        {comparison.options.map((o) => (
          <Card key={o.name}>
            <CardContent className="pt-6">
              <h3 className="font-semibold text-base mb-2">{o.name}</h3>
              <p className="text-sm text-muted-foreground">{o.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Comparison table */}
      <div className="mb-12">
        <h2 className="text-xl font-bold mb-4">Comparatif détaillé</h2>
        <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
          <table className="w-full text-sm min-w-[500px]">
            <thead>
              <tr className="border-b">
                <th className="py-3 px-3 text-left font-medium text-muted-foreground text-xs uppercase tracking-wider">
                  Critère
                </th>
                {comparison.options.map((o) => (
                  <th
                    key={o.name}
                    className="py-3 px-3 text-left font-medium text-xs uppercase tracking-wider"
                  >
                    {o.name}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y">
              {comparison.rows.map((row) => (
                <tr key={row.feature} className="hover:bg-muted/30 transition-colors">
                  <td className="py-3 px-3 font-medium text-sm">{row.feature}</td>
                  {row.values.map((val, i) => (
                    <td key={i} className="py-3 px-3 text-sm text-muted-foreground">
                      {val}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Verdict */}
      <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 mb-12">
        <h2 className="text-xl font-bold mb-3">Notre verdict</h2>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {comparison.verdict}
        </p>
      </div>

      {/* Related use cases */}
      {related.length > 0 && (
        <div className="mb-12">
          <h2 className="text-xl font-bold mb-4">Workflows associés</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {related.map(
              (uc) =>
                uc && (
                  <Link
                    key={uc.slug}
                    href={`/use-case/${uc.slug}`}
                    className="group rounded-xl border bg-card p-4 transition-all hover:shadow-sm hover:border-primary/30"
                  >
                    <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
                      {uc.title}
                    </h3>
                    <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                      {uc.subtitle}
                    </p>
                  </Link>
                )
            )}
          </div>
        </div>
      )}

      {/* CTAs */}
      <div className="flex flex-col sm:flex-row gap-3 justify-center">
        <Button size="lg" asChild>
          <Link href="/catalogue">Voir tous les workflows</Link>
        </Button>
        <Button size="lg" variant="outline" asChild>
          <Link href="/calculateur-roi">Calculer votre ROI</Link>
        </Button>
      </div>
    </div>
  );
}
