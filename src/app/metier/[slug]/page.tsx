import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { UseCaseCard } from "@/components/use-case-card";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { useCases } from "@/data/use-cases";
import { metiers } from "@/data/metiers";
import { guides } from "@/data/guides";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return metiers.map((m) => ({ slug: m.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const metier = metiers.find((m) => m.slug === slug);
  if (!metier) return {};
  const count = useCases.filter((uc) =>
    uc.metiers.some((m) => m.toLowerCase().replace(/\s+/g, "-") === slug || m === metier.name)
  ).length;
  return {
    title: `Agents IA pour le métier ${metier.name} — ${count} workflows`,
    description: `${metier.description} ${count} workflows documentés avec tutoriel et ROI.`,
    alternates: { canonical: `/metier/${slug}` },
    openGraph: {
      title: `Agents IA — ${metier.name}`,
      description: `${count} workflows IA pour le métier ${metier.name}. Tutoriels, stack technique et ROI inclus.`,
    },
  };
}

export default async function MetierPage({ params }: PageProps) {
  const { slug } = await params;
  const metier = metiers.find((m) => m.slug === slug);
  if (!metier) notFound();

  const metierName = metier.name;
  const filtered = useCases.filter((uc) =>
    uc.metiers.some((m) => m.toLowerCase().replace(/\s+/g, "-") === slug || m === metierName)
  );

  const metierKeywords = metier.name.toLowerCase().split(/[\s/]+/);
  const relatedGuides = guides.filter((g) =>
    metierKeywords.some((kw) =>
      g.title.toLowerCase().includes(kw) ||
      g.metaDescription.toLowerCase().includes(kw) ||
      g.category.toLowerCase().includes(kw)
    )
  ).slice(0, 3);

  const collectionJsonLd = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: `Agents IA — ${metier.name}`,
    description: metier.description,
    url: `https://agent-catalog-fr.vercel.app/metier/${metier.slug}`,
    numberOfItems: filtered.length,
    provider: {
      "@type": "Organization",
      name: "AgentCatalog",
      url: "https://agent-catalog-fr.vercel.app",
    },
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(collectionJsonLd) }}
      />
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: `Métier ${metier.name}`, url: `https://agent-catalog-fr.vercel.app/metier/${metier.slug}` },
        ]}
      />
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <span className="text-foreground">Métier : {metier.name}</span>
      </nav>

      <header className="mb-8">
        <span className="text-4xl mb-3 block">{metier.icon}</span>
        <h1 className="text-3xl font-bold sm:text-4xl">
          Agents IA — {metier.name}
        </h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">{metier.description}</p>
      </header>

      {filtered.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((uc) => (
            <UseCaseCard key={uc.slug} useCase={uc} />
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed p-6 sm:p-12 text-center">
          <p className="text-muted-foreground">
            Aucun cas d&apos;usage disponible pour ce métier pour le moment.
          </p>
          <Link href="/catalogue" className="mt-2 inline-block text-sm text-primary hover:underline">
            Voir le catalogue complet
          </Link>
        </div>
      )}

      {/* CTA section */}
      <div className="mt-12 rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="flex-1">
            <p className="font-semibold">Vous ne trouvez pas le workflow {metier.name} qu&apos;il vous faut ?</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Décrivez votre besoin et recevez un workflow sur mesure sous 5 jours.
            </p>
          </div>
          <Link
            href={`/demande?q=workflow ${metier.name}`}
            className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Demander un workflow
          </Link>
        </div>
      </div>

      {/* Cross-links */}
      <div className="mt-8 grid gap-4 sm:grid-cols-2">
        <Link
          href="/calculateur-roi"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <p className="font-semibold text-sm group-hover:text-primary transition-colors">
            Calculez votre ROI
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Estimez les gains de temps et d&apos;argent avec notre calculateur gratuit.
          </p>
        </Link>
        <Link
          href="/comparatif"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <p className="font-semibold text-sm group-hover:text-primary transition-colors">
            Comparez les solutions
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Agent IA vs Chatbot, Claude vs ChatGPT, n8n vs Make...
          </p>
        </Link>
      </div>

      {/* Newsletter */}
      <div className="mt-8">
        <NewsletterSignup variant="inline" />
      </div>

      {/* Related guides */}
      {relatedGuides.length > 0 && (
        <div className="mt-12">
          <h2 className="text-xl font-bold mb-4">Guides pratiques associés</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {relatedGuides.map((g) => (
              <Link
                key={g.slug}
                href={`/guide/${g.slug}`}
                className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="secondary" className="text-xs">{g.category}</Badge>
                  <span className="text-xs text-muted-foreground">{g.readTime}</span>
                </div>
                <h3 className="font-semibold text-sm leading-snug group-hover:text-primary transition-colors">
                  {g.title}
                </h3>
                <p className="mt-1 text-xs text-primary font-medium">Lire le guide &rarr;</p>
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Other metiers */}
      <div className="mt-16 border-t pt-8">
        <h2 className="text-xl font-bold mb-4">Autres métiers</h2>
        <div className="flex flex-wrap gap-2">
          {metiers
            .filter((m) => m.slug !== slug)
            .slice(0, 8)
            .map((m) => (
              <Link
                key={m.slug}
                href={`/metier/${m.slug}`}
                className="rounded-lg border px-3 py-2 text-sm transition-colors hover:bg-accent"
              >
                {m.icon} {m.name}
              </Link>
            ))}
        </div>
      </div>
    </div>
  );
}
