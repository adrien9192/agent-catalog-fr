import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { UseCaseCard } from "@/components/use-case-card";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";
import { guides } from "@/data/guides";
import { useCases } from "@/data/use-cases";
import { comparisons } from "@/data/comparisons";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return guides.map((g) => ({ slug: g.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const guide = guides.find((g) => g.slug === slug);
  if (!guide) return {};
  return {
    title: guide.metaTitle,
    description: guide.metaDescription,
    openGraph: {
      title: guide.metaTitle,
      description: guide.metaDescription,
      type: "article",
      publishedTime: guide.publishedAt,
      modifiedTime: guide.updatedAt,
    },
  };
}

export default async function GuidePage({ params }: PageProps) {
  const { slug } = await params;
  const guide = guides.find((g) => g.slug === slug);
  if (!guide) notFound();

  const related = guide.relatedUseCases
    .map((slug) => useCases.find((uc) => uc.slug === slug))
    .filter(Boolean);

  // Find related comparisons based on guide content keywords
  const guideWords = guide.title.toLowerCase();
  const relatedComparisons = comparisons.filter((c) =>
    c.options.some((o) => guideWords.includes(o.name.toLowerCase())) ||
    c.title.toLowerCase().split(" ").some((w) => w.length > 4 && guideWords.includes(w))
  ).slice(0, 3);

  return (
    <article className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: "Guides", url: "https://agent-catalog-fr.vercel.app/guide" },
          { name: guide.title, url: `https://agent-catalog-fr.vercel.app/guide/${guide.slug}` },
        ]}
      />
      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{
          __html: JSON.stringify({
            "@context": "https://schema.org",
            "@type": "Article",
            headline: guide.title,
            description: guide.metaDescription,
            datePublished: guide.publishedAt,
            dateModified: guide.updatedAt,
            author: {
              "@type": "Person",
              name: "Adrien Laine",
              jobTitle: "Expert IA & Automatisation",
              url: "https://agent-catalog-fr.vercel.app/a-propos",
              worksFor: {
                "@type": "Organization",
                name: "AgentCatalog",
                url: "https://agent-catalog-fr.vercel.app",
              },
            },
            publisher: {
              "@type": "Organization",
              name: "AgentCatalog",
              url: "https://agent-catalog-fr.vercel.app",
            },
            mainEntityOfPage: `https://agent-catalog-fr.vercel.app/guide/${guide.slug}`,
            image: `https://agent-catalog-fr.vercel.app/guide/${guide.slug}/opengraph-image`,
            articleSection: guide.category,
            inLanguage: "fr-FR",
            isAccessibleForFree: true,
            wordCount: guide.sections.reduce((acc, s) => acc + s.content.split(" ").length, 0),
          }),
        }}
      />

      {/* Breadcrumb */}
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <Link href="/guide" className="hover:text-foreground">Guides</Link>
        {" / "}
        <span className="text-foreground line-clamp-1">{guide.title}</span>
      </nav>

      {/* Header */}
      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <Badge variant="secondary">{guide.category}</Badge>
          <span className="text-sm text-muted-foreground">{guide.readTime} de lecture</span>
          <span className="text-sm text-muted-foreground">
            Publié le {new Date(guide.publishedAt).toLocaleDateString("fr-FR", {
              day: "numeric",
              month: "long",
              year: "numeric",
            })}
          </span>
        </div>
        {/* Author byline for E-E-A-T */}
        <div className="flex items-center gap-3 mb-6">
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
            AL
          </div>
          <div>
            <Link href="/a-propos" className="text-sm font-medium hover:text-primary transition-colors">
              Adrien Laine
            </Link>
            <p className="text-xs text-muted-foreground">Expert IA & Automatisation | AgentCatalog</p>
          </div>
        </div>
        <h1 className="text-2xl font-bold sm:text-4xl lg:text-5xl leading-tight">
          {guide.title}
        </h1>
        <p className="mt-4 text-base sm:text-lg text-muted-foreground max-w-3xl leading-relaxed">
          {guide.excerpt}
        </p>
      </header>

      {/* Table of contents */}
      <div className="mb-10 rounded-xl border bg-muted/30 p-4 sm:p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-3">
          Sommaire
        </h2>
        <ol className="space-y-1.5">
          {guide.sections.map((section, i) => (
            <li key={i}>
              <a
                href={`#section-${i}`}
                className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                {i + 1}. {section.title}
              </a>
            </li>
          ))}
        </ol>
      </div>

      {/* Content */}
      <div className="space-y-10">
        {guide.sections.map((section, i) => (
          <section key={i} id={`section-${i}`}>
            <h2 className="text-xl font-bold sm:text-2xl mb-4">
              {section.title}
            </h2>
            <div className="prose prose-sm max-w-none text-muted-foreground">
              {section.content.split("\n\n").map((paragraph, pi) => (
                <p key={pi} className="leading-relaxed mb-4">
                  {paragraph.split("**").map((part, partI) =>
                    partI % 2 === 1 ? (
                      <strong key={partI} className="text-foreground font-semibold">{part}</strong>
                    ) : (
                      part
                    )
                  )}
                </p>
              ))}
            </div>
          </section>
        ))}
      </div>

      <Separator className="my-10" />

      {/* CTA */}
      <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <div className="flex-1">
            <p className="font-semibold text-lg">
              Passez de la théorie à la pratique
            </p>
            <p className="mt-1 text-sm text-muted-foreground">
              Nos workflows incluent le code, le schéma d&apos;architecture et les étapes
              détaillées. Prêts à copier et déployer.
            </p>
          </div>
          <Link
            href="/catalogue"
            className="shrink-0 rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Voir les workflows
          </Link>
        </div>
      </div>

      {/* Related use cases */}
      {related.length > 0 && (
        <div className="mt-12">
          <h2 className="text-xl font-bold mb-6">Workflows associés</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {related.map((uc) => uc && (
              <UseCaseCard key={uc.slug} useCase={uc} />
            ))}
          </div>
        </div>
      )}

      {/* Cross-links: comparisons + ROI */}
      <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Link
          href="/calculateur-roi"
          className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Calculateur ROI
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Estimez vos gains avec un agent IA.
          </p>
        </Link>
        {relatedComparisons.length > 0
          ? relatedComparisons.slice(0, 2).map((c) => (
              <Link
                key={c.slug}
                href={`/comparatif/${c.slug}`}
                className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <div className="flex items-center gap-1.5 mb-1.5">
                  {c.options.map((o) => (
                    <Badge key={o.name} variant="outline" className="text-[10px] px-1.5 py-0">
                      {o.name}
                    </Badge>
                  ))}
                </div>
                <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
                  {c.title}
                </h3>
              </Link>
            ))
          : (
            <>
              <Link
                href="/comparatif/n8n-vs-make-vs-zapier"
                className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
                  n8n vs Make vs Zapier
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Quel orchestrateur pour vos agents ?
                </p>
              </Link>
              <Link
                href="/comparatif/agent-ia-vs-chatbot"
                className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
                  Agent IA vs Chatbot
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  Comprendre les différences clés.
                </p>
              </Link>
            </>
          )
        }
      </div>

      {/* Newsletter */}
      <div className="mt-8">
        <NewsletterSignup variant="inline" />
      </div>

      {/* Other guides */}
      <div className="mt-12 border-t pt-8">
        <h2 className="text-xl font-bold mb-4">Autres guides</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {guides
            .filter((g) => g.slug !== slug)
            .slice(0, 4)
            .map((g) => (
              <Link
                key={g.slug}
                href={`/guide/${g.slug}`}
                className="rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="secondary" className="text-xs">{g.category}</Badge>
                  <span className="text-xs text-muted-foreground">{g.readTime}</span>
                </div>
                <h3 className="font-semibold text-sm leading-snug hover:text-primary transition-colors">
                  {g.title}
                </h3>
              </Link>
            ))}
        </div>
      </div>
    </article>
  );
}
