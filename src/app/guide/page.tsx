import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { guides } from "@/data/guides";

export const metadata: Metadata = {
  title: "Guides IA — Automatiser votre entreprise avec l'IA",
  description:
    "Guides pratiques pour déployer des agents IA dans votre entreprise. Support client, sales, RH, finance. Stack technique, ROI et conformité RGPD.",
};

function GuidesCollectionJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: "Guides pratiques IA pour l'entreprise",
    description: "Guides pour déployer des agents IA. Support client, sales, RH, finance.",
    url: "https://agent-catalog-fr.vercel.app/guide",
    hasPart: guides.map((g) => ({
      "@type": "Article",
      headline: g.title,
      url: `https://agent-catalog-fr.vercel.app/guide/${g.slug}`,
      datePublished: g.publishedAt,
    })),
  };
  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function GuidesPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <GuidesCollectionJsonLd />
      <div className="mb-10">
        <Badge variant="secondary" className="mb-3 text-xs">
          Ressources gratuites
        </Badge>
        <h1 className="text-3xl font-bold sm:text-4xl">
          Guides pratiques IA
        </h1>
        <p className="mt-3 text-muted-foreground max-w-2xl">
          {guides.length} guides pour déployer des agents IA dans votre
          entreprise. Stack technique, ROI, conformité RGPD et étapes
          d&apos;implémentation.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        {guides.map((guide) => (
          <Link key={guide.slug} href={`/guide/${guide.slug}`} className="group block">
            <Card className="h-full transition-all duration-200 hover:shadow-md hover:border-primary/30 group-hover:-translate-y-0.5">
              <CardHeader className="pb-3">
                <div className="flex items-center gap-2 mb-2">
                  <Badge variant="secondary" className="text-xs">
                    {guide.category}
                  </Badge>
                  <span className="text-xs text-muted-foreground">
                    {guide.readTime} de lecture
                  </span>
                </div>
                <h2 className="text-lg font-semibold leading-snug group-hover:text-primary transition-colors">
                  {guide.title}
                </h2>
              </CardHeader>
              <CardContent className="pt-0">
                <p className="text-sm text-muted-foreground line-clamp-3">
                  {guide.excerpt}
                </p>
                <p className="mt-3 text-sm font-medium text-primary">
                  Lire le guide &rarr;
                </p>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {/* Cross-links */}
      <div className="mt-12 grid gap-4 sm:grid-cols-2">
        <Link
          href="/calculateur-roi"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <p className="font-semibold text-sm group-hover:text-primary transition-colors">
            Calculez votre ROI
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Estimez les gains de temps et d&apos;argent avec notre outil gratuit.
          </p>
        </Link>
        <Link
          href="/comparatif"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <p className="font-semibold text-sm group-hover:text-primary transition-colors">
            Comparatifs outils IA
          </p>
          <p className="text-xs text-muted-foreground mt-1">
            Claude vs ChatGPT, n8n vs Make, Agent IA vs Chatbot...
          </p>
        </Link>
      </div>

      {/* CTA */}
      <div className="mt-12 rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8 text-center">
        <h2 className="text-xl font-bold sm:text-2xl">
          Vous préférez un workflow prêt à déployer ?
        </h2>
        <p className="mt-2 text-sm text-muted-foreground max-w-lg mx-auto">
          Nos guides expliquent la théorie. Nos workflows vous donnent le code,
          le schéma d&apos;architecture et les étapes détaillées.
        </p>
        <div className="mt-4 flex flex-col sm:flex-row gap-3 justify-center">
          <Link
            href="/catalogue"
            className="rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Voir les workflows
          </Link>
          <Link
            href="/demande"
            className="rounded-lg border px-6 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
          >
            Demander un workflow sur mesure
          </Link>
        </div>
      </div>

      {/* Newsletter */}
      <div className="mt-12">
        <NewsletterSignup variant="inline" />
      </div>
    </div>
  );
}
