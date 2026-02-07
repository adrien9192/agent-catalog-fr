import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { DifficultyBadge } from "@/components/difficulty-badge";
import { UseCaseJsonLd } from "@/components/json-ld";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";
import { RelatedUseCases } from "@/components/related-use-cases";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { StickyCTABar } from "@/components/sticky-cta-bar";
import { useCases } from "@/data/use-cases";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return useCases.map((uc) => ({ slug: uc.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const uc = useCases.find((u) => u.slug === slug);
  if (!uc) return {};
  return {
    title: uc.metaTitle,
    description: uc.metaDescription,
    openGraph: {
      title: uc.metaTitle,
      description: uc.metaDescription,
    },
  };
}

export default async function UseCasePage({ params }: PageProps) {
  const { slug } = await params;
  const uc = useCases.find((u) => u.slug === slug);
  if (!uc) notFound();

  return (
    <article className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8 pb-20">
      <UseCaseJsonLd useCase={uc} />
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: "Catalogue", url: "https://agent-catalog-fr.vercel.app/catalogue" },
          { name: uc.title, url: `https://agent-catalog-fr.vercel.app/use-case/${uc.slug}` },
        ]}
      />
      <StickyCTABar title={uc.title} difficulty={uc.difficulty} />
      {/* Breadcrumb */}
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <Link href="/catalogue" className="hover:text-foreground">Catalogue</Link>
        {" / "}
        <span className="text-foreground">{uc.title}</span>
      </nav>

      {/* Header */}
      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <DifficultyBadge difficulty={uc.difficulty} />
          {uc.functions.map((fn) => (
            <Badge key={fn} variant="secondary">{fn}</Badge>
          ))}
        </div>
        <h1 className="text-3xl font-bold sm:text-4xl lg:text-5xl">{uc.title}</h1>
        <p className="mt-3 text-lg text-muted-foreground max-w-3xl">{uc.subtitle}</p>
        <div className="mt-4 flex flex-wrap gap-1.5">
          {uc.sectors.map((s) => (
            <Link key={s} href={`/secteur/${s.toLowerCase().replace(/\s+/g, "-")}`}>
              <Badge variant="outline" className="cursor-pointer">{s}</Badge>
            </Link>
          ))}
          {uc.metiers.map((m) => (
            <Link key={m} href={`/metier/${m.toLowerCase().replace(/\s+/g, "-")}`}>
              <Badge variant="outline" className="cursor-pointer">{m}</Badge>
            </Link>
          ))}
        </div>
      </header>

      <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
        <div className="reading-width space-y-10 lg:mx-0 lg:max-w-none">
          {/* 1. Présentation */}
          <section>
            <h2 className="text-2xl font-bold mb-4">Présentation du cas d&apos;usage</h2>

            <div className="space-y-6">
              <div>
                <h3 className="font-semibold text-lg mb-2">Problème</h3>
                <p className="text-muted-foreground leading-relaxed">{uc.problem}</p>
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-2">Valeur apportée</h3>
                <p className="text-muted-foreground leading-relaxed">{uc.value}</p>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold">Inputs</h3>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      {uc.inputs.map((input, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-primary shrink-0" />
                          {input}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold">Outputs</h3>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      {uc.outputs.map((output, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-chart-2 shrink-0" />
                          {output}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold">Risques</h3>
                  </CardHeader>
                  <CardContent>
                    <ul className="space-y-1 text-sm text-muted-foreground">
                      {uc.risks.map((risk, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="mt-1.5 h-1.5 w-1.5 rounded-full bg-destructive shrink-0" />
                          {risk}
                        </li>
                      ))}
                    </ul>
                  </CardContent>
                </Card>
                <Card className="border-primary/20 bg-primary/5">
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold">ROI indicatif</h3>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm font-medium">{uc.roiIndicatif}</p>
                  </CardContent>
                </Card>
              </div>
            </div>
          </section>

          <Separator />

          {/* 2. Stack recommandée */}
          <section>
            <h2 className="text-2xl font-bold mb-4">Stack recommandée</h2>
            <div className="grid gap-4 sm:grid-cols-2">
              <Card>
                <CardHeader className="pb-2">
                  <h3 className="font-semibold">Stack principale</h3>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {uc.recommendedStack.map((item, i) => (
                      <li key={i} className="flex items-center justify-between text-sm">
                        <span>{item.name}</span>
                        <Badge variant="outline" className="text-xs">{item.category}</Badge>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <h3 className="font-semibold">Alternatives low-cost</h3>
                </CardHeader>
                <CardContent>
                  <ul className="space-y-2">
                    {uc.lowCostAlternatives.map((item, i) => (
                      <li key={i} className="flex items-center justify-between text-sm">
                        <span>{item.name}</span>
                        <div className="flex items-center gap-1">
                          {item.isFree && (
                            <Badge variant="secondary" className="text-xs">Gratuit</Badge>
                          )}
                          <Badge variant="outline" className="text-xs">{item.category}</Badge>
                        </div>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>

            {/* Architecture diagram */}
            <div className="mt-6">
              <h3 className="font-semibold mb-3">Schéma d&apos;architecture</h3>
              <div className="overflow-x-auto rounded-lg border bg-muted/50 p-4">
                <pre className="text-xs sm:text-sm font-mono whitespace-pre leading-relaxed">
                  {uc.architectureDiagram}
                </pre>
              </div>
            </div>
          </section>

          <Separator />

          {/* 3. Tutoriel */}
          <section>
            <h2 className="text-2xl font-bold mb-6">Tutoriel d&apos;implémentation</h2>
            <div className="space-y-8">
              {uc.tutorial.map((section, i) => (
                <div key={i}>
                  <h3 className="text-xl font-semibold mb-3">
                    <span className="text-primary mr-2">{i + 1}.</span>
                    {section.title}
                  </h3>
                  <div className="prose prose-sm max-w-none text-muted-foreground mb-4">
                    {section.content.split("\n").map((paragraph, pi) => (
                      <p key={pi} className="leading-relaxed mb-2">{paragraph}</p>
                    ))}
                  </div>
                  {section.codeSnippets?.map((snippet, si) => (
                    <div key={si} className="mb-4">
                      {snippet.filename && (
                        <div className="rounded-t-lg border border-b-0 bg-muted px-3 py-1.5 text-xs font-mono text-muted-foreground">
                          {snippet.filename}
                        </div>
                      )}
                      <div className={`overflow-x-auto rounded-lg border bg-[#1e1e2e] p-4 ${snippet.filename ? "rounded-t-none" : ""}`}>
                        <pre className="text-xs sm:text-sm font-mono text-[#cdd6f4] whitespace-pre">
                          {snippet.code}
                        </pre>
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </section>

          {/* Mid-page CTA */}
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <p className="font-semibold">Besoin d&apos;aide pour implémenter ce workflow ?</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Notre équipe peut adapter ce workflow à votre contexte spécifique.
                </p>
              </div>
              <div className="flex gap-2">
                <Link
                  href="/demande"
                  className="shrink-0 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Demander un accompagnement
                </Link>
              </div>
            </div>
          </div>

          <Separator />

          {/* 4. Workflow n8n / Automatisation */}
          {uc.n8nWorkflow && (
            <section>
              <h2 className="text-2xl font-bold mb-4">Workflow n8n / Automatisation</h2>
              <Card>
                <CardContent className="pt-6">
                  <p className="text-sm text-muted-foreground leading-relaxed mb-4">
                    {uc.n8nWorkflow.description}
                  </p>
                  <div className="flex flex-wrap gap-2 mb-4">
                    <Badge variant="secondary" className="text-xs">
                      Trigger : {uc.n8nWorkflow.triggerType}
                    </Badge>
                  </div>
                  <h4 className="font-semibold text-sm mb-2">Nodes du workflow :</h4>
                  <div className="flex flex-wrap gap-1.5">
                    {uc.n8nWorkflow.nodes.map((node, i) => (
                      <div key={i} className="flex items-center gap-1">
                        {i > 0 && <span className="text-muted-foreground text-xs">→</span>}
                        <Badge variant="outline" className="text-xs font-mono">
                          {node}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </section>
          )}

          <Separator />

          {/* 5. Enterprise */}
          {uc.enterprise && (
            <section>
              <h2 className="text-2xl font-bold mb-4">Considérations Enterprise</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold text-sm">Données personnelles (PII/RGPD)</h3>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">{uc.enterprise.piiHandling}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold text-sm">Audit & Traçabilité</h3>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">{uc.enterprise.auditLog}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold text-sm">Human-in-the-Loop</h3>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">{uc.enterprise.humanInTheLoop}</p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader className="pb-2">
                    <h3 className="font-semibold text-sm">Monitoring & Alertes</h3>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-muted-foreground leading-relaxed">{uc.enterprise.monitoring}</p>
                  </CardContent>
                </Card>
              </div>
            </section>
          )}
        </div>

        {/* Sidebar */}
        <aside className="hidden lg:block">
          <div className="sticky top-20 space-y-4">
            <Card>
              <CardHeader className="pb-2">
                <h3 className="font-semibold text-sm">Informations</h3>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Difficulté</span>
                  <DifficultyBadge difficulty={uc.difficulty} />
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Fonctions</span>
                  <span>{uc.functions.join(", ")}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Mis à jour</span>
                  <span>{uc.updatedAt}</span>
                </div>
                {uc.estimatedTime && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Temps estimé</span>
                    <span className="font-medium">{uc.estimatedTime}</span>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-2">
                <h3 className="font-semibold text-sm">Sommaire</h3>
              </CardHeader>
              <CardContent>
                <nav className="space-y-1 text-sm">
                  <p className="text-muted-foreground hover:text-foreground cursor-pointer">Présentation</p>
                  <p className="text-muted-foreground hover:text-foreground cursor-pointer">Stack recommandée</p>
                  <p className="text-muted-foreground hover:text-foreground cursor-pointer">Tutoriel</p>
                  <p className="text-muted-foreground hover:text-foreground cursor-pointer">Workflow n8n</p>
                  <p className="text-muted-foreground hover:text-foreground cursor-pointer">Enterprise</p>
                </nav>
              </CardContent>
            </Card>
          </div>
        </aside>
      </div>

      {/* Newsletter CTA */}
      <div className="mt-12">
        <NewsletterSignup variant="inline" />
      </div>

      {/* Related use cases */}
      <RelatedUseCases
        currentSlug={uc.slug}
        functions={uc.functions}
        sectors={uc.sectors}
      />
    </article>
  );
}
