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
import { guides } from "@/data/guides";

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
          {uc.estimatedTime && (
            <Badge variant="outline" className="text-xs">
              ⏱ {uc.estimatedTime}
            </Badge>
          )}
          {uc.functions.map((fn) => (
            <Badge key={fn} variant="secondary">{fn}</Badge>
          ))}
        </div>
        <h1 className="text-2xl font-bold sm:text-4xl lg:text-5xl">{uc.title}</h1>
        <p className="mt-3 text-base sm:text-lg text-muted-foreground max-w-3xl">{uc.subtitle}</p>
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

      {/* Mobile TOC */}
      <div className="lg:hidden mb-8 rounded-xl border bg-muted/30 p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-2">Sommaire</h2>
        <nav className="flex flex-wrap gap-2">
          {uc.storytelling && (
            <a href="#histoire" className="rounded-full border px-3 py-1 text-xs hover:bg-accent transition-colors">Cas concret</a>
          )}
          {uc.beforeAfter && (
            <a href="#avant-apres" className="rounded-full border px-3 py-1 text-xs hover:bg-accent transition-colors">Avant / Après</a>
          )}
          <a href="#tutoriel" className="rounded-full border px-3 py-1 text-xs hover:bg-accent transition-colors">
            {uc.n8nTutorial ? "Tutoriel n8n" : "Tutoriel"}
          </a>
        </nav>
      </div>

      <div className="grid gap-8 lg:grid-cols-[1fr_320px]">
        <div className="reading-width space-y-10 lg:mx-0 lg:max-w-none">
          {/* 1. Storytelling */}
          {uc.storytelling && (
            <section id="histoire">
              <div className="rounded-xl border bg-muted/20 p-5 sm:p-8 space-y-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Badge variant="secondary" className="text-xs">{uc.storytelling.sector}</Badge>
                  <span>Cas concret</span>
                </div>
                <p className="font-semibold">{uc.storytelling.persona}</p>
                <p className="text-sm text-muted-foreground leading-relaxed">{uc.storytelling.painPoint}</p>
                <Separator />
                <p className="text-sm leading-relaxed">{uc.storytelling.story}</p>
                <div className="rounded-lg border-l-4 border-l-primary bg-primary/5 p-4">
                  <p className="text-sm font-medium leading-relaxed">{uc.storytelling.result}</p>
                </div>
              </div>
            </section>
          )}

          {/* 2. Before / After */}
          {uc.beforeAfter && (
            <section id="avant-apres">
              <h2 className="text-2xl font-bold mb-4">Ce que fait ce workflow</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="rounded-xl border p-4 sm:p-5">
                  <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">Avant — {uc.beforeAfter.inputLabel}</p>
                  <div className="rounded-lg bg-muted/50 p-4">
                    <p className="text-sm italic leading-relaxed">&laquo; {uc.beforeAfter.inputText} &raquo;</p>
                  </div>
                </div>
                <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 sm:p-5">
                  <p className="text-xs font-semibold uppercase tracking-wider text-primary mb-3">Après — Classification IA</p>
                  <div className="space-y-2">
                    {uc.beforeAfter.outputFields.map((field, i) => (
                      <div key={i} className="flex items-start gap-2 text-sm">
                        <span className="font-medium shrink-0 w-28">{field.label}</span>
                        <span className="text-muted-foreground">{field.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              <div className="mt-4 rounded-lg border bg-muted/30 p-4">
                <p className="text-sm font-medium">{uc.roiIndicatif}</p>
                <Link href="/calculateur-roi" className="text-xs text-primary font-medium hover:underline mt-1 inline-block">
                  Calculer votre ROI personnalisé &rarr;
                </Link>
              </div>
            </section>
          )}

          <Separator />

          {/* Prerequisites */}
          {uc.prerequisites && uc.prerequisites.length > 0 && (
            <div className="rounded-xl border border-primary/20 bg-primary/5 p-4 sm:p-6">
              <h3 className="font-semibold mb-3">Avant de commencer</h3>
              <ul className="space-y-2">
                {uc.prerequisites.map((p, i) => (
                  <li key={i} className="flex items-start gap-2 text-sm">
                    <span className="mt-0.5 text-primary">&#10003;</span>
                    <span>{p}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* 3. Tutoriel n8n (primary) or code tutorial (fallback) */}
          <section id="tutoriel">
            {uc.n8nTutorial && uc.n8nTutorial.length > 0 ? (
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold mb-2">Tutoriel n8n — pas à pas</h2>
                  <p className="text-sm text-muted-foreground">
                    Suivez ce guide nœud par nœud pour construire le workflow dans n8n.
                    Chaque étape inclut les instructions de configuration, les adaptations possibles et la gestion des erreurs.
                  </p>
                </div>

                {/* Workflow overview */}
                {uc.n8nWorkflow && (
                  <div className="mb-8 rounded-xl border bg-muted/30 p-4 sm:p-6">
                    <h3 className="font-semibold text-sm mb-3">Vue d&apos;ensemble du workflow</h3>
                    <div className="flex flex-wrap items-center gap-1.5 mb-3">
                      {uc.n8nWorkflow.nodes.map((node, i) => (
                        <div key={i} className="flex items-center gap-1">
                          {i > 0 && <span className="text-muted-foreground text-xs">→</span>}
                          <Badge variant="outline" className="text-xs font-mono">
                            {node}
                          </Badge>
                        </div>
                      ))}
                    </div>
                    <Badge variant="secondary" className="text-xs">
                      Trigger : {uc.n8nWorkflow.triggerType}
                    </Badge>
                  </div>
                )}

                {/* N8n tutorial steps */}
                <div className="space-y-6">
                  {uc.n8nTutorial.map((step, i) => (
                    <div key={i} className="rounded-xl border overflow-hidden">
                      {/* Step header */}
                      <div className="flex items-center gap-3 border-b bg-muted/40 px-4 py-3 sm:px-6">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground text-sm font-bold">
                          {i + 1}
                        </span>
                        <div className="flex items-center gap-2 min-w-0">
                          <span className="text-lg" aria-hidden="true">{step.nodeIcon}</span>
                          <h3 className="font-semibold truncate">{step.nodeLabel}</h3>
                          <Badge variant="secondary" className="text-[10px] shrink-0 hidden sm:inline-flex">{step.nodeType}</Badge>
                        </div>
                      </div>

                      <div className="p-4 sm:p-6 space-y-4">
                        {/* Description */}
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {step.description}
                        </p>

                        {/* Expected output */}
                        {step.expectedOutput && (
                          <div className="rounded-lg border border-chart-2/30 bg-chart-2/5 p-3">
                            <p className="text-xs font-semibold mb-1.5 text-chart-2">Résultat attendu :</p>
                            <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap">{step.expectedOutput}</pre>
                          </div>
                        )}

                        {/* Variants (when available) or single configuration */}
                        {step.variants && step.variants.length > 0 ? (
                          <div className="space-y-3">
                            <h4 className="text-sm font-semibold flex items-center gap-1.5">
                              <span className="text-primary">&#9655;</span> Choisissez votre outil
                            </h4>
                            <p className="text-xs text-muted-foreground">{step.configuration}</p>
                            <div className="space-y-2">
                              {step.variants.map((variant, vi) => (
                                <details key={vi} className="group/variant rounded-lg border overflow-hidden">
                                  <summary className="cursor-pointer flex items-center gap-2 px-4 py-2.5 hover:bg-muted/40 transition-colors select-none">
                                    <span className="text-base" aria-hidden="true">{variant.toolIcon}</span>
                                    <span className="text-sm font-medium">{variant.toolName}</span>
                                    {variant.isFree && (
                                      <Badge variant="secondary" className="text-[10px] ml-1">Gratuit</Badge>
                                    )}
                                    <span className="ml-auto text-xs text-muted-foreground group-open/variant:hidden">Voir la config</span>
                                    <span className="ml-auto text-xs text-muted-foreground hidden group-open/variant:inline">Masquer</span>
                                  </summary>
                                  <div className="border-t px-4 py-3 space-y-3">
                                    <div className="overflow-x-auto rounded-lg border bg-[#1e1e2e] p-4">
                                      <pre className="text-xs sm:text-sm font-mono text-[#cdd6f4] whitespace-pre-wrap">
                                        {variant.configuration}
                                      </pre>
                                    </div>
                                    {variant.errorHandling && (
                                      <div className="rounded-lg border border-destructive/20 bg-destructive/5 p-3 text-xs text-muted-foreground leading-relaxed">
                                        <p className="font-semibold text-foreground mb-1">Erreurs fréquentes :</p>
                                        {variant.errorHandling.split("\n").map((line, li) => (
                                          <p key={li} className="mb-0.5 last:mb-0">{line}</p>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                </details>
                              ))}
                            </div>
                            {/* General error handling for the step */}
                            {step.errorHandling && (
                              <details className="group">
                                <summary className="cursor-pointer text-sm font-semibold flex items-center gap-1.5 select-none">
                                  <span className="text-destructive">&#9888;</span> Erreurs générales
                                  <span className="ml-auto text-xs text-muted-foreground group-open:hidden">Afficher</span>
                                  <span className="ml-auto text-xs text-muted-foreground hidden group-open:inline">Masquer</span>
                                </summary>
                                <div className="mt-2 rounded-lg border border-destructive/20 bg-destructive/5 p-4 text-sm text-muted-foreground leading-relaxed">
                                  {step.errorHandling.split("\n").map((line, li) => (
                                    <p key={li} className="mb-1 last:mb-0">{line}</p>
                                  ))}
                                </div>
                              </details>
                            )}
                          </div>
                        ) : (
                          <>
                            {/* Single configuration (no variants) */}
                            <div>
                              <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
                                <span className="text-primary">&#9655;</span> Configuration
                              </h4>
                              <div className="overflow-x-auto rounded-lg border bg-[#1e1e2e] p-4">
                                <pre className="text-xs sm:text-sm font-mono text-[#cdd6f4] whitespace-pre-wrap">
                                  {step.configuration}
                                </pre>
                              </div>
                            </div>

                            {/* Customization */}
                            {step.customization && (
                              <div>
                                <h4 className="text-sm font-semibold mb-2 flex items-center gap-1.5">
                                  <span className="text-primary">&#9881;</span> Adapter à votre contexte
                                </h4>
                                <div className="rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground leading-relaxed">
                                  {step.customization.split("\n").map((line, li) => (
                                    <p key={li} className="mb-1 last:mb-0">{line}</p>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Error handling */}
                            {step.errorHandling && (
                              <details className="group">
                                <summary className="cursor-pointer text-sm font-semibold flex items-center gap-1.5 select-none">
                                  <span className="text-destructive">&#9888;</span> Erreurs fréquentes et solutions
                                  <span className="ml-auto text-xs text-muted-foreground group-open:hidden">Afficher</span>
                                  <span className="ml-auto text-xs text-muted-foreground hidden group-open:inline">Masquer</span>
                                </summary>
                                <div className="mt-2 rounded-lg border border-destructive/20 bg-destructive/5 p-4 text-sm text-muted-foreground leading-relaxed">
                                  {step.errorHandling.split("\n").map((line, li) => (
                                    <p key={li} className="mb-1 last:mb-0">{line}</p>
                                  ))}
                                </div>
                              </details>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Collapsible code tutorial */}
                {uc.tutorial.length > 0 && (
                  <details className="mt-8 rounded-xl border">
                    <summary className="cursor-pointer px-4 py-3 sm:px-6 font-semibold text-sm flex items-center gap-2 select-none hover:bg-muted/40 transition-colors">
                      <span>&#128187;</span> Alternative : implémentation code (Python)
                      <span className="ml-auto text-xs text-muted-foreground">Cliquez pour déplier</span>
                    </summary>
                    <div className="px-4 pb-4 sm:px-6 sm:pb-6 space-y-8 border-t pt-6">
                      {uc.tutorial.map((section, i) => (
                        <div key={i}>
                          <h3 className="text-lg font-semibold mb-3">
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
                                <pre className="text-xs sm:text-sm font-mono text-[#cdd6f4] whitespace-pre-wrap">
                                  {snippet.code}
                                </pre>
                              </div>
                            </div>
                          ))}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </>
            ) : (
              <>
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
                            <pre className="text-xs sm:text-sm font-mono text-[#cdd6f4] whitespace-pre-wrap">
                              {snippet.code}
                            </pre>
                          </div>
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </>
            )}
          </section>

          {/* Contextual CTA */}
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <p className="font-semibold">Vous voulez ce workflow connecté à vos outils, avec vos règles métier ?</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  On configure le workflow pour vous en 48h : connexion à votre CRM, vos catégories, vos SLA, vos canaux de notification.
                </p>
              </div>
              <div className="flex gap-2">
                <Link
                  href="/demande"
                  className="shrink-0 rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Demander la configuration
                </Link>
              </div>
            </div>
          </div>

          {/* Workflow n8n summary (only when no n8n tutorial) */}
          {!uc.n8nTutorial && uc.n8nWorkflow && (
            <>
              <Separator />
              <section id="workflow-n8n">
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
            </>
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
                  {uc.storytelling && (
                    <a href="#histoire" className="block text-muted-foreground hover:text-foreground transition-colors">Cas concret</a>
                  )}
                  {uc.beforeAfter && (
                    <a href="#avant-apres" className="block text-muted-foreground hover:text-foreground transition-colors">Avant / Après</a>
                  )}
                  <a href="#tutoriel" className="block text-muted-foreground hover:text-foreground transition-colors">
                    {uc.n8nTutorial ? "Tutoriel n8n" : "Tutoriel"}
                  </a>
                  {!uc.n8nTutorial && uc.n8nWorkflow && (
                    <a href="#workflow-n8n" className="block text-muted-foreground hover:text-foreground transition-colors">Workflow n8n</a>
                  )}
                </nav>
              </CardContent>
            </Card>

            <Card className="border-primary/30 bg-primary/5">
              <CardContent className="pt-6 space-y-3">
                <p className="text-sm font-semibold">On le configure pour vous ?</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Workflow connecté à vos outils, vos règles métier, vos SLA. Prêt en 48h.
                </p>
                <Link
                  href="/demande"
                  className="block w-full rounded-lg bg-primary px-4 py-2.5 text-center text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Demander la configuration
                </Link>
                <Link
                  href="/pricing"
                  className="block text-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  Voir les plans &rarr;
                </Link>
              </CardContent>
            </Card>
          </div>
        </aside>
      </div>

      {/* Cross-links: ROI + Comparisons */}
      <div className="mt-12 grid gap-4 sm:grid-cols-2">
        <Link
          href="/calculateur-roi"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Calculez le ROI de cet agent
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Estimez les gains de temps et d&apos;argent avec notre calculateur gratuit.
          </p>
        </Link>
        <Link
          href="/comparatif/agent-ia-vs-chatbot"
          className="group rounded-xl border p-4 sm:p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Agent IA vs Chatbot : le comparatif
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Comprenez les différences et choisissez la bonne solution.
          </p>
        </Link>
      </div>

      {/* Related guides */}
      {(() => {
        const relatedGuides = guides.filter((g) =>
          g.relatedUseCases.includes(uc.slug)
        ).slice(0, 3);
        if (relatedGuides.length === 0) return null;
        return (
          <div className="mt-12">
            <h2 className="text-lg font-semibold mb-4">Guides connexes</h2>
            <div className="grid gap-3 sm:grid-cols-3">
              {relatedGuides.map((g) => (
                <Link
                  key={g.slug}
                  href={`/guide/${g.slug}`}
                  className="group rounded-xl border p-4 transition-all hover:shadow-sm hover:border-primary/30"
                >
                  <Badge variant="secondary" className="mb-2 text-[10px]">
                    {g.category} · {g.readTime}
                  </Badge>
                  <h3 className="text-sm font-medium group-hover:text-primary transition-colors line-clamp-2">
                    {g.title}
                  </h3>
                </Link>
              ))}
            </div>
          </div>
        );
      })()}

      {/* Newsletter CTA */}
      <div className="mt-8">
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
