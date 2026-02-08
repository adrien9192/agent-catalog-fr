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

const WORKFLOW_COUNT = useCases.filter((u) => u.n8nTutorial && u.n8nTutorial.length > 0).length || useCases.length;

export default async function UseCasePage({ params }: PageProps) {
  const { slug } = await params;
  const uc = useCases.find((u) => u.slug === slug);
  if (!uc) notFound();

  const midTutorialIndex = uc.n8nTutorial ? Math.floor(uc.n8nTutorial.length / 2) : 0;

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

      {/* Fast-track banner for "pressés" */}
      {uc.n8nTutorial && (
        <div className="mb-8 rounded-xl border border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10 p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-sm">Pas le temps de configurer ? Obtenez ce workflow prêt à importer.</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Import en 1 clic dans n8n + accès aux {WORKFLOW_COUNT}+ workflows du catalogue.
              <span className="font-medium text-foreground"> Essai gratuit 14 jours.</span>
            </p>
          </div>
          <Link
            href="/pricing"
            className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
          >
            Obtenir le workflow Pro
          </Link>
        </div>
      )}

      {/* Header */}
      <header className="mb-10">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          <DifficultyBadge difficulty={uc.difficulty} />
          {uc.estimatedTime && (
            <Badge variant="outline" className="text-xs">
              {uc.estimatedTime}
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
          <a href="#obtenir" className="rounded-full border border-primary/30 bg-primary/5 px-3 py-1 text-xs text-primary font-medium hover:bg-primary/10 transition-colors">
            Obtenir le workflow
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

          {/* 3. Tutorial */}
          <section id="tutoriel">
            {uc.n8nTutorial && uc.n8nTutorial.length > 0 ? (
              <>
                <div className="mb-6">
                  <h2 className="text-2xl font-bold mb-2">Tutoriel n8n — pas à pas</h2>
                  <p className="text-sm text-muted-foreground">
                    Suivez ce guide noeud par noeud pour construire le workflow dans n8n.
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
                          {i > 0 && <span className="text-muted-foreground text-xs">&rarr;</span>}
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
                    <div key={i}>
                      <div className="rounded-xl border overflow-hidden">
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

                          {/* Variants or single configuration */}
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

                      {/* Mid-tutorial CTA — inserted after the middle node */}
                      {i === midTutorialIndex && uc.n8nTutorial && uc.n8nTutorial.length > 3 && (
                        <div className="my-6 rounded-xl border-2 border-dashed border-primary/30 bg-gradient-to-r from-primary/5 to-transparent p-5 sm:p-6">
                          <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
                            <div className="flex-1">
                              <p className="font-semibold text-sm">Ce workflow existe en version prête à importer</p>
                              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                                Importez-le en 1 clic dans n8n au lieu de configurer {uc.n8nTutorial.length} noeuds manuellement.
                                Il ne reste qu&apos;à brancher vos clés API.
                              </p>
                            </div>
                            <Link
                              href="/pricing"
                              className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                            >
                              Essai gratuit 14 jours
                            </Link>
                          </div>
                        </div>
                      )}
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

          {/* Blurred JSON preview */}
          {uc.n8nTutorial && (
            <div className="rounded-xl border overflow-hidden">
              <div className="border-b bg-muted/40 px-4 py-3 sm:px-6 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm">&#128274;</span>
                  <h3 className="font-semibold text-sm">Workflow n8n — fichier JSON importable</h3>
                </div>
                <Badge variant="secondary" className="text-[10px]">Pro</Badge>
              </div>
              <div className="relative p-4 sm:p-6">
                <div className="overflow-hidden rounded-lg border bg-[#1e1e2e] p-4 max-h-32 select-none" style={{ filter: "blur(3px)", WebkitUserSelect: "none" }}>
                  <pre className="text-xs font-mono text-[#cdd6f4] whitespace-pre-wrap" aria-hidden="true">
{`{
  "name": "${uc.title} — Workflow n8n",
  "nodes": [
    { "type": "n8n-nodes-base.webhook", "position": [250, 300], "parameters": { "path": "triage-ticket", "httpMethod": "POST" } },
    { "type": "n8n-nodes-base.httpRequest", "position": [500, 300], "parameters": { "url": "https://api.openai.com/v1/..." } },
    { "type": "n8n-nodes-base.switch", "position": [750, 300], "parameters": { "rules": [...] } },
    ...
  ],
  "connections": { ... }
}`}
                  </pre>
                </div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <Link
                    href="/pricing"
                    className="rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors shadow-lg"
                  >
                    Débloquer le workflow JSON &rarr;
                  </Link>
                </div>
              </div>
            </div>
          )}

          {/* DIY vs Pro comparison table */}
          <section id="obtenir">
            <h2 className="text-2xl font-bold mb-4">Construire soi-même ou gagner du temps ?</h2>
            <div className="overflow-x-auto rounded-xl border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b bg-muted/30">
                    <th className="text-left p-3 sm:p-4 font-medium text-muted-foreground w-1/3"></th>
                    <th className="text-center p-3 sm:p-4 font-semibold">Gratuit (DIY)</th>
                    <th className="text-center p-3 sm:p-4 font-semibold text-primary bg-primary/5">Pro — 29&euro;/mois</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Tutoriel pas à pas</td>
                    <td className="p-3 sm:p-4 text-center">&#10003;</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5">&#10003;</td>
                  </tr>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Workflow JSON importable</td>
                    <td className="p-3 sm:p-4 text-center text-muted-foreground">&#10007;</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">Import en 1 clic</td>
                  </tr>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Temps de mise en place</td>
                    <td className="p-3 sm:p-4 text-center">~{uc.estimatedTime || "2-4h"}</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">~5 min</td>
                  </tr>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Templates n8n exportables</td>
                    <td className="p-3 sm:p-4 text-center text-muted-foreground">&#10007;</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">&#10003;</td>
                  </tr>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Accès au catalogue complet</td>
                    <td className="p-3 sm:p-4 text-center">Tutoriels uniquement</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">{WORKFLOW_COUNT}+ workflows</td>
                  </tr>
                  <tr className="border-b">
                    <td className="p-3 sm:p-4 text-muted-foreground">Nouveaux workflows chaque mois</td>
                    <td className="p-3 sm:p-4 text-center text-muted-foreground">&#10007;</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">Accès anticipé</td>
                  </tr>
                  <tr>
                    <td className="p-3 sm:p-4 text-muted-foreground">Guide personnalisé</td>
                    <td className="p-3 sm:p-4 text-center text-muted-foreground">&#10007;</td>
                    <td className="p-3 sm:p-4 text-center bg-primary/5 font-medium">1 par mois</td>
                  </tr>
                </tbody>
              </table>
            </div>

            {/* ROI anchored to price */}
            <div className="mt-4 rounded-lg border bg-muted/30 p-4 flex flex-col sm:flex-row items-start sm:items-center gap-3">
              <div className="flex-1">
                <p className="text-sm font-medium">{uc.roiIndicatif}</p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  L&apos;abonnement Pro coûte 29&euro;/mois — soit moins de 1&euro; par jour pour automatiser ce processus.
                </p>
              </div>
              <Link
                href="/pricing"
                className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Essai gratuit 14 jours
              </Link>
            </div>
          </section>

          {/* Final contextual CTA */}
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <p className="font-semibold">Vous avez lu le tutoriel — il ne reste plus qu&apos;à le brancher</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Obtenez le workflow JSON prêt à importer + accès à tous les workflows du catalogue.
                  Sans engagement. Annulation en un clic.
                </p>
              </div>
              <div className="flex flex-col gap-2">
                <Link
                  href="/pricing"
                  className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-center text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Commencer l&apos;essai gratuit
                </Link>
                <Link
                  href="/pricing"
                  className="text-center text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  Comparer tous les plans &rarr;
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
                          {i > 0 && <span className="text-muted-foreground text-xs">&rarr;</span>}
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
                    <span className="text-muted-foreground">DIY</span>
                    <span>{uc.estimatedTime}</span>
                  </div>
                )}
                {uc.n8nTutorial && (
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Avec Pro</span>
                    <span className="font-medium text-primary">~5 min</span>
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
                  <a href="#obtenir" className="block text-primary font-medium hover:text-primary/80 transition-colors">
                    Obtenir le workflow &rarr;
                  </a>
                </nav>
              </CardContent>
            </Card>

            {/* Sidebar Pro CTA */}
            <Card className="border-primary/30 bg-primary/5">
              <CardContent className="pt-6 space-y-3">
                <div className="flex items-baseline justify-between">
                  <p className="text-sm font-semibold">Plan Pro</p>
                  <p className="text-lg font-bold">29&euro;<span className="text-xs font-normal text-muted-foreground">/mois</span></p>
                </div>
                <ul className="space-y-1.5 text-xs text-muted-foreground">
                  <li className="flex items-center gap-1.5"><span className="text-primary">&#10003;</span> Ce workflow prêt à importer</li>
                  <li className="flex items-center gap-1.5"><span className="text-primary">&#10003;</span> {WORKFLOW_COUNT}+ workflows inclus</li>
                  <li className="flex items-center gap-1.5"><span className="text-primary">&#10003;</span> Templates n8n exportables</li>
                  <li className="flex items-center gap-1.5"><span className="text-primary">&#10003;</span> Nouveaux workflows chaque mois</li>
                </ul>
                <Link
                  href="/pricing"
                  className="block w-full rounded-lg bg-primary px-4 py-2.5 text-center text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Essai gratuit 14 jours
                </Link>
                <p className="text-center text-[10px] text-muted-foreground">Sans carte bancaire. Annulation en un clic.</p>
              </CardContent>
            </Card>

            {/* Email capture for not-ready visitors */}
            <Card>
              <CardContent className="pt-6 space-y-2">
                <p className="text-sm font-semibold">Pas encore prêt ?</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Recevez ce workflow + nos meilleurs tutos IA chaque semaine. Gratuit.
                </p>
                <NewsletterSignup variant="compact" />
              </CardContent>
            </Card>
          </div>
        </aside>
      </div>

      {/* Cross-links */}
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
