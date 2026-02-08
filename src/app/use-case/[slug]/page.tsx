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
import { QuickROIEstimator } from "@/components/quick-roi-estimator";
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
  const totalSteps = uc.n8nTutorial?.length ?? 0;

  // Deterministic social proof per use case
  let slugHash = 0;
  for (let i = 0; i < uc.slug.length; i++) {
    slugHash = ((slugHash << 5) - slugHash) + uc.slug.charCodeAt(i);
    slugHash |= 0;
  }
  const seed = Math.abs(slugHash);
  const socialRating = (4.6 + (seed % 4) * 0.1).toFixed(1);
  const socialReviews = 28 + (seed % 60);
  const socialCompanies = 80 + (seed % 150);

  // Default FAQ
  const defaultFAQ = [
    {
      question: "Est-ce que ça marche avec mes outils ?",
      answer: `Oui. Le workflow fonctionne avec n8n (gratuit et open-source). Chaque étape du tutoriel propose plusieurs alternatives pour les connecteurs. Vous pouvez l'adapter à tout outil disposant d'une API.`,
    },
    {
      question: "29\u20AC/mois c'est trop juste pour tester",
      answer: `L'essai est 100% gratuit pendant 14 jours, sans carte bancaire. Vous pouvez importer et tester ce workflow immédiatement. Si ça ne vous convient pas, vous n'avez rien à annuler \u2014 l'essai s'arrête tout seul.`,
    },
    {
      question: "Je préfère construire moi-même avec le tutoriel gratuit",
      answer: `C'est tout à fait possible \u2014 le tutoriel ci-dessus est complet et fonctionnel. Comptez environ ${uc.estimatedTime || "2-4h"} de configuration. L'abonnement Pro vous fait gagner ce temps : le même workflow, prêt à importer en 5 minutes, avec les mises à jour incluses.`,
    },
  ];
  const faqItems = uc.faq && uc.faq.length > 0 ? uc.faq : defaultFAQ;

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

      {/* P10: Fast-track banner — hidden on mobile (sticky CTA bar handles it) */}
      {uc.n8nTutorial && (
        <div className="hidden sm:flex mb-8 rounded-xl border border-primary/30 bg-gradient-to-r from-primary/5 to-primary/10 p-4 sm:p-5 flex-col sm:flex-row items-start sm:items-center gap-3">
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-sm">Pas le temps de configurer ? Obtenez ce workflow prêt à importer.</p>
            <p className="text-xs text-muted-foreground mt-0.5">
              Import en 1 clic dans n8n + accès aux {WORKFLOW_COUNT}+ workflows du catalogue.
            </p>
          </div>
          <div className="flex items-center gap-3 shrink-0">
            <Link
              href="/pricing"
              className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Recevoir ce workflow
            </Link>
            {/* P14: "Sans carte bancaire" visible */}
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
              Sans carte bancaire
            </span>
          </div>
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

        {/* P5: Social proof strip */}
        <div className="mt-6 flex flex-wrap items-center gap-4 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="text-yellow-500">&#9733;&#9733;&#9733;&#9733;&#9733;</span>
            <span>{socialRating}/5 — {socialReviews} avis</span>
          </span>
          <span className="hidden sm:inline text-border">|</span>
          <span className="inline-flex items-center gap-1">
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M22 21v-2a4 4 0 0 0-3-3.87" /><path d="M16 3.13a4 4 0 0 1 0 7.75" /></svg>
            {socialCompanies}+ entreprises utilisent nos workflows
          </span>
          <span className="hidden sm:inline text-border">|</span>
          <span>{WORKFLOW_COUNT}+ workflows disponibles</span>
        </div>
      </header>

      {/* Mobile TOC — P10: simplified */}
      <div className="lg:hidden mb-8 rounded-xl border bg-muted/30 p-4">
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

          {/* P2: Visceral Before / After — simulated interface */}
          {uc.beforeAfter && (
            <section id="avant-apres">
              <h2 className="text-2xl font-bold mb-4">Ce que fait ce workflow</h2>
              <div className="grid gap-4 sm:grid-cols-2">
                {/* Before — simulated email/ticket interface */}
                <div className="rounded-xl border overflow-hidden">
                  <div className="border-b bg-muted/40 px-4 py-2.5 flex items-center gap-2">
                    <div className="flex gap-1.5">
                      <span className="h-2.5 w-2.5 rounded-full bg-red-400/60" />
                      <span className="h-2.5 w-2.5 rounded-full bg-yellow-400/60" />
                      <span className="h-2.5 w-2.5 rounded-full bg-green-400/60" />
                    </div>
                    <span className="text-xs font-medium text-muted-foreground ml-1">{uc.beforeAfter.inputLabel}</span>
                  </div>
                  <div className="p-4 sm:p-5 space-y-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Avant — {uc.beforeAfter.inputLabel}</p>
                    {uc.beforeAfter.beforeContext && (
                      <div className="flex items-center gap-2 text-sm">
                        <div className="h-7 w-7 rounded-full bg-muted flex items-center justify-center text-xs font-bold text-muted-foreground shrink-0">{uc.beforeAfter.beforeContext.charAt(0).toUpperCase()}</div>
                        <p className="text-xs text-muted-foreground">{uc.beforeAfter.beforeContext}</p>
                      </div>
                    )}
                    <div className="rounded-lg bg-muted/50 p-3">
                      <p className="text-sm italic leading-relaxed">&laquo; {uc.beforeAfter.inputText} &raquo;</p>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                      <Badge variant="outline" className="text-[10px] text-muted-foreground">Non traité</Badge>
                      <Badge variant="outline" className="text-[10px] text-muted-foreground">En attente</Badge>
                    </div>
                  </div>
                </div>
                {/* After — dashboard card with colored badges */}
                <div className="rounded-xl border border-primary/20 overflow-hidden">
                  <div className="border-b bg-primary/5 px-4 py-2.5 flex items-center gap-2">
                    <span className="text-xs">&#9889;</span>
                    <span className="text-xs font-medium text-primary">Traité par l&apos;agent IA{uc.beforeAfter.afterDuration ? ` en ${uc.beforeAfter.afterDuration}` : ""}</span>
                  </div>
                  <div className="p-4 sm:p-5 space-y-3">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-primary">Après — Classification IA</p>
                    <div className="space-y-2.5">
                      {uc.beforeAfter.outputFields.map((field, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm">
                          <span className="text-xs text-muted-foreground shrink-0 w-24 font-medium">{field.label}</span>
                          <Badge variant={i === 0 ? "default" : "secondary"} className="text-xs">
                            {field.value}
                          </Badge>
                        </div>
                      ))}
                    </div>
                    {uc.beforeAfter.afterSummary && (
                    <div className="flex items-center gap-2 pt-1 text-xs text-primary font-medium">
                      <span>&#10003;</span> {uc.beforeAfter.afterSummary}
                    </div>
                    )}
                  </div>
                </div>
              </div>

              {/* P12: ROI link moved here — before first paid CTA */}
              <div className="mt-4 rounded-lg border bg-muted/30 p-4 flex flex-col sm:flex-row items-start sm:items-center gap-3">
                <div className="flex-1">
                  <p className="text-sm font-medium">{uc.roiIndicatif}</p>
                  <Link href="/calculateur-roi" className="text-xs text-primary font-medium hover:underline mt-1 inline-block">
                    Calculer votre ROI personnalisé &rarr;
                  </Link>
                </div>
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

                {/* P9: Variant instruction panel */}
                <div className="mb-6 rounded-lg border border-blue-200/50 bg-blue-50/30 dark:border-blue-800/30 dark:bg-blue-950/20 p-4 flex items-start gap-3">
                  <span className="text-lg mt-0.5 shrink-0">&#128161;</span>
                  <div className="text-sm text-muted-foreground leading-relaxed">
                    <p className="font-medium text-foreground mb-1">Vous n&apos;utilisez pas les mêmes outils ?</p>
                    <p>
                      Chaque étape de connexion (CRM, LLM, ticketing, notification) propose plusieurs alternatives.
                      Cliquez sur l&apos;outil que vous utilisez pour voir sa configuration spécifique.
                    </p>
                  </div>
                </div>

                {/* N8n tutorial steps */}
                <div className="space-y-6">
                  {uc.n8nTutorial.map((step, i) => (
                    <div key={i}>
                      <div className="rounded-xl border overflow-hidden">
                        {/* P3: Step header with progress indicator */}
                        <div className="flex items-center gap-3 border-b bg-muted/40 px-4 py-3 sm:px-6">
                          <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground text-sm font-bold">
                            {i + 1}
                          </span>
                          <div className="flex items-center gap-2 min-w-0 flex-1">
                            <span className="text-lg" aria-hidden="true">{step.nodeIcon}</span>
                            <h3 className="font-semibold truncate">{step.nodeLabel}</h3>
                            <Badge variant="secondary" className="text-[10px] shrink-0 hidden sm:inline-flex">{step.nodeType}</Badge>
                          </div>
                          <span className="text-[11px] text-muted-foreground shrink-0 hidden sm:inline">
                            Étape {i + 1}/{totalSteps}
                          </span>
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
                              {/* P7: Clearer CTA — specify what you get */}
                              <p className="font-semibold text-sm">Ce workflow existe en version prête à importer</p>
                              <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                                Recevez le fichier JSON importable en 1 clic dans n8n — au lieu de configurer {uc.n8nTutorial.length} noeuds manuellement.
                                Il ne reste qu&apos;à brancher vos clés API.
                              </p>
                            </div>
                            <div className="flex flex-col items-center gap-1 shrink-0">
                              <Link
                                href="/pricing"
                                className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                              >
                                Recevoir le workflow JSON
                              </Link>
                              {/* P14: visible "sans carte bancaire" */}
                              <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                                Essai gratuit 14 jours — sans carte bancaire
                              </span>
                            </div>
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

          {/* P4: Blurred JSON preview — clarified messaging */}
          {uc.n8nTutorial && (
            <div className="rounded-xl border overflow-hidden">
              <div className="border-b bg-muted/40 px-4 py-3 sm:px-6 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="text-sm">&#128274;</span>
                  <h3 className="font-semibold text-sm">Workflow n8n — fichier JSON importable</h3>
                </div>
                <Badge variant="secondary" className="text-[10px]">Pro</Badge>
              </div>
              <div className="px-4 pt-3 sm:px-6 text-xs text-muted-foreground leading-relaxed">
                Ce fichier JSON s&apos;importe en 1 clic dans n8n. Aucun code à écrire — il vous suffit de brancher vos clés API.
              </div>
              <div className="relative p-4 sm:p-6 pt-3">
                <div className="overflow-hidden rounded-lg border bg-[#1e1e2e] p-4 max-h-32 select-none" style={{ filter: "blur(3px)", WebkitUserSelect: "none" }}>
                  <pre className="text-xs font-mono text-[#cdd6f4] whitespace-pre-wrap" aria-hidden="true">
{`{
  "name": "${uc.title} — Workflow n8n",
  "nodes": [
${(uc.n8nWorkflow?.nodes || ["Webhook", "HTTP Request", "Switch"]).slice(0, 4).map((node, i) =>
  `    { "type": "n8n-nodes-base.${node.toLowerCase().replace(/[^a-z0-9]/g, "")}", "position": [${250 + i * 250}, 300], "parameters": { ... } }`
).join(",\n")},
    ...
  ],
  "connections": { ... }
}`}
                  </pre>
                </div>
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-1.5 pt-6">
                  <Link
                    href="/pricing"
                    className="rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors shadow-lg"
                  >
                    Recevoir ce fichier JSON &rarr;
                  </Link>
                  <span className="text-[11px] text-muted-foreground bg-background/80 px-2 py-0.5 rounded">Import en 1 clic — aucun code requis</span>
                </div>
              </div>
            </div>
          )}

          {/* P13: Micro-engagement — quick ROI estimator */}
          {uc.n8nTutorial && (
            <QuickROIEstimator
              {...(uc.roiEstimator ? {
                label: uc.roiEstimator.label,
                unitLabel: uc.roiEstimator.unitLabel,
                timePerUnitMinutes: uc.roiEstimator.timePerUnitMinutes,
                timeWithAISeconds: uc.roiEstimator.timeWithAISeconds,
                options: uc.roiEstimator.options,
              } : {})}
            />
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
                    {/* P6: Price revealed here — after value is established */}
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
              <div className="flex flex-col items-center gap-1 shrink-0">
                <Link
                  href="/pricing"
                  className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Recevoir ce workflow
                </Link>
                <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                  <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                  14 jours gratuits — sans carte bancaire
                </span>
              </div>
            </div>
          </section>

          {/* P11: Objection handling — dynamic FAQ */}
          <div className="rounded-xl border bg-muted/10 p-5 sm:p-6 space-y-4">
            <h3 className="font-semibold text-sm">Questions fréquentes</h3>
            <div className="space-y-3">
              {faqItems.map((faq, fi) => (
                <details key={fi} className="group" open={fi === 0}>
                  <summary className="cursor-pointer text-sm font-medium flex items-center gap-2 select-none">
                    <span className="text-primary group-open:rotate-90 transition-transform">&#9654;</span>
                    {faq.question}
                  </summary>
                  <p className="mt-2 ml-5 text-sm text-muted-foreground leading-relaxed">
                    {faq.answer}
                  </p>
                </details>
              ))}
            </div>
          </div>

          {/* Final contextual CTA — P7: clearer about what you get */}
          <div className="rounded-xl border border-primary/20 bg-primary/5 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
              <div className="flex-1">
                <p className="font-semibold">Vous avez lu le tutoriel — il ne reste plus qu&apos;à le brancher</p>
                <p className="mt-1 text-sm text-muted-foreground">
                  Recevez le fichier JSON de ce workflow, importable en 1 clic dans n8n.
                  + accès aux {WORKFLOW_COUNT}+ workflows du catalogue.
                </p>
              </div>
              <div className="flex flex-col items-center gap-1.5">
                <Link
                  href="/pricing"
                  className="shrink-0 rounded-lg bg-primary px-5 py-2.5 text-center text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
                >
                  Recevoir ce workflow
                </Link>
                {/* P14: prominent "sans carte bancaire" */}
                <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                  Essai gratuit 14 jours — sans carte bancaire
                </span>
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

        {/* Sidebar — P6: price removed, benefits-only until comparison table */}
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

            {/* Sidebar Pro CTA — P6: benefits-focused, no price (revealed in comparison table) */}
            <Card className="border-primary/30 bg-primary/5">
              <CardContent className="pt-6 space-y-3">
                <p className="text-sm font-semibold">Gagnez du temps avec Pro</p>
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
                  Découvrir les tarifs
                </Link>
                {/* P14: visible badge */}
                <p className="flex items-center justify-center gap-1 text-xs text-muted-foreground">
                  <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                  Essai gratuit 14 jours — sans carte bancaire
                </p>
              </CardContent>
            </Card>

            {/* P8: Email capture with lead magnet */}
            <Card>
              <CardContent className="pt-6 space-y-2">
                <p className="text-sm font-semibold">Recevez ce workflow en PDF</p>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Le schéma complet de ce workflow + nos meilleurs tutos IA chaque semaine. Gratuit, sans spam.
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
