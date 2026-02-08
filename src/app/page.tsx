import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { PromptBar } from "@/components/prompt-bar";
import { UseCaseCard } from "@/components/use-case-card";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { ExitIntentPopup } from "@/components/exit-intent-popup";
import { useCases } from "@/data/use-cases";
import { sectors } from "@/data/sectors";
import { guides } from "@/data/guides";
import { comparisons } from "@/data/comparisons";

const functions = [
  { label: "Support client", icon: "🎧", param: "Support" },
  { label: "Sales & CRM", icon: "📈", param: "Sales" },
  { label: "Ressources humaines", icon: "👥", param: "RH" },
  { label: "Marketing", icon: "📣", param: "Marketing" },
  { label: "Finance", icon: "💰", param: "Finance" },
  { label: "IT & DevOps", icon: "🔧", param: "IT" },
  { label: "Supply Chain", icon: "📦", param: "Supply Chain" },
];

const toolLogos = [
  { name: "n8n", domain: "n8n.io" },
  { name: "OpenAI", domain: "openai.com" },
  { name: "Anthropic", domain: "anthropic.com" },
  { name: "Mistral", domain: "mistral.ai" },
  { name: "Slack", domain: "slack.com" },
  { name: "HubSpot", domain: "hubspot.com" },
  { name: "Zendesk", domain: "zendesk.com" },
  { name: "Salesforce", domain: "salesforce.com" },
  { name: "Notion", domain: "notion.so" },
  { name: "Google", domain: "google.com" },
];

const WORKFLOW_COUNT = useCases.length;
const GUIDE_COUNT = guides.length;

const stats = [
  { value: `${WORKFLOW_COUNT}+`, label: "workflows documentés" },
  { value: "60%", label: "de temps gagné en moyenne", sub: "sur les tâches automatisées" },
  { value: `${GUIDE_COUNT}`, label: "guides pratiques" },
  { value: "100%", label: "tutoriels gratuits", sub: "sans limite d'accès" },
];

const testimonials = [
  {
    quote: "On a déployé l'agent de triage support en 3 jours. 40% de tickets résolus automatiquement dès la première semaine.",
    name: "Marie L.",
    role: "Head of Ops",
    company: "Scale-up SaaS B2B",
    metric: "40% tickets auto-résolus",
  },
  {
    quote: "Le workflow de qualification leads nous a permis de doubler le taux de conversion MQL → SQL sans recruter.",
    name: "Thomas R.",
    role: "Directeur Commercial",
    company: "ESN, 200 collaborateurs",
    metric: "2x taux de conversion",
  },
  {
    quote: "Les tutoriels sont incroyablement détaillés. Même notre équipe sans background data a pu implémenter en autonomie.",
    name: "Sophie M.",
    role: "CTO",
    company: "Fintech Paris",
    metric: "Implémentation en 5 jours",
  },
];

const steps = [
  {
    step: "1",
    title: "Décrivez votre problème",
    description: "Parcourez le catalogue par fonction ou décrivez votre besoin. Chaque workflow est classé par secteur, difficulté et ROI estimé.",
    link: "/catalogue",
    linkLabel: "Parcourir le catalogue",
  },
  {
    step: "2",
    title: "Obtenez un workflow n8n clé en main",
    description: "Tutoriel pas-à-pas avec alternatives pour chaque outil (CRM, LLM, notification). Import JSON en 1 clic ou suivez le guide noeud par noeud.",
    link: "/use-case/agent-triage-support-client",
    linkLabel: "Voir un exemple",
  },
  {
    step: "3",
    title: "Déployez en 5 minutes",
    description: "Importez le workflow dans n8n, branchez vos clés API, testez. Estimation de ROI et bonnes pratiques enterprise incluses.",
    link: "/calculateur-roi",
    linkLabel: "Calculer mon ROI",
  },
];

// Sector use-case counts
function getSectorCount(sectorSlug: string): number {
  return useCases.filter((uc) =>
    uc.sectors.some((s) => s.toLowerCase().replace(/\s+/g, "-") === sectorSlug)
  ).length;
}

// Featured: first 6
const featured = useCases.slice(0, 6);
const featuredSlugs = new Set(featured.map((uc) => uc.slug));

// Recently added: deduped from featured
const recentlyAdded = useCases
  .filter((uc) => !featuredSlugs.has(uc.slug))
  .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
  .slice(0, 3);

function HomePageJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: "AgentCatalog",
    url: "https://agent-catalog-fr.vercel.app",
    applicationCategory: "BusinessApplication",
    operatingSystem: "Web browser",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "EUR",
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function HomePage() {
  return (
    <>
      <HomePageJsonLd />

      {/* Hero */}
      <section className="dotted-grid relative overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <Badge variant="secondary" className="mb-4 text-xs">
              {WORKFLOW_COUNT} workflows prêts à l&apos;emploi
            </Badge>
            <h1 className="text-3xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Automatisez les tâches répétitives{" "}
              <span className="gradient-text">de votre équipe</span>
              <br className="hidden sm:block" />
              {" "}grâce à l&apos;IA
            </h1>
            <p className="mt-4 text-base text-muted-foreground sm:text-xl max-w-2xl mx-auto leading-relaxed">
              Gagnez 10h/semaine en automatisant support, ventes, RH et finance.
              Workflows n8n importables en 1 clic, tutoriels pas-à-pas. Gratuit.
            </p>

            {/* CTA pills */}
            <div className="mt-8 flex flex-wrap justify-center gap-2">
              {functions.map((fn) => (
                <Link key={fn.param} href={`/catalogue?fn=${fn.param}`}>
                  <Badge
                    variant="outline"
                    className="cursor-pointer px-3 py-1.5 text-sm transition-colors hover:bg-primary hover:text-primary-foreground"
                  >
                    <span className="mr-1">{fn.icon}</span> {fn.label}
                  </Badge>
                </Link>
              ))}
            </div>

            {/* Prompt bar */}
            <div className="mt-10">
              <PromptBar />
            </div>

            {/* Mini workflow visual */}
            <div className="mt-10 mx-auto max-w-xl">
              <div className="flex items-center justify-center gap-1.5 sm:gap-3">
                {[
                  { icon: "📥", label: "Webhook" },
                  { icon: "🤖", label: "LLM" },
                  { icon: "🔀", label: "Routage" },
                  { icon: "✅", label: "Action", primary: true },
                ].map((node, i) => (
                  <div key={node.label} className="flex items-center gap-1.5 sm:gap-3">
                    {i > 0 && <span className="text-muted-foreground text-xs sm:text-sm">&rarr;</span>}
                    <div className={`rounded-lg border px-2.5 py-1.5 sm:px-3 sm:py-2 text-[11px] sm:text-xs font-medium shadow-sm ${
                      node.primary
                        ? "border-primary/30 bg-primary/5 text-primary"
                        : "bg-card"
                    }`}>
                      <span className="mr-1">{node.icon}</span>{node.label}
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-2 text-[11px] text-muted-foreground">
                Chaque workflow = un fichier JSON importable dans n8n en 1 clic
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Tool logos */}
      <section className="border-y bg-muted/10 overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
          <p className="text-center text-xs text-muted-foreground mb-4">
            Compatible avec vos outils
          </p>
          <div className="flex flex-wrap items-center justify-center gap-6 sm:gap-8">
            {toolLogos.map((tool) => (
              <div key={tool.name} className="flex items-center gap-2 opacity-60 hover:opacity-100 transition-opacity">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={`https://logo.clearbit.com/${tool.domain}`}
                  alt={tool.name}
                  width={20}
                  height={20}
                  className="rounded-sm"
                  loading="lazy"
                />
                <span className="text-xs font-medium text-muted-foreground hidden sm:inline">{tool.name}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Social proof / stats */}
      <section className="bg-muted/20">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 gap-4 sm:gap-6 sm:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-3xl font-bold text-primary">{stat.value}</p>
                <p className="mt-1 text-sm text-muted-foreground">{stat.label}</p>
                {stat.sub && (
                  <p className="text-[11px] text-muted-foreground/70">{stat.sub}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust signals — "marché français" promoted */}
      <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3 text-xs sm:text-sm text-muted-foreground">
          <div className="flex items-center gap-2 font-medium text-foreground">
            <span className="text-base">🇫🇷</span>
            <span>Conçu pour le marché français</span>
          </div>
          <span className="hidden sm:inline text-border">|</span>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            <span>Conforme RGPD</span>
          </div>
          <span className="hidden sm:inline text-border">|</span>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <span>Tutoriels open-source</span>
          </div>
          <span className="hidden sm:inline text-border">|</span>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            <span>Nouveaux workflows chaque semaine</span>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h2 className="text-2xl font-bold sm:text-3xl">
            De l&apos;idée au déploiement en 3 étapes
          </h2>
          <p className="mt-2 text-muted-foreground max-w-xl mx-auto">
            Chaque workflow est conçu pour être opérationnel rapidement, sans équipe data dédiée.
          </p>
        </div>
        <div className="grid gap-6 sm:grid-cols-3">
          {steps.map((s) => (
            <Card key={s.step} className="relative overflow-hidden">
              <CardContent className="pt-6">
                <span className="absolute -top-2 -right-2 text-7xl font-black text-primary/5">
                  {s.step}
                </span>
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-primary text-primary-foreground font-bold text-sm mb-4">
                  {s.step}
                </div>
                <h3 className="font-semibold text-lg mb-2">{s.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed mb-3">{s.description}</p>
                <Link href={s.link} className="text-xs text-primary font-medium hover:underline">
                  {s.linkLabel} &rarr;
                </Link>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Featured use cases — BEFORE testimonials */}
      <section className="border-t">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="flex items-end justify-between mb-8">
            <div>
              <h2 className="text-2xl font-bold sm:text-3xl">
                Workflows les plus demandés
              </h2>
              <p className="mt-1 text-muted-foreground">
                Choisis par les équipes ops, support et sales des entreprises françaises.
              </p>
            </div>
            <Button variant="outline" size="sm" asChild className="hidden sm:inline-flex">
              <Link href="/catalogue">Voir les {WORKFLOW_COUNT} workflows</Link>
            </Button>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {featured.map((uc) => (
              <UseCaseCard key={uc.slug} useCase={uc} />
            ))}
          </div>

          <div className="mt-6 text-center sm:hidden">
            <Button variant="outline" asChild>
              <Link href="/catalogue">Voir tous les workflows</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Testimonials — AFTER seeing the product */}
      <section className="border-t bg-muted/30">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center mb-10">
            <Badge variant="secondary" className="mb-4 text-xs">
              Retours clients
            </Badge>
            <h2 className="text-2xl font-bold sm:text-3xl">
              Ils automatisent avec nos workflows
            </h2>
          </div>
          <div className="grid gap-6 sm:grid-cols-3">
            {testimonials.map((t) => (
              <div
                key={t.name}
                className="rounded-xl border bg-card p-5 sm:p-6 flex flex-col"
              >
                <div className="mb-3">
                  <Badge variant="outline" className="text-xs font-medium text-primary">
                    {t.metric}
                  </Badge>
                </div>
                <blockquote className="flex-1 text-sm text-muted-foreground leading-relaxed mb-4">
                  &ldquo;{t.quote}&rdquo;
                </blockquote>
                <div className="border-t pt-3">
                  <p className="text-sm font-semibold">{t.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {t.role}, {t.company}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing teaser — after proof of value */}
      <section className="border-t bg-primary/5">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center gap-6 sm:gap-8">
            <div className="flex-1 text-center sm:text-left">
              <h2 className="text-xl font-bold sm:text-2xl">
                À partir de 29&euro;/mois — soit moins de 1&euro;/jour
              </h2>
              <p className="mt-2 text-sm text-muted-foreground max-w-lg">
                Accédez à tous les workflows en JSON importable, les mises à jour, et les nouveaux workflows chaque semaine. Essai gratuit 14 jours, sans carte bancaire.
              </p>
            </div>
            <div className="flex flex-col items-center gap-1.5">
              <Button size="lg" asChild>
                <Link href="/pricing">Voir les tarifs</Link>
              </Button>
              <span className="inline-flex items-center gap-1 text-[11px] text-muted-foreground">
                <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
                Tutoriels gratuits — Pro pour les JSON importables
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* Sectors — with workflow counts */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <h2 className="text-2xl font-bold sm:text-3xl mb-2">
          Votre secteur, nos workflows
        </h2>
        <p className="text-muted-foreground mb-8">
          Des solutions adaptées à chaque industrie, de la banque au retail.
        </p>
        <div className="grid gap-2 sm:gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
          {sectors.slice(0, 8).map((sector) => {
            const count = getSectorCount(sector.slug);
            return (
              <Link
                key={sector.slug}
                href={`/secteur/${sector.slug}`}
                className="group rounded-xl border bg-card p-4 transition-all hover:shadow-sm hover:border-primary/30"
              >
                <span className="text-2xl">{sector.icon}</span>
                <h3 className="mt-2 font-semibold text-sm group-hover:text-primary transition-colors">
                  {sector.name}
                </h3>
                <p className="mt-1 text-xs text-muted-foreground">
                  {count} workflow{count > 1 ? "s" : ""} disponible{count > 1 ? "s" : ""}
                </p>
              </Link>
            );
          })}
        </div>
      </section>

      {/* ROI Calculator CTA — with proof */}
      <section className="border-t bg-primary/5">
        <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
          <div className="flex flex-col sm:flex-row items-center gap-6 sm:gap-8">
            <div className="flex-1 text-center sm:text-left">
              <h2 className="text-xl font-bold sm:text-2xl">
                Combien pourriez-vous économiser avec l&apos;IA ?
              </h2>
              <p className="mt-2 text-sm text-muted-foreground max-w-lg">
                En moyenne, un workflow de triage support économise 15h/semaine pour une équipe de 5 agents.
                Estimez votre propre gain avec notre calculateur gratuit.
              </p>
            </div>
            <Button size="lg" asChild>
              <Link href="/calculateur-roi">Calculer mon ROI</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Recently added — deduped from featured */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="flex items-end justify-between mb-8">
          <div>
            <Badge variant="secondary" className="mb-2 text-xs">
              Nouveautés
            </Badge>
            <h2 className="text-2xl font-bold sm:text-3xl">
              Derniers workflows ajoutés
            </h2>
          </div>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {recentlyAdded.map((uc) => (
            <UseCaseCard key={uc.slug} useCase={uc} />
          ))}
        </div>
      </section>

      {/* Guides section */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="flex items-end justify-between mb-8">
          <div>
            <Badge variant="secondary" className="mb-2 text-xs">
              Ressources gratuites
            </Badge>
            <h2 className="text-2xl font-bold sm:text-3xl">
              Guides pratiques IA
            </h2>
            <p className="mt-1 text-muted-foreground">
              Apprenez à déployer l&apos;IA dans votre entreprise, département par département.
            </p>
          </div>
          <Button variant="outline" size="sm" asChild className="hidden sm:inline-flex">
            <Link href="/guide">Tous les guides</Link>
          </Button>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {guides.slice(0, 3).map((g) => (
            <Link
              key={g.slug}
              href={`/guide/${g.slug}`}
              className="group rounded-xl border bg-card p-5 transition-all hover:shadow-sm hover:border-primary/30"
            >
              <div className="flex items-center gap-2 mb-3">
                <Badge variant="secondary" className="text-xs">{g.category}</Badge>
                <span className="text-xs text-muted-foreground">{g.readTime}</span>
              </div>
              <h3 className="font-semibold text-sm leading-snug group-hover:text-primary transition-colors">
                {g.title}
              </h3>
              <p className="mt-2 text-xs text-primary font-medium">
                Lire le guide &rarr;
              </p>
            </Link>
          ))}
        </div>
        <div className="mt-6 text-center sm:hidden">
          <Button variant="outline" asChild>
            <Link href="/guide">Voir tous les guides</Link>
          </Button>
        </div>
      </section>

      {/* Comparisons — 2-col grid for readability */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="text-center mb-8">
          <Badge variant="secondary" className="mb-2 text-xs">
            Guides de choix
          </Badge>
          <h2 className="text-2xl font-bold sm:text-3xl">
            Choisissez la bonne solution
          </h2>
          <p className="mt-2 text-muted-foreground max-w-lg mx-auto">
            Des comparatifs objectifs pour prendre les bonnes décisions technologiques.
          </p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 max-w-3xl mx-auto">
          {comparisons.slice(0, 4).map((c) => (
            <Link
              key={c.slug}
              href={`/comparatif/${c.slug}`}
              className="group rounded-xl border bg-card p-5 transition-all hover:shadow-sm hover:border-primary/30"
            >
              <div className="flex flex-wrap gap-1 mb-3">
                {c.options.map((o) => (
                  <Badge key={o.name} variant="outline" className="text-xs">
                    {o.name}
                  </Badge>
                ))}
              </div>
              <h3 className="font-semibold text-sm leading-snug group-hover:text-primary transition-colors">
                {c.title}
              </h3>
              <p className="mt-2 text-xs text-primary font-medium">
                Lire le comparatif &rarr;
              </p>
            </Link>
          ))}
        </div>
        <div className="mt-4 text-center">
          <Link href="/comparatif" className="text-sm text-primary font-medium hover:underline">
            Voir tous les comparatifs ({comparisons.length}) &rarr;
          </Link>
        </div>
      </section>

      {/* CTA: custom request — positive framing */}
      <section className="border-t bg-muted/30">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-bold sm:text-3xl">
              Besoin d&apos;un workflow sur mesure ?
            </h2>
            <p className="mt-3 text-muted-foreground max-w-lg mx-auto">
              Décrivez votre processus à automatiser et notre équipe développera
              un workflow adapté avec tutoriel complet et estimation de ROI.
            </p>
            <div className="mt-6">
              <Button size="lg" asChild>
                <Link href="/demande">Demander un workflow sur mesure</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ — collapsible with internal links */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl">
          <h2 className="text-2xl font-bold sm:text-3xl text-center mb-10">
            Questions fréquentes
          </h2>
          <div className="space-y-3">
            {[
              {
                q: "Qu'est-ce qu'un Agent IA ?",
                a: "Un Agent IA est un programme autonome qui exécute des tâches complexes en utilisant l'intelligence artificielle. Contrairement à un chatbot classique, il peut prendre des décisions, interagir avec vos outils (CRM, email, ERP) et s'améliorer au fil du temps.",
                link: "/comparatif/agent-ia-vs-chatbot",
                linkLabel: "Lire le comparatif Agent IA vs Chatbot",
              },
              {
                q: "Les workflows sont-ils vraiment gratuits ?",
                a: "Oui, tous les tutoriels (pas-à-pas, schémas d'architecture, estimation de ROI) sont accessibles gratuitement et sans limite. L'abonnement Pro (à partir de 29€/mois) ajoute les fichiers JSON importables en 1 clic et l'accès anticipé aux nouveaux workflows.",
                link: "/pricing",
                linkLabel: "Voir les tarifs",
              },
              {
                q: "Faut-il une équipe technique pour implémenter ?",
                a: "Pas nécessairement. Chaque workflow utilise n8n (plateforme no-code/low-code) avec un tutoriel noeud par noeud. Les alternatives Python sont aussi documentées pour les équipes techniques.",
                link: "/comparatif/no-code-vs-pro-code-ia",
                linkLabel: "No-code vs Pro-code : le comparatif",
              },
              {
                q: "Est-ce compatible avec mes outils existants ?",
                a: "Oui. Chaque étape du tutoriel propose plusieurs alternatives (Zendesk, Freshdesk, Crisp pour le support ; HubSpot, Pipedrive, Salesforce pour le CRM ; Slack, Teams, email pour les notifications). Les schémas d'architecture montrent comment connecter vos outils.",
              },
              {
                q: "Mes données sont-elles en sécurité ?",
                a: "n8n s'auto-héberge sur votre propre serveur — vos données ne transitent jamais par nos systèmes. Chaque workflow intègre les bonnes pratiques RGPD : anonymisation des données personnelles, audit trail et human-in-the-loop pour les décisions critiques.",
              },
              {
                q: "Quel LLM choisir : Claude, GPT-4 ou Mistral ?",
                a: "Claude excelle en analyse de documents et fiabilité, GPT-4 en génération de code et écosystème large, Mistral en souveraineté européenne et rapport qualité/prix. Chaque tutoriel propose les 4 alternatives (+ Ollama gratuit en local) avec la configuration spécifique.",
                link: "/comparatif/gpt4-vs-claude-vs-mistral",
                linkLabel: "Lire le comparatif GPT-4 vs Claude vs Mistral",
              },
              {
                q: "Combien de temps pour déployer un agent IA ?",
                a: "Avec le tutoriel pas-à-pas, un workflow se déploie en 2 à 4 heures via n8n. Avec l'abonnement Pro, le fichier JSON s'importe en 5 minutes — il ne reste qu'à brancher vos clés API.",
              },
              {
                q: "Quel ROI attendre d'un agent IA ?",
                a: "Le ROI varie selon le cas d'usage. Un workflow de triage support économise en moyenne 60% du temps de première réponse. Un workflow de qualification leads double le taux de conversion MQL→SQL.",
                link: "/calculateur-roi",
                linkLabel: "Calculer votre ROI personnalisé",
              },
            ].map((faq, i) => (
              <details key={faq.q} className="group rounded-xl border overflow-hidden" open={i === 0}>
                <summary className="cursor-pointer px-4 py-4 sm:px-5 font-semibold text-sm sm:text-base flex items-center gap-2 select-none hover:bg-muted/30 transition-colors">
                  <span className="text-primary group-open:rotate-90 transition-transform shrink-0">&#9654;</span>
                  {faq.q}
                </summary>
                <div className="px-4 pb-4 sm:px-5 sm:pb-5 ml-5">
                  <p className="text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
                  {faq.link && (
                    <Link href={faq.link} className="inline-block mt-2 text-xs text-primary font-medium hover:underline">
                      {faq.linkLabel} &rarr;
                    </Link>
                  )}
                </div>
              </details>
            ))}
          </div>
        </div>
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "FAQPage",
              mainEntity: [
                {
                  "@type": "Question",
                  name: "Qu'est-ce qu'un Agent IA ?",
                  acceptedAnswer: { "@type": "Answer", text: "Un Agent IA est un programme autonome qui exécute des tâches complexes en utilisant l'intelligence artificielle. Il peut prendre des décisions, interagir avec vos outils et s'améliorer au fil du temps." },
                },
                {
                  "@type": "Question",
                  name: "Les workflows sont-ils vraiment gratuits ?",
                  acceptedAnswer: { "@type": "Answer", text: "Oui, tous les tutoriels sont accessibles gratuitement et sans limite. L'abonnement Pro ajoute les fichiers JSON importables et l'accès anticipé." },
                },
                {
                  "@type": "Question",
                  name: "Faut-il une équipe technique pour implémenter ?",
                  acceptedAnswer: { "@type": "Answer", text: "Pas nécessairement. Chaque workflow utilise n8n (no-code/low-code) avec un tutoriel noeud par noeud." },
                },
                {
                  "@type": "Question",
                  name: "Est-ce compatible avec mes outils existants ?",
                  acceptedAnswer: { "@type": "Answer", text: "Oui. Chaque étape propose plusieurs alternatives (Zendesk, HubSpot, Slack, etc.). Les schémas d'architecture montrent comment connecter vos outils." },
                },
                {
                  "@type": "Question",
                  name: "Mes données sont-elles en sécurité ?",
                  acceptedAnswer: { "@type": "Answer", text: "n8n s'auto-héberge sur votre serveur. Vos données ne transitent jamais par nos systèmes. Bonnes pratiques RGPD incluses." },
                },
                {
                  "@type": "Question",
                  name: "Quel LLM choisir : Claude, GPT-4 ou Mistral ?",
                  acceptedAnswer: { "@type": "Answer", text: "Claude excelle en analyse de documents, GPT-4 en génération de code, Mistral en souveraineté européenne. Chaque tutoriel propose les 4 alternatives." },
                },
                {
                  "@type": "Question",
                  name: "Combien de temps pour déployer un agent IA ?",
                  acceptedAnswer: { "@type": "Answer", text: "Un workflow se déploie en 2 à 4 heures via n8n. Avec l'abonnement Pro, import en 5 minutes." },
                },
                {
                  "@type": "Question",
                  name: "Quel ROI attendre d'un agent IA ?",
                  acceptedAnswer: { "@type": "Answer", text: "Un workflow de triage support économise 60% du temps de première réponse. Un workflow de qualification leads double le taux de conversion." },
                },
              ],
            }),
          }}
        />
      </section>

      {/* Newsletter — realistic promise */}
      <section className="dotted-grid">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-bold sm:text-3xl">
              Nouveaux workflows et guides IA chaque semaine
            </h2>
            <p className="mt-3 text-muted-foreground">
              Rejoignez les professionnels qui reçoivent nos meilleurs
              cas d&apos;usage d&apos;Agent IA avec tutoriel et ROI. Gratuit.
            </p>
            <NewsletterSignup variant="hero" />
          </div>
        </div>
      </section>

      <ExitIntentPopup />
    </>
  );
}
