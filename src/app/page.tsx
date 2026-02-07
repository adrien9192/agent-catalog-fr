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

const functions = ["Support", "Sales", "RH", "Marketing", "Finance", "IT", "Supply Chain"];

const stats = [
  { value: "30+", label: "workflows documentés" },
  { value: "60%", label: "de temps gagné en moyenne" },
  { value: "13", label: "fonctions couvertes" },
  { value: "100%", label: "gratuit pour démarrer" },
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
    title: "Trouvez votre workflow",
    description: "Parcourez le catalogue ou décrivez votre besoin. Chaque workflow est classé par fonction, secteur et difficulté.",
  },
  {
    step: "2",
    title: "Suivez le tutoriel",
    description: "Stack technique, code prêt à copier, schéma d'architecture. Tout est documenté étape par étape.",
  },
  {
    step: "3",
    title: "Déployez en production",
    description: "Estimez le ROI, gérez les risques, et mettez en production avec les bonnes pratiques enterprise.",
  },
];

export default function HomePage() {
  const featured = useCases.slice(0, 6);

  return (
    <>
      {/* Hero */}
      <section className="dotted-grid relative overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <Badge variant="secondary" className="mb-4 text-xs">
              Gratuit et open-source
            </Badge>
            <h1 className="text-3xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Déployez un Agent IA{" "}
              <span className="gradient-text">en quelques heures,</span>
              <br className="hidden sm:block" />
              pas en 6 mois
            </h1>
            <p className="mt-4 text-base text-muted-foreground sm:text-xl max-w-2xl mx-auto leading-relaxed">
              25+ workflows IA documentés avec tutoriel pas-à-pas, stack technique
              et estimation de ROI. Prêts à copier et déployer. Gratuit.
            </p>

            {/* CTA pills */}
            <div className="mt-8 flex flex-wrap justify-center gap-2">
              {functions.map((fn) => (
                <Link key={fn} href={`/catalogue?fn=${fn}`}>
                  <Badge
                    variant="outline"
                    className="cursor-pointer px-3 py-1.5 text-sm transition-colors hover:bg-primary hover:text-primary-foreground"
                  >
                    {fn}
                  </Badge>
                </Link>
              ))}
            </div>

            {/* Prompt bar */}
            <div className="mt-10">
              <PromptBar />
            </div>
          </div>
        </div>
      </section>

      {/* Social proof / stats */}
      <section className="border-y bg-muted/20">
        <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 gap-4 sm:gap-6 sm:grid-cols-4">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <p className="text-3xl font-bold text-primary">{stat.value}</p>
                <p className="mt-1 text-sm text-muted-foreground">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trust signals */}
      <section className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-3 text-xs sm:text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
            </svg>
            <span>Conforme RGPD</span>
          </div>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <polyline points="20 6 9 17 4 12"/>
            </svg>
            <span>Code open-source</span>
          </div>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
            </svg>
            <span>Mis à jour quotidiennement</span>
          </div>
          <div className="flex items-center gap-2">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/>
              <circle cx="9" cy="7" r="4"/>
              <path d="M23 21v-2a4 4 0 0 0-3-3.87"/>
              <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
            </svg>
            <span>Conçu pour le marché français</span>
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
                <p className="text-sm text-muted-foreground leading-relaxed">{s.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      {/* Testimonials */}
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

      {/* Featured use cases */}
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
              <Link href="/catalogue">Voir tout</Link>
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

      {/* Sectors */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <h2 className="text-2xl font-bold sm:text-3xl mb-2">
          Votre secteur, nos workflows
        </h2>
        <p className="text-muted-foreground mb-8">
          Des solutions adaptées à chaque industrie, de la banque au retail.
        </p>
        <div className="grid gap-2 sm:gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
          {sectors.slice(0, 8).map((sector) => (
            <Link
              key={sector.slug}
              href={`/secteur/${sector.slug}`}
              className="group rounded-xl border bg-card p-4 transition-all hover:shadow-sm hover:border-primary/30"
            >
              <span className="text-2xl">{sector.icon}</span>
              <h3 className="mt-2 font-semibold text-sm group-hover:text-primary transition-colors">
                {sector.name}
              </h3>
              <p className="mt-1 text-xs text-muted-foreground line-clamp-2">
                {sector.description}
              </p>
            </Link>
          ))}
        </div>
      </section>

      {/* Recently added */}
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
          {useCases
            .sort((a, b) => b.createdAt.localeCompare(a.createdAt))
            .slice(0, 3)
            .map((uc) => (
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

      {/* CTA: custom request */}
      <section className="border-t bg-muted/30">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-bold sm:text-3xl">
              Votre workflow n&apos;existe pas encore ?
            </h2>
            <p className="mt-3 text-muted-foreground max-w-lg mx-auto">
              Décrivez votre besoin d&apos;automatisation et notre équipe développera
              un workflow sur mesure avec tutoriel complet et estimation de ROI.
            </p>
            <div className="mt-6">
              <Button size="lg" asChild>
                <Link href="/demande">Demander un workflow sur mesure</Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Newsletter */}
      <section className="dotted-grid">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-bold sm:text-3xl">
              Un nouveau workflow chaque matin
            </h2>
            <p className="mt-3 text-muted-foreground">
              Rejoignez les professionnels qui reçoivent chaque jour un nouveau
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
