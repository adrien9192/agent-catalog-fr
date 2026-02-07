import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { PromptBar } from "@/components/prompt-bar";
import { UseCaseCard } from "@/components/use-case-card";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { useCases } from "@/data/use-cases";
import { sectors } from "@/data/sectors";

const functions = ["Support", "Sales", "RH", "Marketing", "Finance", "IT", "Supply Chain"];

export default function HomePage() {
  const featured = useCases.slice(0, 6);

  return (
    <>
      {/* Hero */}
      <section className="dotted-grid relative overflow-hidden">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 sm:py-24 lg:px-8 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <Badge variant="secondary" className="mb-4 text-xs">
              10+ cas d&apos;usage documentés
            </Badge>
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl lg:text-6xl">
              Agents IA{" "}
              <span className="gradient-text">implantables</span>
              <br />
              en entreprise
            </h1>
            <p className="mt-4 text-lg text-muted-foreground sm:text-xl max-w-2xl mx-auto leading-relaxed">
              Découvrez des cas d&apos;usage concrets, avec stack technique,
              tutoriels pas-à-pas et estimation de ROI. Prêts à déployer.
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

      {/* Featured use cases */}
      <section className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
        <div className="flex items-end justify-between mb-8">
          <div>
            <h2 className="text-2xl font-bold sm:text-3xl">
              Cas d&apos;usage populaires
            </h2>
            <p className="mt-1 text-muted-foreground">
              Les solutions IA les plus demandées par les entreprises.
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
            <Link href="/catalogue">Voir tous les cas d&apos;usage</Link>
          </Button>
        </div>
      </section>

      {/* Sectors */}
      <section className="border-t bg-muted/30">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <h2 className="text-2xl font-bold sm:text-3xl mb-8">
            Explorer par secteur
          </h2>
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-4">
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
        </div>
      </section>

      {/* Newsletter */}
      <section className="dotted-grid">
        <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-2xl font-bold sm:text-3xl">
              Un cas d&apos;usage par jour dans votre boîte mail
            </h2>
            <p className="mt-3 text-muted-foreground">
              Recevez chaque matin un nouveau cas d&apos;usage d&apos;Agent IA avec
              tutoriel complet, stack recommandée et estimation de ROI.
            </p>
            <NewsletterSignup variant="hero" />
          </div>
        </div>
      </section>
    </>
  );
}
