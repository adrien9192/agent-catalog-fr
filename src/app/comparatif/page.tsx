import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { comparisons } from "@/data/comparisons";

export const metadata: Metadata = {
  title: "Comparatifs — Agent IA, Chatbot, Claude vs ChatGPT, n8n vs Make",
  description:
    "Comparez les solutions d'IA pour votre entreprise. Claude vs ChatGPT, Agent IA vs Chatbot, n8n vs Make vs Zapier, IA interne vs SaaS. 5 comparatifs détaillés.",
};

export default function ComparatifIndexPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="text-center mb-12">
        <Badge variant="secondary" className="mb-4 text-xs">
          Guides de choix
        </Badge>
        <h1 className="text-3xl font-bold sm:text-4xl lg:text-5xl">
          Comparatifs <span className="gradient-text">objectifs</span>
        </h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          Des comparaisons détaillées pour vous aider à choisir les bons outils
          et la bonne approche pour votre entreprise.
        </p>
      </div>

      <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 max-w-5xl mx-auto">
        {comparisons.map((c) => (
          <Link
            key={c.slug}
            href={`/comparatif/${c.slug}`}
            className="group rounded-xl border bg-card p-6 transition-all hover:shadow-md hover:border-primary/30"
          >
            <div className="flex items-center gap-2 mb-3">
              {c.options.map((o) => (
                <Badge key={o.name} variant="outline" className="text-xs">
                  {o.name}
                </Badge>
              ))}
            </div>
            <h2 className="font-semibold text-base leading-snug group-hover:text-primary transition-colors">
              {c.title}
            </h2>
            <p className="mt-2 text-sm text-muted-foreground line-clamp-2">
              {c.intro.slice(0, 150)}...
            </p>
            <p className="mt-3 text-xs text-primary font-medium">
              Lire le comparatif &rarr;
            </p>
          </Link>
        ))}
      </div>

      <div className="mt-16 text-center">
        <p className="text-muted-foreground mb-4">
          Vous ne trouvez pas le comparatif que vous cherchez ?
        </p>
        <Button variant="outline" asChild>
          <Link href="/demande">Suggérer un comparatif</Link>
        </Button>
      </div>
    </div>
  );
}
