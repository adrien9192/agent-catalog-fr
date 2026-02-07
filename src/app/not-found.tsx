import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-24 sm:px-6 text-center">
      <p className="text-6xl font-bold text-primary mb-4">404</p>
      <h1 className="text-2xl font-bold sm:text-3xl mb-3">
        Page introuvable
      </h1>
      <p className="text-muted-foreground mb-8 max-w-md mx-auto">
        Cette page n&apos;existe pas ou a été déplacée.
        Retrouvez nos workflows IA et guides pratiques ci-dessous.
      </p>

      <div className="flex flex-col sm:flex-row gap-3 justify-center">
        <Button asChild>
          <Link href="/catalogue">Voir les 55 workflows</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/guide">Lire nos guides</Link>
        </Button>
        <Button variant="outline" asChild>
          <Link href="/demande">Workflow sur mesure</Link>
        </Button>
      </div>

      <div className="mt-12 rounded-xl border bg-muted/30 p-6 text-left max-w-lg mx-auto">
        <h2 className="font-semibold mb-3 text-sm">Pages populaires</h2>
        <ul className="space-y-2 text-sm">
          <li>
            <Link href="/use-case/agent-triage-support-client" className="text-muted-foreground hover:text-foreground transition-colors">
              Agent de triage support client
            </Link>
          </li>
          <li>
            <Link href="/use-case/agent-qualification-leads" className="text-muted-foreground hover:text-foreground transition-colors">
              Agent de qualification des leads
            </Link>
          </li>
          <li>
            <Link href="/guide/agent-ia-entreprise-guide-complet" className="text-muted-foreground hover:text-foreground transition-colors">
              Guide : Agent IA en entreprise
            </Link>
          </li>
          <li>
            <Link href="/pricing" className="text-muted-foreground hover:text-foreground transition-colors">
              Tarifs et plans
            </Link>
          </li>
        </ul>
      </div>
    </div>
  );
}
