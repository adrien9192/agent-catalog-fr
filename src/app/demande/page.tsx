"use client";

import { useState, useEffect } from "react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

function DemandeContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "";

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [description, setDescription] = useState(
    initialQuery ? `Je cherche un workflow pour : ${initialQuery}\n\n` : ""
  );
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    if (initialQuery && !description.includes(initialQuery)) {
      setDescription(`Je cherche un workflow pour : ${initialQuery}\n\n`);
    }
  }, [initialQuery]); // eslint-disable-line react-hooks/exhaustive-deps

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus("loading");
    setErrorMsg("");

    try {
      const res = await fetch("/api/request", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          email,
          description,
          searchQuery: initialQuery || undefined,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setErrorMsg(data.error || "Une erreur est survenue.");
        setStatus("error");
        return;
      }

      setStatus("success");
    } catch {
      setErrorMsg("Erreur de connexion. Réessayez.");
      setStatus("error");
    }
  }

  if (status === "success") {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16 sm:px-6 text-center">
        <div className="rounded-2xl border bg-primary/5 p-6 sm:p-12 space-y-4">
          <div className="text-5xl">&#10003;</div>
          <h1 className="text-2xl font-bold">Demande envoyée !</h1>
          <p className="text-muted-foreground max-w-md mx-auto">
            Vous recevrez un email de confirmation sous 48h. Votre workflow
            sera livré sous 5 jours ouvrés avec tutoriel complet et estimation de ROI.
          </p>
          <div className="pt-4 flex flex-col sm:flex-row gap-3 justify-center">
            <Link
              href="/catalogue"
              className="rounded-lg border px-5 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
            >
              Explorer les 30+ workflows
            </Link>
            <Link
              href="/guide"
              className="rounded-lg border px-5 py-2.5 text-sm font-medium hover:bg-accent transition-colors"
            >
              Lire nos guides IA
            </Link>
            <Link
              href="/pricing"
              className="rounded-lg bg-primary px-5 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Voir les plans
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8 sm:px-6">
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <Link href="/catalogue" className="hover:text-foreground">Catalogue</Link>
        {" / "}
        <span className="text-foreground">Demander un workflow</span>
      </nav>

      <div className="mb-8">
        <h1 className="text-3xl font-bold sm:text-4xl">Demander un workflow</h1>
        <p className="mt-3 text-muted-foreground max-w-xl">
          Vous n&apos;avez pas trouvé le cas d&apos;usage que vous cherchez ?
          Décrivez votre besoin et nous développerons un workflow sur mesure avec
          tutoriel complet, stack recommandée et estimation de ROI.
        </p>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <h2 className="font-semibold">Décrivez votre besoin</h2>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label htmlFor="name" className="block text-sm font-medium mb-1.5">
                Votre nom
              </label>
              <input
                id="name"
                type="text"
                required
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Jean Dupont"
                className="w-full rounded-lg border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium mb-1.5">
                Votre email
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="jean@entreprise.com"
                className="w-full rounded-lg border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50"
              />
            </div>

            <div>
              <label htmlFor="description" className="block text-sm font-medium mb-1.5">
                Décrivez le workflow que vous souhaitez
              </label>
              <textarea
                id="description"
                required
                rows={6}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder={"Secteur : ex. Banque, E-commerce, SaaS...\nBesoin : ex. Trier les emails entrants par priorité\nVolume : ex. 500 emails/jour\nOutils existants : ex. Zendesk, Gmail, Salesforce\nRésultat attendu : ex. 50% de temps gagné sur le support"}
                className="w-full rounded-lg border bg-background px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 resize-y"
              />
              {description.length > 0 && description.length < 50 && (
                <p className="mt-1.5 text-xs text-amber-600">
                  Plus votre description est précise, plus le workflow sera adapté à votre contexte.
                </p>
              )}
              {(description.length === 0 || description.length >= 50) && (
                <p className="mt-1.5 text-xs text-muted-foreground">
                  Secteur, outils utilisés, volume de données, résultat attendu...
                </p>
              )}
            </div>

            {status === "error" && (
              <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {errorMsg}
              </div>
            )}

            <button
              type="submit"
              disabled={status === "loading"}
              className="w-full rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {status === "loading" ? "Envoi en cours..." : "Recevoir mon workflow sur mesure"}
            </button>
            <p className="text-xs text-muted-foreground text-center">
              Réponse sous 48h. Workflow livré sous 5 jours ouvrés.
            </p>
          </form>
        </CardContent>
      </Card>

      <div className="mt-8 rounded-xl border bg-muted/30 p-6">
        <h3 className="font-semibold mb-3">Comment ça marche ?</h3>
        <ol className="space-y-3 text-sm text-muted-foreground">
          <li className="flex gap-3">
            <span className="shrink-0 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">1</span>
            <span>Vous décrivez votre besoin d&apos;automatisation</span>
          </li>
          <li className="flex gap-3">
            <span className="shrink-0 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">2</span>
            <span>Notre équipe analyse et développe le workflow complet</span>
          </li>
          <li className="flex gap-3">
            <span className="shrink-0 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">3</span>
            <span>Vous recevez un email dès que le cas d&apos;usage est en ligne</span>
          </li>
        </ol>
      </div>

      {/* What's included */}
      <div className="mt-6 rounded-xl border p-6">
        <h3 className="font-semibold mb-3">Chaque workflow sur mesure inclut :</h3>
        <ul className="grid gap-2 text-sm text-muted-foreground grid-cols-1 sm:grid-cols-2">
          {[
            "Tutoriel pas-à-pas complet",
            "Stack technique recommandée",
            "Alternatives low-cost gratuites",
            "Schéma d'architecture",
            "Code Python fonctionnel",
            "Estimation de ROI chiffrée",
            "Considérations enterprise (RGPD)",
            "Workflow n8n automatisé",
          ].map((item) => (
            <li key={item} className="flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="shrink-0 text-primary">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Trust */}
      <div className="mt-6 flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-xs text-muted-foreground">
        <span className="flex items-center gap-1">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          Conforme RGPD
        </span>
        <span>|</span>
        <span>Réponse sous 48h</span>
        <span>|</span>
        <span>Livraison sous 5 jours ouvrés</span>
      </div>
    </div>
  );
}

export default function DemandePage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-2xl px-4 py-8 sm:px-6">
          <p className="text-muted-foreground">Chargement...</p>
        </div>
      }
    >
      <DemandeContent />
    </Suspense>
  );
}
