"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

interface NewsletterSignupProps {
  variant?: "hero" | "inline" | "footer";
}

export function NewsletterSignup({ variant = "inline" }: NewsletterSignupProps) {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email.trim()) return;

    setStatus("loading");
    try {
      const res = await fetch("/api/newsletter", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      const data = await res.json();

      if (res.ok && data.success) {
        setStatus("success");
        setMessage(data.message || "Inscription réussie !");
        setEmail("");
      } else {
        setStatus("error");
        setMessage(data.error || "Une erreur est survenue.");
      }
    } catch {
      setStatus("error");
      setMessage("Erreur de connexion. Réessayez.");
    }
  };

  if (variant === "hero") {
    return (
      <div className="mt-8 mx-auto max-w-md">
        {status === "success" ? (
          <p className="text-sm text-emerald-600 font-medium text-center">{message}</p>
        ) : (
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="votre@email.com"
              required
              className="flex-1 rounded-lg border border-input bg-background px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button type="submit" disabled={status === "loading"} size="default">
              {status === "loading" ? "..." : "C'est gratuit"}
            </Button>
          </form>
        )}
        {status === "error" && (
          <p className="mt-2 text-sm text-destructive text-center">{message}</p>
        )}
        <p className="mt-2 text-xs text-muted-foreground text-center">
          Rejoignez 500+ pros. Désinscription en un clic.
        </p>
      </div>
    );
  }

  if (variant === "footer") {
    return (
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-3">Newsletter</h3>
        {status === "success" ? (
          <p className="text-sm text-emerald-600">{message}</p>
        ) : (
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="votre@email.com"
              required
              className="flex-1 min-w-0 rounded-md border border-input bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
            <Button type="submit" size="sm" disabled={status === "loading"}>
              {status === "loading" ? "..." : "OK"}
            </Button>
          </form>
        )}
        {status === "error" && (
          <p className="mt-1 text-xs text-destructive">{message}</p>
        )}
        <p className="mt-1.5 text-xs text-muted-foreground">
          1 workflow/jour. Gratuit. Désinscription facile.
        </p>
      </div>
    );
  }

  // inline variant (default)
  return (
    <div className="rounded-xl border bg-muted/30 p-6 sm:p-8">
      <div className="max-w-lg">
        <h3 className="text-lg font-semibold">
          Recevez un cas d&apos;usage par jour
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Chaque matin, un nouveau cas d&apos;usage d&apos;Agent IA avec tutoriel complet, directement dans votre boîte mail.
        </p>
        {status === "success" ? (
          <p className="mt-4 text-sm text-emerald-600 font-medium">{message}</p>
        ) : (
          <form onSubmit={handleSubmit} className="mt-4 flex gap-2">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="votre@email.com"
              required
              className="flex-1 min-w-0 rounded-lg border border-input bg-background px-4 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <Button type="submit" disabled={status === "loading"}>
              {status === "loading" ? "Inscription..." : "S'inscrire gratuitement"}
            </Button>
          </form>
        )}
        {status === "error" && (
          <p className="mt-2 text-sm text-destructive">{message}</p>
        )}
      </div>
    </div>
  );
}
