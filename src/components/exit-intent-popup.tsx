"use client";

import { useState, useEffect, useCallback } from "react";
import { Button } from "@/components/ui/button";

export function ExitIntentPopup() {
  const [show, setShow] = useState(false);
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [message, setMessage] = useState("");

  const handleMouseLeave = useCallback((e: MouseEvent) => {
    if (e.clientY <= 0 && !show) {
      const dismissed = sessionStorage.getItem("exit-popup-dismissed");
      if (!dismissed) {
        setShow(true);
      }
    }
  }, [show]);

  useEffect(() => {
    const timer = setTimeout(() => {
      document.addEventListener("mouseleave", handleMouseLeave);
    }, 5000);

    return () => {
      clearTimeout(timer);
      document.removeEventListener("mouseleave", handleMouseLeave);
    };
  }, [handleMouseLeave]);

  const handleDismiss = () => {
    setShow(false);
    sessionStorage.setItem("exit-popup-dismissed", "1");
  };

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
        setMessage("Inscription confirmée !");
        setTimeout(() => {
          handleDismiss();
        }, 2000);
      } else {
        setStatus("error");
        setMessage(data.error || "Une erreur est survenue.");
      }
    } catch {
      setStatus("error");
      setMessage("Erreur de connexion. Réessayez.");
    }
  };

  if (!show) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative mx-3 sm:mx-4 w-full max-w-md rounded-2xl border bg-background p-5 sm:p-8 shadow-2xl animate-in slide-in-from-bottom-4 duration-300">
        <button
          onClick={handleDismiss}
          className="absolute right-4 top-4 rounded-full p-1 text-muted-foreground hover:text-foreground transition-colors"
          aria-label="Fermer"
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>

        {status === "success" ? (
          <div className="text-center py-4">
            <div className="text-4xl mb-3">&#10003;</div>
            <p className="text-lg font-semibold">{message}</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Votre premier workflow arrive demain matin.
            </p>
          </div>
        ) : (
          <>
            <div className="text-center mb-6">
              <p className="text-sm font-medium text-primary mb-2">Avant de partir...</p>
              <h2 className="text-xl font-bold sm:text-2xl">
                Recevez un workflow IA gratuit chaque matin
              </h2>
              <p className="mt-2 text-sm text-muted-foreground">
                Tutoriel pas-à-pas, stack technique et estimation de ROI.
                Rejoignez les professionnels qui automatisent leur entreprise.
              </p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-3">
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="votre@email.com"
                required
                className="w-full rounded-lg border border-input bg-background px-4 py-3 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
                autoFocus
              />
              <Button type="submit" className="w-full" size="lg" disabled={status === "loading"}>
                {status === "loading" ? "Inscription..." : "Recevoir mon workflow gratuit"}
              </Button>
              {status === "error" && (
                <p className="text-sm text-destructive text-center">{message}</p>
              )}
            </form>

            <p className="mt-3 text-xs text-muted-foreground text-center">
              Gratuit. Désinscription en un clic. Pas de spam.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
