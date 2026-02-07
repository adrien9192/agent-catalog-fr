"use client";

import { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="mx-auto max-w-2xl px-4 py-24 text-center">
      <h1 className="text-3xl font-bold mb-4">Une erreur est survenue</h1>
      <p className="text-muted-foreground mb-8">
        Nous nous excusons pour ce désagrément. Veuillez réessayer ou revenir à
        l&apos;accueil.
      </p>
      <div className="flex justify-center gap-4">
        <Button onClick={reset}>Réessayer</Button>
        <Button variant="outline" asChild>
          <Link href="/">Retour à l&apos;accueil</Link>
        </Button>
      </div>
    </div>
  );
}
