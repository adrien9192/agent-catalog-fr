"use client";

import { useState } from "react";
import Link from "next/link";

interface QuickROIEstimatorProps {
  label?: string;
  unitLabel?: string;
  timePerUnitMinutes?: number;
  timeWithAISeconds?: number;
  options?: number[];
}

export function QuickROIEstimator({
  label = "Combien traitez-vous d'éléments par jour ?",
  unitLabel = "Traitement manuel / sem.",
  timePerUnitMinutes = 3,
  timeWithAISeconds = 30,
  options = [10, 30, 50, 100, 200],
}: QuickROIEstimatorProps) {
  const [volume, setVolume] = useState<number | null>(null);

  const savings = volume
    ? {
        hoursNow: Math.round((volume * 5 * timePerUnitMinutes) / 60),
        hoursWithAI: Math.max(
          1,
          Math.round((volume * 5 * (timeWithAISeconds / 60)) / 60)
        ),
        saved: Math.round(
          (volume * 5 * (timePerUnitMinutes - timeWithAISeconds / 60)) / 60
        ),
      }
    : null;

  return (
    <div className="rounded-xl border bg-muted/20 p-5 sm:p-6">
      <p className="font-semibold text-sm mb-1">{label}</p>
      <p className="text-xs text-muted-foreground mb-4">
        Estimez le temps gagné en 1 clic.
      </p>
      <div className="flex flex-wrap gap-2 mb-4">
        {options.map((v) => (
          <button
            key={v}
            onClick={() => setVolume(v)}
            className={`rounded-lg border px-4 py-2 text-sm font-medium transition-all ${
              volume === v
                ? "border-primary bg-primary text-primary-foreground shadow-sm"
                : "hover:border-primary/50 hover:bg-accent"
            }`}
          >
            {v}
          </button>
        ))}
      </div>
      {savings && savings.saved > 0 && (
        <div className="animate-in fade-in slide-in-from-bottom-2 duration-300 space-y-4">
          <div className="grid grid-cols-3 gap-3 rounded-lg border bg-background p-4">
            <div className="text-center">
              <p className="text-2xl font-bold text-destructive">
                {savings.hoursNow}h
              </p>
              <p className="text-[11px] text-muted-foreground">{unitLabel}</p>
            </div>
            <div className="flex items-center justify-center">
              <span className="text-xl text-muted-foreground">&rarr;</span>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-primary">
                {savings.hoursWithAI}h
              </p>
              <p className="text-[11px] text-muted-foreground">
                Avec l&apos;agent IA
              </p>
            </div>
          </div>
          <p className="text-sm font-medium text-center">
            Vous économiseriez{" "}
            <span className="text-primary font-bold">
              {savings.saved}h par semaine
            </span>{" "}
            — pour 29&euro;/mois.
          </p>
          <div className="flex flex-col items-center gap-1.5">
            <Link
              href="/pricing"
              className="rounded-lg bg-primary px-6 py-2.5 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
            >
              Commencer l&apos;essai gratuit
            </Link>
            <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
              Sans carte bancaire
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
