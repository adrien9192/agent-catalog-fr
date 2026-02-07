"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Suspense } from "react";
import Link from "next/link";
import { UseCaseCard } from "@/components/use-case-card";
import { FilterBar } from "@/components/filter-bar";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { ExitIntentPopup } from "@/components/exit-intent-popup";
import { useCases } from "@/data/use-cases";
import { createSearchIndex, searchUseCases } from "@/lib/search";
import type { Difficulty } from "@/data/types";

const fuseIndex = createSearchIndex(useCases);

const SUGGESTIONS = [
  "support client",
  "qualification leads",
  "recrutement CV",
  "veille concurrentielle",
  "rapports financiers",
  "incidents IT",
  "contenu marketing",
  "onboarding RH",
  "détection fraude",
  "achats fournisseurs",
];

function CatalogueContent() {
  const searchParams = useSearchParams();
  const router = useRouter();

  // Read URL params
  const urlQ = searchParams.get("q") || "";
  const urlFn = searchParams.get("fn");
  const urlDiff = searchParams.get("diff") as Difficulty | null;
  const urlSector = searchParams.get("sector");

  const [activeFunction, setActiveFunction] = useState<string | null>(urlFn);
  const [activeDifficulty, setActiveDifficulty] = useState<Difficulty | null>(urlDiff);
  const [activeSector, setActiveSector] = useState<string | null>(urlSector);
  const [searchQuery, setSearchQuery] = useState(urlQ);

  // Sync URL → state (handles external navigations like PromptBar, back/forward)
  useEffect(() => {
    setSearchQuery(urlQ);
    setActiveFunction(urlFn);
    setActiveDifficulty(urlDiff);
    setActiveSector(urlSector);
  }, [urlQ, urlFn, urlDiff, urlSector]);

  // Sync state → URL (debounced, for user typing in filter bar)
  const syncUrl = useCallback(
    (q: string, fn: string | null, diff: Difficulty | null, sector: string | null) => {
      const params = new URLSearchParams();
      if (q) params.set("q", q);
      if (fn) params.set("fn", fn);
      if (diff) params.set("diff", diff);
      if (sector) params.set("sector", sector);
      const qs = params.toString();
      router.replace(`/catalogue${qs ? `?${qs}` : ""}`, { scroll: false });
    },
    [router]
  );

  useEffect(() => {
    const t = setTimeout(() => {
      syncUrl(searchQuery, activeFunction, activeDifficulty, activeSector);
    }, 300);
    return () => clearTimeout(t);
  }, [searchQuery, activeFunction, activeDifficulty, activeSector, syncUrl]);

  const allFunctions = useMemo(
    () => [...new Set(useCases.flatMap((uc) => uc.functions))].sort(),
    []
  );
  const allDifficulties: Difficulty[] = ["Facile", "Moyen", "Expert"];
  const allSectors = useMemo(
    () => [...new Set(useCases.flatMap((uc) => uc.sectors))].sort(),
    []
  );

  const filtered = useMemo(() => {
    let results = useCases;

    // Per-word search with synonym expansion
    if (searchQuery.trim()) {
      results = searchUseCases(fuseIndex, searchQuery);
    }

    // Apply tag filters on top
    return results.filter((uc) => {
      if (activeFunction && !uc.functions.includes(activeFunction)) return false;
      if (activeDifficulty && uc.difficulty !== activeDifficulty) return false;
      if (activeSector && !uc.sectors.includes(activeSector)) return false;
      return true;
    });
  }, [activeFunction, activeDifficulty, activeSector, searchQuery]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold sm:text-4xl">Workflows IA</h1>
        <p className="mt-2 text-muted-foreground">
          {useCases.length} workflows d&apos;Agents IA documentés, avec tutoriel et ROI. Prêts à déployer.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-[280px_1fr]">
        {/* Sidebar filters */}
        <aside className="lg:sticky lg:top-20 lg:self-start space-y-6">
          <FilterBar
            functions={allFunctions}
            difficulties={allDifficulties}
            sectors={allSectors}
            activeFunction={activeFunction}
            activeDifficulty={activeDifficulty}
            activeSector={activeSector}
            searchQuery={searchQuery}
            resultCount={filtered.length}
            onFunctionChange={setActiveFunction}
            onDifficultyChange={setActiveDifficulty}
            onSectorChange={setActiveSector}
            onSearchChange={setSearchQuery}
          />
        </aside>

        {/* Results */}
        <div>
          <p className="mb-4 text-sm text-muted-foreground">
            {filtered.length} résultat{filtered.length !== 1 ? "s" : ""}
          </p>
          {filtered.length > 0 ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {filtered.map((uc) => (
                <UseCaseCard key={uc.slug} useCase={uc} searchQuery={searchQuery} />
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed p-12 text-center space-y-6">
              <div className="space-y-2">
                <p className="text-lg font-medium">
                  Aucun workflow ne correspond à &quot;{searchQuery}&quot;
                </p>
                <p className="text-sm text-muted-foreground">
                  Pas encore disponible ? Décrivez votre besoin et nous le développerons pour vous.
                </p>
              </div>
              <Link
                href={`/demande?q=${encodeURIComponent(searchQuery)}`}
                className="inline-flex items-center gap-2 rounded-lg bg-primary px-6 py-3 text-sm font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Demander ce workflow
              </Link>
              <div>
                <p className="text-sm text-muted-foreground mb-2">Ou essayez :</p>
                <div className="flex flex-wrap justify-center gap-1.5">
                  {SUGGESTIONS.slice(0, 5).map((s) => (
                    <button
                      key={s}
                      onClick={() => setSearchQuery(s)}
                      className="rounded-full border px-3 py-1 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Newsletter CTA after results */}
          <div className="mt-12">
            <NewsletterSignup variant="inline" />
          </div>
        </div>
      </div>
    </div>
  );
}

export default function CataloguePage() {
  return (
    <Suspense
      fallback={
        <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
          <p className="text-muted-foreground">Chargement...</p>
        </div>
      }
    >
      <CatalogueContent />
      <ExitIntentPopup />
    </Suspense>
  );
}
