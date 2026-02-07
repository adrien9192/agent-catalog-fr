"use client";

import { useState, useMemo, useCallback, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Suspense } from "react";
import { UseCaseCard } from "@/components/use-case-card";
import { FilterBar } from "@/components/filter-bar";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { useCases } from "@/data/use-cases";
import { createSearchIndex, expandQuery } from "@/lib/search";
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

  const initialFn = searchParams.get("fn");
  const initialDiff = searchParams.get("diff") as Difficulty | null;
  const initialQ = searchParams.get("q") || "";

  const [activeFunction, setActiveFunction] = useState<string | null>(initialFn);
  const [activeDifficulty, setActiveDifficulty] = useState<Difficulty | null>(initialDiff);
  const [activeSector, setActiveSector] = useState<string | null>(searchParams.get("sector"));
  const [searchQuery, setSearchQuery] = useState(initialQ);

  // Sync filters → URL
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

  // Debounced URL sync for search query
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

    // Fuse.js search if query present
    if (searchQuery.trim()) {
      const expanded = expandQuery(searchQuery);
      const fuseResults = fuseIndex.search(expanded);
      results = fuseResults.map((r) => r.item);
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
        <h1 className="text-3xl font-bold sm:text-4xl">Catalogue</h1>
        <p className="mt-2 text-muted-foreground">
          {useCases.length} cas d&apos;usage d&apos;Agents IA documentés et prêts à implanter.
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
                <UseCaseCard key={uc.slug} useCase={uc} />
              ))}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed p-12 text-center space-y-4">
              <p className="text-muted-foreground">
                Aucun cas d&apos;usage ne correspond à votre recherche.
              </p>
              <div>
                <p className="text-sm text-muted-foreground mb-2">Suggestions :</p>
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
    </Suspense>
  );
}
