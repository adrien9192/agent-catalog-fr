"use client";

import { useState, useMemo } from "react";
import { useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { UseCaseCard } from "@/components/use-case-card";
import { FilterBar } from "@/components/filter-bar";
import { useCases } from "@/data/use-cases";
import type { Difficulty } from "@/data/types";

function CatalogueContent() {
  const searchParams = useSearchParams();
  const initialFn = searchParams.get("fn");
  const initialDiff = searchParams.get("diff") as Difficulty | null;
  const initialQ = searchParams.get("q") || "";

  const [activeFunction, setActiveFunction] = useState<string | null>(initialFn);
  const [activeDifficulty, setActiveDifficulty] = useState<Difficulty | null>(initialDiff);
  const [activeSector, setActiveSector] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState(initialQ);

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
    return useCases.filter((uc) => {
      if (activeFunction && !uc.functions.includes(activeFunction)) return false;
      if (activeDifficulty && uc.difficulty !== activeDifficulty) return false;
      if (activeSector && !uc.sectors.includes(activeSector)) return false;
      if (searchQuery) {
        const q = searchQuery.toLowerCase();
        const haystack = `${uc.title} ${uc.subtitle} ${uc.problem} ${uc.functions.join(" ")} ${uc.sectors.join(" ")} ${uc.metiers.join(" ")}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
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
        <aside className="lg:sticky lg:top-20 lg:self-start">
          <FilterBar
            functions={allFunctions}
            difficulties={allDifficulties}
            sectors={allSectors}
            activeFunction={activeFunction}
            activeDifficulty={activeDifficulty}
            activeSector={activeSector}
            searchQuery={searchQuery}
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
            <div className="rounded-xl border border-dashed p-12 text-center">
              <p className="text-muted-foreground">
                Aucun cas d&apos;usage ne correspond à vos filtres.
              </p>
            </div>
          )}
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
