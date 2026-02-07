"use client";

import { Badge } from "@/components/ui/badge";
import type { Difficulty } from "@/data/types";

interface FilterBarProps {
  functions: string[];
  difficulties: Difficulty[];
  sectors: string[];
  activeFunction: string | null;
  activeDifficulty: Difficulty | null;
  activeSector: string | null;
  searchQuery: string;
  resultCount: number;
  onFunctionChange: (fn: string | null) => void;
  onDifficultyChange: (d: Difficulty | null) => void;
  onSectorChange: (s: string | null) => void;
  onSearchChange: (q: string) => void;
}

export function FilterBar({
  functions,
  difficulties,
  sectors,
  activeFunction,
  activeDifficulty,
  activeSector,
  searchQuery,
  resultCount,
  onFunctionChange,
  onDifficultyChange,
  onSectorChange,
  onSearchChange,
}: FilterBarProps) {
  const hasActiveFilters = activeFunction || activeDifficulty || activeSector || searchQuery;

  return (
    <div className="space-y-4">
      {/* Search */}
      <div className="relative">
        <svg
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Rechercher... (ex: SAV, compta, lead)"
          className="w-full rounded-lg border border-input bg-background px-9 py-2.5 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring"
        />
        {searchQuery && (
          <button
            onClick={() => onSearchChange("")}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            aria-label="Effacer la recherche"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        )}
      </div>

      {/* Result count + clear */}
      {hasActiveFilters && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>{resultCount} résultat{resultCount !== 1 ? "s" : ""}</span>
          <button
            onClick={() => {
              onFunctionChange(null);
              onDifficultyChange(null);
              onSectorChange(null);
              onSearchChange("");
            }}
            className="text-primary hover:underline"
          >
            Réinitialiser
          </button>
        </div>
      )}

      {/* Filters */}
      <div className="space-y-3">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Fonction
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Badge
              variant={activeFunction === null ? "default" : "outline"}
              className="cursor-pointer text-xs"
              onClick={() => onFunctionChange(null)}
            >
              Toutes
            </Badge>
            {functions.map((fn) => (
              <Badge
                key={fn}
                variant={activeFunction === fn ? "default" : "outline"}
                className="cursor-pointer text-xs"
                onClick={() => onFunctionChange(fn === activeFunction ? null : fn)}
              >
                {fn}
              </Badge>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Difficulté
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Badge
              variant={activeDifficulty === null ? "default" : "outline"}
              className="cursor-pointer text-xs"
              onClick={() => onDifficultyChange(null)}
            >
              Toutes
            </Badge>
            {difficulties.map((d) => (
              <Badge
                key={d}
                variant={activeDifficulty === d ? "default" : "outline"}
                className="cursor-pointer text-xs"
                onClick={() => onDifficultyChange(d === activeDifficulty ? null : d)}
              >
                {d}
              </Badge>
            ))}
          </div>
        </div>

        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Secteur
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Badge
              variant={activeSector === null ? "default" : "outline"}
              className="cursor-pointer text-xs"
              onClick={() => onSectorChange(null)}
            >
              Tous
            </Badge>
            {sectors.map((s) => (
              <Badge
                key={s}
                variant={activeSector === s ? "default" : "outline"}
                className="cursor-pointer text-xs"
                onClick={() => onSectorChange(s === activeSector ? null : s)}
              >
                {s}
              </Badge>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
