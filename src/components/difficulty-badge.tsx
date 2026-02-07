import { Badge } from "@/components/ui/badge";
import type { Difficulty } from "@/data/types";

const difficultyConfig: Record<Difficulty, { className: string }> = {
  Facile: {
    className: "bg-emerald-100 text-emerald-800 hover:bg-emerald-100 border-emerald-200",
  },
  Moyen: {
    className: "bg-amber-100 text-amber-800 hover:bg-amber-100 border-amber-200",
  },
  Expert: {
    className: "bg-red-100 text-red-800 hover:bg-red-100 border-red-200",
  },
};

interface DifficultyBadgeProps {
  difficulty: Difficulty;
}

export function DifficultyBadge({ difficulty }: DifficultyBadgeProps) {
  const config = difficultyConfig[difficulty];
  return (
    <Badge variant="outline" className={`shrink-0 text-xs ${config.className}`}>
      {difficulty}
    </Badge>
  );
}
