import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { DifficultyBadge } from "@/components/difficulty-badge";
import type { UseCase } from "@/data/types";

interface UseCaseCardProps {
  useCase: UseCase;
  searchQuery?: string;
}

function highlightText(text: string, query: string) {
  if (!query || query.length < 2) return text;
  const words = query.toLowerCase().split(/\s+/).filter((w) => w.length >= 2);
  if (words.length === 0) return text;

  const pattern = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
  const parts = text.split(pattern);

  return parts.map((part, i) =>
    words.some((w) => part.toLowerCase() === w) ? (
      <mark key={i} className="bg-primary/20 text-foreground rounded-sm px-0.5">
        {part}
      </mark>
    ) : (
      part
    )
  );
}

export function UseCaseCard({ useCase, searchQuery }: UseCaseCardProps) {
  return (
    <Link href={`/use-case/${useCase.slug}`} className="group block">
      <Card className="h-full transition-all duration-200 hover:shadow-md hover:border-primary/30 group-hover:-translate-y-0.5">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-lg font-semibold leading-snug group-hover:text-primary transition-colors">
              {searchQuery ? highlightText(useCase.title, searchQuery) : useCase.title}
            </h3>
            <DifficultyBadge difficulty={useCase.difficulty} />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {searchQuery ? highlightText(useCase.subtitle, searchQuery) : useCase.subtitle}
          </p>
        </CardHeader>
        <CardContent className="pt-0">
          <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
            {searchQuery ? highlightText(useCase.problem, searchQuery) : useCase.problem}
          </p>
          <div className="flex flex-wrap gap-1.5">
            {useCase.functions.map((fn) => (
              <Badge key={fn} variant="secondary" className="text-xs">
                {fn}
              </Badge>
            ))}
            {useCase.sectors.slice(0, 2).map((s) => (
              <Badge key={s} variant="outline" className="text-xs">
                {s}
              </Badge>
            ))}
            {useCase.estimatedTime && (
              <Badge variant="outline" className="text-xs text-muted-foreground">
                {useCase.estimatedTime}
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
