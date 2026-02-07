import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { DifficultyBadge } from "@/components/difficulty-badge";
import type { UseCase } from "@/data/types";

interface UseCaseCardProps {
  useCase: UseCase;
}

export function UseCaseCard({ useCase }: UseCaseCardProps) {
  return (
    <Link href={`/use-case/${useCase.slug}`} className="group block">
      <Card className="h-full transition-all duration-200 hover:shadow-md hover:border-primary/30 group-hover:-translate-y-0.5">
        <CardHeader className="pb-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-lg font-semibold leading-snug group-hover:text-primary transition-colors">
              {useCase.title}
            </h3>
            <DifficultyBadge difficulty={useCase.difficulty} />
          </div>
          <p className="text-sm text-muted-foreground leading-relaxed">
            {useCase.subtitle}
          </p>
        </CardHeader>
        <CardContent className="pt-0">
          <p className="text-sm text-muted-foreground line-clamp-2 mb-4">
            {useCase.problem}
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
          </div>
        </CardContent>
      </Card>
    </Link>
  );
}
