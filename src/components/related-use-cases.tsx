import { useCases } from "@/data/use-cases";
import { UseCaseCard } from "@/components/use-case-card";
import type { UseCase } from "@/data/types";

interface RelatedUseCasesProps {
  currentSlug: string;
  functions: string[];
  sectors: string[];
  maxItems?: number;
}

export function RelatedUseCases({
  currentSlug,
  functions,
  sectors,
  maxItems = 3,
}: RelatedUseCasesProps) {
  const scored = useCases
    .filter((uc) => uc.slug !== currentSlug)
    .map((uc) => {
      let score = 0;
      // Shared functions = strong signal
      for (const fn of functions) {
        if (uc.functions.includes(fn)) score += 3;
      }
      // Shared sectors = moderate signal
      for (const s of sectors) {
        if (uc.sectors.includes(s)) score += 1;
      }
      return { uc, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, maxItems);

  if (scored.length === 0) return null;

  return (
    <section className="mt-12 border-t pt-10">
      <h2 className="text-2xl font-bold mb-6">Cas d&apos;usage similaires</h2>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {scored.map(({ uc }: { uc: UseCase }) => (
          <UseCaseCard key={uc.slug} useCase={uc} />
        ))}
      </div>
    </section>
  );
}
