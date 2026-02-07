import { notFound } from "next/navigation";
import Link from "next/link";
import type { Metadata } from "next";
import { UseCaseCard } from "@/components/use-case-card";
import { useCases } from "@/data/use-cases";
import { sectors } from "@/data/sectors";

interface PageProps {
  params: Promise<{ slug: string }>;
}

export async function generateStaticParams() {
  return sectors.map((s) => ({ slug: s.slug }));
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { slug } = await params;
  const sector = sectors.find((s) => s.slug === slug);
  if (!sector) return {};
  return {
    title: `Agents IA pour le secteur ${sector.name}`,
    description: sector.description,
  };
}

export default async function SectorPage({ params }: PageProps) {
  const { slug } = await params;
  const sector = sectors.find((s) => s.slug === slug);
  if (!sector) notFound();

  const sectorName = sector.name;
  const filtered = useCases.filter((uc) =>
    uc.sectors.some((s) => s.toLowerCase().replace(/\s+/g, "-") === slug || s === sectorName)
  );

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <span className="text-foreground">Secteur : {sector.name}</span>
      </nav>

      <header className="mb-8">
        <span className="text-4xl mb-3 block">{sector.icon}</span>
        <h1 className="text-3xl font-bold sm:text-4xl">
          Agents IA — {sector.name}
        </h1>
        <p className="mt-2 text-muted-foreground max-w-2xl">{sector.description}</p>
      </header>

      {filtered.length > 0 ? (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {filtered.map((uc) => (
            <UseCaseCard key={uc.slug} useCase={uc} />
          ))}
        </div>
      ) : (
        <div className="rounded-xl border border-dashed p-12 text-center">
          <p className="text-muted-foreground">
            Aucun cas d&apos;usage disponible pour ce secteur pour le moment.
          </p>
          <Link href="/catalogue" className="mt-2 inline-block text-sm text-primary hover:underline">
            Voir le catalogue complet
          </Link>
        </div>
      )}

      {/* Other sectors */}
      <div className="mt-16 border-t pt-8">
        <h2 className="text-xl font-bold mb-4">Autres secteurs</h2>
        <div className="flex flex-wrap gap-2">
          {sectors
            .filter((s) => s.slug !== slug)
            .slice(0, 8)
            .map((s) => (
              <Link
                key={s.slug}
                href={`/secteur/${s.slug}`}
                className="rounded-lg border px-3 py-2 text-sm transition-colors hover:bg-accent"
              >
                {s.icon} {s.name}
              </Link>
            ))}
        </div>
      </div>
    </div>
  );
}
