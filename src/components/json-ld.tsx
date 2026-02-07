import type { UseCase } from "@/data/types";

interface JsonLdProps {
  useCase: UseCase;
  siteUrl?: string;
}

export function UseCaseJsonLd({
  useCase: uc,
  siteUrl = "https://agent-catalog-fr.vercel.app",
}: JsonLdProps) {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "HowTo",
    name: uc.title,
    description: uc.subtitle,
    step: uc.tutorial.map((section, i) => ({
      "@type": "HowToStep",
      position: i + 1,
      name: section.title,
      text: section.content,
    })),
    tool: uc.recommendedStack.map((item) => ({
      "@type": "HowToTool",
      name: item.name,
    })),
    totalTime: uc.difficulty === "Facile" ? "PT2H" : uc.difficulty === "Moyen" ? "PT8H" : "PT24H",
    inLanguage: "fr",
    url: `${siteUrl}/use-case/${uc.slug}`,
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}
