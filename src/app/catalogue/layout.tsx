import type { Metadata } from "next";
import { useCases } from "@/data/use-cases";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";

export const metadata: Metadata = {
  title: "Catalogue — Workflows IA pour l'entreprise",
  description:
    `Parcourez nos ${useCases.length} workflows d'Agents IA documentés. Filtrez par fonction, secteur et difficulté. Tutoriels, stack technique et ROI inclus. Gratuit.`,
  alternates: {
    canonical: "/catalogue",
  },
  openGraph: {
    title: "Catalogue — Workflows IA pour l'entreprise",
    description:
      `Parcourez nos ${useCases.length} workflows d'Agents IA documentés. Filtrez par fonction, secteur et difficulté.`,
  },
};

function CatalogueJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "CollectionPage",
    name: "Catalogue de workflows IA",
    description: `${useCases.length} workflows d'Agents IA documentés avec tutoriel, stack technique et estimation de ROI.`,
    url: "https://agent-catalog-fr.vercel.app/catalogue",
    numberOfItems: useCases.length,
    mainEntity: {
      "@type": "ItemList",
      numberOfItems: useCases.length,
      itemListElement: useCases.slice(0, 10).map((uc, i) => ({
        "@type": "ListItem",
        position: i + 1,
        url: `https://agent-catalog-fr.vercel.app/use-case/${uc.slug}`,
        name: uc.title,
      })),
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function CatalogueLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <CatalogueJsonLd />
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: "Catalogue", url: "https://agent-catalog-fr.vercel.app/catalogue" },
        ]}
      />
      {children}
    </>
  );
}
