import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Catalogue — Workflows IA pour l'entreprise",
  description:
    "Parcourez nos 20+ workflows d'Agents IA documentés. Filtrez par fonction, secteur et difficulté. Tutoriels, stack technique et ROI inclus. Gratuit.",
  openGraph: {
    title: "Catalogue — Workflows IA pour l'entreprise",
    description:
      "Parcourez nos 20+ workflows d'Agents IA documentés. Filtrez par fonction, secteur et difficulté.",
  },
};

export default function CatalogueLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
