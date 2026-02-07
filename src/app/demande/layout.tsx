import type { Metadata } from "next";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";

export const metadata: Metadata = {
  title: "Demander un workflow sur mesure — AgentCatalog",
  description:
    "Décrivez votre besoin d'automatisation et recevez un workflow d'Agent IA sur mesure avec tutoriel complet, stack technique et estimation de ROI. Réponse sous 48h.",
  openGraph: {
    title: "Demander un workflow sur mesure — AgentCatalog",
    description:
      "Décrivez votre besoin d'automatisation et recevez un workflow d'Agent IA sur mesure. Réponse sous 48h.",
  },
};

export default function DemandeLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: "Catalogue", url: "https://agent-catalog-fr.vercel.app/catalogue" },
          { name: "Demander un workflow", url: "https://agent-catalog-fr.vercel.app/demande" },
        ]}
      />
      {children}
    </>
  );
}
