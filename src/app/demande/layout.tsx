import type { Metadata } from "next";

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
  return children;
}
