import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Calculateur ROI — Estimez les gains de votre Agent IA",
  description:
    "Calculez gratuitement le retour sur investissement d'un agent IA pour votre entreprise. Support client, qualification leads, RH, finance. Estimation personnalisée.",
  openGraph: {
    title: "Calculateur ROI — Estimez les gains de votre Agent IA",
    description:
      "Calculez gratuitement le retour sur investissement d'un agent IA. Estimation personnalisée par fonction.",
  },
};

export default function CalculateurROILayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return children;
}
