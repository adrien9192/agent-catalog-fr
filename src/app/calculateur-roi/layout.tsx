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

function CalculatorJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "SoftwareApplication",
    name: "Calculateur ROI Agent IA",
    applicationCategory: "BusinessApplication",
    operatingSystem: "Web browser",
    description:
      "Calculez gratuitement le retour sur investissement d'un agent IA pour votre entreprise : support client, ventes, RH, finance.",
    offers: {
      "@type": "Offer",
      price: "0",
      priceCurrency: "EUR",
    },
    author: {
      "@type": "Organization",
      name: "AgentCatalog",
      url: "https://agent-catalog-fr.vercel.app",
    },
    inLanguage: "fr-FR",
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function CalculateurROILayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <>
      <CalculatorJsonLd />
      {children}
    </>
  );
}
