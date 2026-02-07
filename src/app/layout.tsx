import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";
import "./globals.css";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  viewportFit: "cover",
};

export const metadata: Metadata = {
  title: {
    default: "AgentCatalog — Workflows d'Agents IA prêts à déployer",
    template: "%s | AgentCatalog",
  },
  description:
    "Automatisez votre entreprise avec des workflows d'Agents IA documentés. Tutoriels, stack technique et ROI pour chaque solution. Gratuit.",
  keywords: [
    "agent IA",
    "workflow IA",
    "automatisation entreprise",
    "intelligence artificielle",
    "n8n",
    "Make",
    "support client IA",
    "qualification leads IA",
    "RGPD",
    "tutoriel IA",
  ],
  metadataBase: new URL("https://agent-catalog-fr.vercel.app"),
  alternates: {
    canonical: "/",
  },
  openGraph: {
    locale: "fr_FR",
    type: "website",
    siteName: "AgentCatalog",
    url: "https://agent-catalog-fr.vercel.app",
    description:
      "Automatisez votre entreprise avec des workflows d'Agents IA documentés. Tutoriels, stack technique et ROI pour chaque solution. Gratuit.",
  },
  twitter: {
    card: "summary_large_image",
  },
};

function OrganizationJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "Organization",
    name: "AgentCatalog",
    url: "https://agent-catalog-fr.vercel.app",
    description:
      "Workflows d'Agents IA documentés pour l'entreprise française. Tutoriels, stack technique et ROI.",
    foundingDate: "2025",
    areaServed: "FR",
    knowsLanguage: "fr",
    sameAs: ["https://github.com/adrien9192/agent-catalog-fr"],
    contactPoint: {
      "@type": "ContactPoint",
      email: "adrienlaine91@gmail.com",
      contactType: "customer service",
      availableLanguage: "French",
    },
    offers: {
      "@type": "Offer",
      description: "Workflows IA documentés gratuits avec tutoriels et ROI",
      price: "0",
      priceCurrency: "EUR",
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className={`${inter.variable} font-sans antialiased`}>
        <OrganizationJsonLd />
        <div className="flex min-h-svh flex-col">
          <Header />
          <main className="flex-1">{children}</main>
          <Footer />
        </div>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
