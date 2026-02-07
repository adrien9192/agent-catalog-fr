import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Header } from "@/components/header";
import { Footer } from "@/components/footer";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "AgentCatalog — Workflows d'Agents IA prêts à déployer",
    template: "%s | AgentCatalog",
  },
  description:
    "Automatisez votre entreprise avec des workflows d'Agents IA documentés. Tutoriels, stack technique et ROI pour chaque solution. Gratuit.",
  metadataBase: new URL("https://agent-catalog-fr.vercel.app"),
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

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="fr">
      <body className={`${inter.variable} font-sans antialiased`}>
        <div className="flex min-h-svh flex-col">
          <Header />
          <main className="flex-1">{children}</main>
          <Footer />
        </div>
      </body>
    </html>
  );
}
