import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { NewsletterSignup } from "@/components/newsletter-signup";

export const metadata: Metadata = {
  title: "Tarifs — Workflows IA pour votre entreprise",
  description:
    "Accédez gratuitement à nos workflows d'Agents IA documentés ou optez pour un accompagnement sur mesure. Tarifs adaptés au marché français.",
};

const tiers = [
  {
    name: "Découverte",
    price: "Gratuit",
    period: "pour toujours",
    description: "Accédez à tous les workflows documentés et commencez à automatiser.",
    cta: "Commencer gratuitement",
    ctaHref: "/catalogue",
    ctaVariant: "outline" as const,
    highlighted: false,
    features: [
      "Accès aux 10+ workflows documentés",
      "Tutoriels pas-à-pas complets",
      "Stack technique et alternatives low-cost",
      "Schémas d'architecture",
      "Estimation de ROI par workflow",
      "Newsletter quotidienne",
    ],
  },
  {
    name: "Pro",
    price: "299\u00A0\u20AC",
    period: "/mois",
    description: "Pour les équipes qui veulent accélérer leur adoption de l'IA.",
    cta: "Demander un accès",
    ctaHref: "/demande?plan=pro",
    ctaVariant: "default" as const,
    highlighted: true,
    badge: "Populaire",
    features: [
      "Tout le plan Découverte",
      "Workflows sur mesure (2/mois)",
      "Support prioritaire par email",
      "Templates n8n/Make exportables",
      "Accès anticipé aux nouveaux workflows",
      "Audit de vos processus existants",
      "Guide d'implémentation personnalisé",
    ],
  },
  {
    name: "Enterprise",
    price: "Sur devis",
    period: "",
    description: "Accompagnement complet pour transformer vos opérations avec l'IA.",
    cta: "Nous contacter",
    ctaHref: "/demande?plan=enterprise",
    ctaVariant: "outline" as const,
    highlighted: false,
    features: [
      "Tout le plan Pro",
      "Workflows illimités sur mesure",
      "Accompagnement dédié (1 consultant)",
      "Intégration avec vos outils existants",
      "Formation de vos équipes",
      "SLA et support téléphonique",
      "Conformité RGPD & audit sécurité",
      "Dashboard ROI personnalisé",
    ],
  },
];

const faqs = [
  {
    q: "Le plan Découverte est-il vraiment gratuit ?",
    a: "Oui, vous avez accès à l'intégralité des workflows documentés, tutoriels et estimations de ROI sans aucun frais. Aucune carte bancaire requise.",
  },
  {
    q: "Qu'est-ce qu'un workflow sur mesure ?",
    a: "C'est un cas d'usage développé spécifiquement pour votre besoin : tutoriel complet, stack adaptée à vos outils, schéma d'architecture et estimation de ROI sur mesure.",
  },
  {
    q: "Puis-je changer de plan à tout moment ?",
    a: "Oui, vous pouvez upgrader ou downgrader votre plan à tout moment. Pas d'engagement annuel obligatoire.",
  },
  {
    q: "Comment fonctionne le plan Enterprise ?",
    a: "Nous commençons par un audit de vos processus, puis nous développons et déployons des workflows IA adaptés à votre contexte, avec formation et support continu.",
  },
];

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center mb-16">
        <Badge variant="secondary" className="mb-4 text-xs">
          Tarifs simples, sans surprise
        </Badge>
        <h1 className="text-3xl font-bold sm:text-4xl lg:text-5xl">
          Commencez gratuitement,{" "}
          <span className="gradient-text">évoluez à votre rythme</span>
        </h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          Tous nos workflows sont accessibles gratuitement.
          Les plans payants ajoutent l&apos;accompagnement et le sur mesure.
        </p>
      </div>

      {/* Pricing cards */}
      <div className="grid gap-6 lg:grid-cols-3 max-w-5xl mx-auto">
        {tiers.map((tier) => (
          <Card
            key={tier.name}
            className={`relative flex flex-col ${
              tier.highlighted
                ? "border-primary shadow-lg scale-[1.02] lg:scale-105"
                : ""
            }`}
          >
            {tier.badge && (
              <div className="absolute -top-3 left-1/2 -translate-x-1/2">
                <Badge className="px-3 py-1">{tier.badge}</Badge>
              </div>
            )}
            <CardHeader className="pb-4">
              <h3 className="text-lg font-semibold">{tier.name}</h3>
              <div className="flex items-baseline gap-1">
                <span className="text-4xl font-bold">{tier.price}</span>
                {tier.period && (
                  <span className="text-muted-foreground text-sm">{tier.period}</span>
                )}
              </div>
              <p className="text-sm text-muted-foreground mt-2">{tier.description}</p>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col">
              <ul className="space-y-3 flex-1">
                {tier.features.map((feature) => (
                  <li key={feature} className="flex items-start gap-2 text-sm">
                    <svg
                      width="16"
                      height="16"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2.5"
                      className="shrink-0 mt-0.5 text-primary"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    {feature}
                  </li>
                ))}
              </ul>
              <div className="mt-8">
                <Button
                  variant={tier.ctaVariant}
                  className="w-full"
                  size="lg"
                  asChild
                >
                  <Link href={tier.ctaHref}>{tier.cta}</Link>
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* FAQ */}
      <div className="mt-20 max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-10">Questions fréquentes</h2>
        <div className="space-y-6">
          {faqs.map((faq) => (
            <div key={faq.q} className="rounded-xl border p-6">
              <h3 className="font-semibold mb-2">{faq.q}</h3>
              <p className="text-sm text-muted-foreground leading-relaxed">{faq.a}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Newsletter CTA */}
      <div className="mt-16">
        <NewsletterSignup variant="inline" />
      </div>
    </div>
  );
}
