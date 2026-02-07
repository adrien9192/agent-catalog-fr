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
    cta: "Accès immédiat gratuit",
    ctaHref: "/catalogue",
    ctaVariant: "outline" as const,
    highlighted: false,
    features: [
      "Accès aux 30+ workflows documentés",
      "Tutoriels pas-à-pas complets",
      "Stack technique et alternatives low-cost",
      "Schémas d'architecture",
      "Estimation de ROI par workflow",
      "Newsletter quotidienne",
    ],
  },
  {
    name: "Pro",
    price: "29\u00A0\u20AC",
    period: "/mois",
    description: "Pour les équipes qui veulent accélérer leur adoption de l'IA.",
    cta: "Essai gratuit 14 jours",
    ctaHref: "/demande?plan=pro",
    ctaVariant: "default" as const,
    highlighted: true,
    badge: "Le plus populaire",
    subtext: "Sans carte bancaire. Annulation en un clic.",
    features: [
      "Tout le plan Découverte",
      "Workflows sur mesure (1/mois)",
      "Templates n8n/Make exportables",
      "Support prioritaire par email",
      "Accès anticipé aux nouveaux workflows",
      "Guide d'implémentation personnalisé",
    ],
  },
  {
    name: "Équipe",
    price: "99\u00A0\u20AC",
    period: "/mois",
    description: "Pour les équipes qui déploient l'IA à grande échelle.",
    cta: "Planifier un appel",
    ctaHref: "/demande?plan=equipe",
    ctaVariant: "outline" as const,
    highlighted: false,
    features: [
      "Tout le plan Pro",
      "Workflows sur mesure (3/mois)",
      "Audit de vos processus existants",
      "Accompagnement à l'implémentation",
      "Formation de vos équipes",
      "Conformité RGPD & bonnes pratiques",
      "Support Slack dédié",
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
    a: "Oui, vous pouvez upgrader ou downgrader à tout moment. Pas d'engagement annuel obligatoire. Satisfait ou remboursé sous 14 jours.",
  },
  {
    q: "Quel plan choisir pour commencer ?",
    a: "Commencez par le plan Découverte (gratuit) pour explorer les workflows existants. Passez au Pro quand vous avez besoin d'un workflow adapté à votre contexte spécifique.",
  },
  {
    q: "Combien de temps pour recevoir un workflow sur mesure ?",
    a: "Les workflows sur mesure sont livrés sous 5 jours ouvrés. Vous recevrez un email dès que votre workflow est en ligne, avec le tutoriel complet.",
  },
  {
    q: "Mes données sont-elles sécurisées ?",
    a: "Oui. Hébergement en Europe, conformité RGPD native, chiffrement des données en transit et au repos. Nous ne stockons aucune donnée métier de nos clients.",
  },
  {
    q: "Quel support technique est inclus ?",
    a: "Plan Découverte : accès à la documentation et guides. Plan Pro : support par email avec réponse sous 24h ouvrées. Plan Équipe : canal Slack dédié avec réponse prioritaire.",
  },
  {
    q: "Puis-je annuler à tout moment ?",
    a: "Oui, aucun engagement. Vous pouvez annuler votre abonnement en un clic depuis votre espace. Les workflows déjà reçus restent accessibles.",
  },
];

function PricingFaqJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: faqs.map((faq) => ({
      "@type": "Question",
      name: faq.q,
      acceptedAnswer: {
        "@type": "Answer",
        text: faq.a,
      },
    })),
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function PricingPage() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <PricingFaqJsonLd />
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
                ? "border-primary shadow-lg lg:scale-105"
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
                {tier.subtext && (
                  <p className="mt-2 text-xs text-center text-muted-foreground">{tier.subtext}</p>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Trust / Guarantee */}
      <div className="mt-12 flex flex-wrap items-center justify-center gap-x-6 gap-y-3 text-xs sm:text-sm text-muted-foreground">
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          </svg>
          <span>Satisfait ou remboursé 14 jours</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
          <span>Sans engagement</span>
        </div>
        <div className="flex items-center gap-2">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <rect x="1" y="4" width="22" height="16" rx="2"/>
            <line x1="1" y1="10" x2="23" y2="10"/>
          </svg>
          <span>Paiement sécurisé</span>
        </div>
      </div>

      {/* Comparison table */}
      <div className="mt-20 max-w-4xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-8">Comparatif détaillé</h2>
        <div className="overflow-x-auto -mx-4 px-4 sm:mx-0 sm:px-0">
          <table className="w-full text-sm min-w-[500px]">
            <thead>
              <tr className="border-b">
                <th className="py-3 px-2 sm:px-4 text-left font-medium text-muted-foreground text-xs sm:text-sm">Fonctionnalité</th>
                <th className="py-3 px-2 sm:px-4 text-center font-medium text-xs sm:text-sm">Découverte</th>
                <th className="py-3 px-2 sm:px-4 text-center font-medium text-primary text-xs sm:text-sm">Pro</th>
                <th className="py-3 px-2 sm:px-4 text-center font-medium text-xs sm:text-sm">Équipe</th>
              </tr>
            </thead>
            <tbody className="divide-y">
              {[
                { feature: "Workflows documentés", free: "30+", pro: "30+", team: "30+" },
                { feature: "Tutoriels pas-à-pas", free: "check", pro: "check", team: "check" },
                { feature: "Estimation de ROI", free: "check", pro: "check", team: "check" },
                { feature: "Newsletter quotidienne", free: "check", pro: "check", team: "check" },
                { feature: "Workflows sur mesure", free: "cross", pro: "1/mois", team: "3/mois" },
                { feature: "Templates n8n/Make", free: "cross", pro: "check", team: "check" },
                { feature: "Support prioritaire", free: "cross", pro: "Email", team: "Slack dédié" },
                { feature: "Accès anticipé", free: "cross", pro: "check", team: "check" },
                { feature: "Audit processus", free: "cross", pro: "cross", team: "check" },
                { feature: "Formation équipe", free: "cross", pro: "cross", team: "check" },
                { feature: "Accompagnement RGPD", free: "cross", pro: "cross", team: "check" },
              ].map((row) => (
                <tr key={row.feature}>
                  <td className="py-3 px-2 sm:px-4 font-medium text-xs sm:text-sm">{row.feature}</td>
                  {[row.free, row.pro, row.team].map((val, i) => (
                    <td key={i} className="py-3 px-2 sm:px-4 text-center text-xs sm:text-sm">
                      {val === "check" ? (
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="inline text-primary">
                          <polyline points="20 6 9 17 4 12" />
                        </svg>
                      ) : val === "cross" ? (
                        <span className="text-muted-foreground/40">—</span>
                      ) : (
                        <span className={i === 1 ? "font-medium text-primary" : ""}>{val}</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* FAQ */}
      <div className="mt-20 max-w-3xl mx-auto">
        <h2 className="text-2xl font-bold text-center mb-10">Questions fréquentes</h2>
        <div className="space-y-6">
          {faqs.map((faq) => (
            <div key={faq.q} className="rounded-xl border p-4 sm:p-6">
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
