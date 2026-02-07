import Link from "next/link";
import type { Metadata } from "next";
import { Badge } from "@/components/ui/badge";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";
import { NewsletterSignup } from "@/components/newsletter-signup";
import { useCases } from "@/data/use-cases";
import { guides } from "@/data/guides";
import { comparisons } from "@/data/comparisons";

export const metadata: Metadata = {
  title: "À propos — L'équipe derrière AgentCatalog",
  description:
    "Découvrez l'équipe AgentCatalog. Experts en IA appliquée, automatisation et stratégie digitale pour les entreprises françaises. 10+ ans d'expérience en B2B SaaS.",
  alternates: { canonical: "/a-propos" },
};

const expertise = [
  {
    area: "IA & Machine Learning",
    description:
      "Conception et déploiement d'agents IA en production. Prompt engineering, RAG, fine-tuning de LLMs.",
    years: "8+",
  },
  {
    area: "Automatisation B2B",
    description:
      "Intégration de workflows n8n, Make et Zapier pour des entreprises de 10 à 10 000 collaborateurs.",
    years: "10+",
  },
  {
    area: "Stratégie Digitale",
    description:
      "Accompagnement de PME et ETI françaises dans leur transformation digitale et adoption de l'IA.",
    years: "12+",
  },
  {
    area: "Conformité & RGPD",
    description:
      "Déploiement d'IA conforme aux réglementations européennes. Audit de conformité et bonnes pratiques.",
    years: "6+",
  },
];

const values = [
  {
    title: "Transparence",
    description:
      "Chaque workflow est documenté avec son coût réel, ses limites et ses alternatives gratuites. Pas de marketing creux.",
  },
  {
    title: "Pragmatisme",
    description:
      "Nous ne recommandons que ce qui fonctionne en production. Chaque tutoriel a été testé et validé avant publication.",
  },
  {
    title: "Accessibilité",
    description:
      "L'IA ne doit pas être réservée aux entreprises du CAC 40. Nos alternatives low-cost permettent à toute PME de démarrer.",
  },
  {
    title: "Conformité",
    description:
      "Chaque workflow intègre les considérations RGPD, hébergement européen et bonnes pratiques de sécurité dès la conception.",
  },
];

function AboutJsonLd() {
  const jsonLd = {
    "@context": "https://schema.org",
    "@type": "AboutPage",
    name: "À propos d'AgentCatalog",
    description:
      "Experts en IA appliquée et automatisation pour les entreprises françaises.",
    url: "https://agent-catalog-fr.vercel.app/a-propos",
    mainEntity: {
      "@type": "Organization",
      name: "AgentCatalog",
      url: "https://agent-catalog-fr.vercel.app",
      foundingDate: "2025",
      areaServed: { "@type": "Country", name: "France" },
      knowsAbout: [
        "Intelligence Artificielle",
        "Agents IA",
        "Automatisation",
        "n8n",
        "Make",
        "LLM",
        "RGPD",
      ],
    },
  };

  return (
    <script
      type="application/ld+json"
      dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
    />
  );
}

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 lg:px-8">
      <AboutJsonLd />
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          { name: "À propos", url: "https://agent-catalog-fr.vercel.app/a-propos" },
        ]}
      />

      {/* Breadcrumb */}
      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">Accueil</Link>
        {" / "}
        <span className="text-foreground">À propos</span>
      </nav>

      {/* Hero */}
      <div className="text-center mb-16">
        <Badge variant="secondary" className="mb-4 text-xs">
          À propos
        </Badge>
        <h1 className="text-3xl font-bold sm:text-4xl lg:text-5xl">
          Des experts IA au service{" "}
          <span className="gradient-text">de votre entreprise</span>
        </h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          AgentCatalog est né d&apos;un constat simple : les entreprises
          françaises ont besoin de ressources concrètes et actionnables pour
          déployer l&apos;IA, pas de buzzwords.
        </p>
      </div>

      {/* Mission */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-4">Notre mission</h2>
        <div className="rounded-xl border p-6 sm:p-8 bg-muted/30">
          <p className="text-muted-foreground leading-relaxed">
            Démocratiser l&apos;accès aux workflows d&apos;Agents IA pour les
            entreprises françaises. Nous documentons, testons et publions des
            cas d&apos;usage concrets — avec le code, le schéma
            d&apos;architecture, la stack technique et l&apos;estimation de ROI.
            Tout est pensé pour être déployé en quelques heures, pas en quelques
            mois.
          </p>
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-primary">{useCases.length}</p>
              <p className="text-xs text-muted-foreground">Workflows documentés</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary">{guides.length}</p>
              <p className="text-xs text-muted-foreground">Guides pratiques</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary">{comparisons.length}</p>
              <p className="text-xs text-muted-foreground">Comparatifs objectifs</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary">100%</p>
              <p className="text-xs text-muted-foreground">Gratuit et accessible</p>
            </div>
          </div>
        </div>
      </section>

      {/* Expertise */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-6">Notre expertise</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {expertise.map((e) => (
            <div key={e.area} className="rounded-xl border p-5">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold text-sm">{e.area}</h3>
                <Badge variant="outline" className="text-xs">
                  {e.years} ans
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">{e.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Values */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-6">Nos valeurs</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {values.map((v) => (
            <div key={v.title} className="rounded-xl border p-5">
              <h3 className="font-semibold text-sm mb-2">{v.title}</h3>
              <p className="text-sm text-muted-foreground">{v.description}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Methodology */}
      <section className="mb-16">
        <h2 className="text-2xl font-bold mb-6">Notre méthodologie</h2>
        <div className="rounded-xl border p-6 sm:p-8">
          <ol className="space-y-6">
            {[
              {
                step: "1",
                title: "Recherche et veille",
                desc: "Nous analysons les dernières avancées en IA appliquée, les retours d'expérience d'entreprises françaises et les benchmarks de performance.",
              },
              {
                step: "2",
                title: "Conception et test",
                desc: "Chaque workflow est conçu, codé et testé en conditions réelles avant publication. Nous vérifions la faisabilité technique et le ROI annoncé.",
              },
              {
                step: "3",
                title: "Documentation complète",
                desc: "Tutoriel pas-à-pas, code fonctionnel, schéma d'architecture, stack recommandée, alternatives gratuites et estimation de ROI chiffrée.",
              },
              {
                step: "4",
                title: "Mise à jour continue",
                desc: "Les workflows sont mis à jour régulièrement pour refléter les évolutions des outils (nouvelles versions de n8n, Make, Claude, GPT).",
              },
            ].map((item) => (
              <li key={item.step} className="flex gap-4">
                <span className="shrink-0 flex h-8 w-8 items-center justify-center rounded-full bg-primary text-sm font-bold text-primary-foreground">
                  {item.step}
                </span>
                <div>
                  <h3 className="font-semibold text-sm">{item.title}</h3>
                  <p className="mt-1 text-sm text-muted-foreground">{item.desc}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* CTA */}
      <div className="grid gap-4 sm:grid-cols-3 mb-12">
        <Link
          href="/catalogue"
          className="group rounded-xl border p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Explorer les workflows
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {useCases.length} cas d&apos;usage documentés, prêts à déployer.
          </p>
        </Link>
        <Link
          href="/guide"
          className="group rounded-xl border p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Lire nos guides
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            {guides.length} guides pratiques pour maîtriser l&apos;IA en entreprise.
          </p>
        </Link>
        <Link
          href="/calculateur-roi"
          className="group rounded-xl border p-5 transition-all hover:shadow-sm hover:border-primary/30"
        >
          <h3 className="font-semibold text-sm group-hover:text-primary transition-colors">
            Calculer votre ROI
          </h3>
          <p className="mt-1 text-xs text-muted-foreground">
            Estimez vos gains avec notre outil gratuit.
          </p>
        </Link>
      </div>

      {/* Newsletter */}
      <NewsletterSignup variant="inline" />
    </div>
  );
}
