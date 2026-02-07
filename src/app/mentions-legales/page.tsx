import Link from "next/link";
import type { Metadata } from "next";
import { BreadcrumbJsonLd } from "@/components/breadcrumb-json-ld";

export const metadata: Metadata = {
  title: "Mentions légales & Politique de confidentialité — AgentCatalog",
  description:
    "Mentions légales, politique de confidentialité et conditions d'utilisation d'AgentCatalog. Conformité RGPD, hébergement européen.",
  alternates: { canonical: "/mentions-legales" },
};

export default function MentionsLegalesPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6 lg:px-8">
      <BreadcrumbJsonLd
        items={[
          { name: "Accueil", url: "https://agent-catalog-fr.vercel.app" },
          {
            name: "Mentions légales",
            url: "https://agent-catalog-fr.vercel.app/mentions-legales",
          },
        ]}
      />

      <nav className="mb-6 text-sm text-muted-foreground">
        <Link href="/" className="hover:text-foreground">
          Accueil
        </Link>
        {" / "}
        <span className="text-foreground">Mentions légales</span>
      </nav>

      <h1 className="text-3xl font-bold mb-8">
        Mentions légales & Politique de confidentialité
      </h1>

      <div className="prose prose-sm max-w-none space-y-8 text-muted-foreground [&_h2]:text-foreground [&_h2]:text-xl [&_h2]:font-bold [&_h2]:mt-8 [&_h2]:mb-4 [&_h3]:text-foreground [&_h3]:font-semibold [&_h3]:mt-4 [&_h3]:mb-2 [&_p]:leading-relaxed [&_p]:mb-3 [&_ul]:space-y-1 [&_ul]:mb-4 [&_li]:leading-relaxed">
        <section>
          <h2>1. Informations légales</h2>
          <p>
            Le site <strong>agent-catalog-fr.vercel.app</strong> (ci-après
            &laquo;&nbsp;AgentCatalog&nbsp;&raquo;) est un projet personnel
            d&apos;Adrien Laine.
          </p>
          <ul className="list-disc pl-5">
            <li>Contact : adrienlaine91@gmail.com</li>
            <li>Hébergement : Vercel Inc., San Francisco, CA, USA</li>
            <li>
              Les données utilisateurs sont stockées via Brevo (Sendinblue SAS),
              hébergé en Union Européenne
            </li>
          </ul>
        </section>

        <section>
          <h2>2. Politique de confidentialité (RGPD)</h2>

          <h3>2.1 Données collectées</h3>
          <p>AgentCatalog collecte les données suivantes :</p>
          <ul className="list-disc pl-5">
            <li>
              <strong>Newsletter :</strong> adresse email, date
              d&apos;inscription, préférences de secteur/fonction (optionnel)
            </li>
            <li>
              <strong>Demande de workflow :</strong> nom, adresse email,
              description du besoin
            </li>
            <li>
              <strong>Analytics :</strong> données anonymisées de navigation via
              Vercel Analytics (pas de cookies tiers)
            </li>
          </ul>

          <h3>2.2 Finalité du traitement</h3>
          <ul className="list-disc pl-5">
            <li>
              Envoi de la newsletter quotidienne (base légale : consentement)
            </li>
            <li>
              Traitement des demandes de workflow sur mesure (base légale :
              exécution contractuelle)
            </li>
            <li>
              Amélioration du service (base légale : intérêt légitime)
            </li>
          </ul>

          <h3>2.3 Durée de conservation</h3>
          <ul className="list-disc pl-5">
            <li>Données newsletter : jusqu&apos;à désinscription</li>
            <li>
              Données de demande : 24 mois après la dernière interaction
            </li>
            <li>Analytics : 12 mois (données anonymisées)</li>
          </ul>

          <h3>2.4 Sous-traitants</h3>
          <ul className="list-disc pl-5">
            <li>
              <strong>Brevo (Sendinblue SAS)</strong> — gestion des contacts et
              envoi d&apos;emails. Siège social : Paris, France. Données
              hébergées en UE.
            </li>
            <li>
              <strong>Vercel Inc.</strong> — hébergement du site et analytics.
              Conforme au Data Privacy Framework EU-US.
            </li>
          </ul>

          <h3>2.5 Vos droits</h3>
          <p>
            Conformément au RGPD, vous disposez des droits suivants sur vos
            données personnelles :
          </p>
          <ul className="list-disc pl-5">
            <li>Droit d&apos;accès</li>
            <li>Droit de rectification</li>
            <li>Droit à l&apos;effacement (&laquo;&nbsp;droit à l&apos;oubli&nbsp;&raquo;)</li>
            <li>Droit à la portabilité</li>
            <li>Droit d&apos;opposition</li>
            <li>Droit de retrait du consentement</li>
          </ul>
          <p>
            Pour exercer ces droits, contactez-nous à{" "}
            <a
              href="mailto:adrienlaine91@gmail.com"
              className="text-primary hover:underline"
            >
              adrienlaine91@gmail.com
            </a>
            . Nous répondons sous 30 jours.
          </p>

          <h3>2.6 Newsletter — Désinscription</h3>
          <p>
            Vous pouvez vous désinscrire de la newsletter à tout moment en
            cliquant sur le lien de désinscription présent dans chaque email, ou
            en nous contactant directement.
          </p>
        </section>

        <section>
          <h2>3. Cookies</h2>
          <p>
            AgentCatalog n&apos;utilise <strong>aucun cookie tiers</strong> ni
            cookie de tracking publicitaire. Seuls des cookies techniques
            strictement nécessaires au fonctionnement du site peuvent être
            utilisés (aucun consentement requis selon la directive ePrivacy).
          </p>
          <p>
            Vercel Analytics collecte des données de performance anonymisées sans
            déposer de cookies sur votre navigateur.
          </p>
        </section>

        <section>
          <h2>4. Conditions d&apos;utilisation</h2>

          <h3>4.1 Accès au service</h3>
          <p>
            L&apos;accès au catalogue de workflows et aux guides est gratuit et
            sans inscription obligatoire. Les plans payants (Pro, Équipe)
            donnent accès à des fonctionnalités supplémentaires décrites sur la{" "}
            <Link href="/pricing" className="text-primary hover:underline">
              page tarifs
            </Link>
            .
          </p>

          <h3>4.2 Propriété intellectuelle</h3>
          <p>
            Les tutoriels, schémas d&apos;architecture, extraits de code et
            contenus publiés sur AgentCatalog sont mis à disposition à titre
            informatif. Vous êtes libre de les utiliser dans vos projets
            professionnels. La reproduction intégrale du site à des fins
            commerciales est interdite.
          </p>

          <h3>4.3 Limitation de responsabilité</h3>
          <p>
            Les workflows et tutoriels sont fournis &laquo;&nbsp;en
            l&apos;état&nbsp;&raquo;. AgentCatalog ne garantit pas que les
            résultats obtenus correspondent aux estimations de ROI indiquées.
            Les utilisateurs sont responsables de l&apos;adaptation des
            workflows à leur contexte spécifique.
          </p>
        </section>

        <section>
          <h2>5. Modification de cette page</h2>
          <p>
            Cette page peut être mise à jour à tout moment. Dernière
            modification : février 2026.
          </p>
        </section>
      </div>

      <div className="mt-12 rounded-xl border p-6 text-center">
        <p className="text-sm text-muted-foreground">
          Une question sur vos données personnelles ?
        </p>
        <a
          href="mailto:adrienlaine91@gmail.com"
          className="mt-2 inline-block text-sm font-medium text-primary hover:underline"
        >
          Contactez-nous : adrienlaine91@gmail.com
        </a>
      </div>
    </div>
  );
}
