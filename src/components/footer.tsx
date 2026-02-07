import Link from "next/link";
import { NewsletterSignup } from "@/components/newsletter-signup";

const footerLinks = [
  {
    title: "Produit",
    links: [
      { href: "/catalogue", label: "Tous les workflows" },
      { href: "/pricing", label: "Tarifs" },
      { href: "/demande", label: "Workflow sur mesure" },
    ],
  },
  {
    title: "Par fonction",
    links: [
      { href: "/catalogue?fn=Support", label: "Support client" },
      { href: "/catalogue?fn=Sales", label: "Sales & CRM" },
      { href: "/catalogue?fn=RH", label: "Ressources humaines" },
      { href: "/catalogue?fn=Finance", label: "Finance" },
      { href: "/catalogue?fn=Marketing", label: "Marketing" },
      { href: "/catalogue?fn=IT", label: "IT & DevOps" },
    ],
  },
  {
    title: "Ressources",
    links: [
      { href: "/catalogue?diff=Facile", label: "Workflows faciles" },
      { href: "/catalogue?diff=Moyen", label: "Workflows intermédiaires" },
      { href: "/catalogue?diff=Expert", label: "Workflows experts" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="border-t border-border/60 bg-muted/30">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-4">
          {footerLinks.map((section) => (
            <div key={section.title}>
              <h3 className="text-sm font-semibold text-foreground">{section.title}</h3>
              <ul className="mt-3 space-y-2">
                {section.links.map((link) => (
                  <li key={link.href + link.label}>
                    <Link
                      href={link.href}
                      className="text-sm text-muted-foreground transition-colors hover:text-foreground"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
          <div>
            <NewsletterSignup variant="footer" />
          </div>
        </div>
        <div className="mt-10 border-t border-border/60 pt-6 space-y-4">
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
              Conforme RGPD
            </span>
            <span>|</span>
            <span>Données hébergées en Europe</span>
            <span>|</span>
            <a href="mailto:adrienlaine91@gmail.com" className="hover:text-foreground transition-colors">
              Contact : adrienlaine91@gmail.com
            </a>
          </div>
          <div className="flex flex-col sm:flex-row items-center justify-between gap-2">
            <p className="text-sm text-muted-foreground">
              AgentCatalog — Workflows d&apos;Agents IA pour l&apos;entreprise.
            </p>
            <p className="text-xs text-muted-foreground">
              &copy; {new Date().getFullYear()} AgentCatalog. Tous droits réservés.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
