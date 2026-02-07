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
        <div className="mt-10 border-t border-border/60 pt-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <p className="text-sm text-muted-foreground">
            AgentCatalog — Workflows d&apos;Agents IA pour l&apos;entreprise.
          </p>
          <p className="text-xs text-muted-foreground">
            &copy; {new Date().getFullYear()} AgentCatalog. Tous droits réservés.
          </p>
        </div>
      </div>
    </footer>
  );
}
