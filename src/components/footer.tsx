import Link from "next/link";
import { NewsletterSignup } from "@/components/newsletter-signup";

const footerLinks = [
  {
    title: "Navigation",
    links: [
      { href: "/", label: "Accueil" },
      { href: "/catalogue", label: "Catalogue" },
    ],
  },
  {
    title: "Fonctions",
    links: [
      { href: "/catalogue?fn=Support", label: "Support" },
      { href: "/catalogue?fn=Sales", label: "Sales" },
      { href: "/catalogue?fn=RH", label: "RH" },
      { href: "/catalogue?fn=Finance", label: "Finance" },
      { href: "/catalogue?fn=IT", label: "IT" },
    ],
  },
  {
    title: "Ressources",
    links: [
      { href: "/catalogue?diff=Facile", label: "Guides Faciles" },
      { href: "/catalogue?diff=Expert", label: "Cas Experts" },
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
                  <li key={link.href}>
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
        <div className="mt-10 border-t border-border/60 pt-6">
          <p className="text-center text-sm text-muted-foreground">
            AgentCatalog — Catalogue de cas d&apos;usage d&apos;Agents IA en entreprise.
          </p>
        </div>
      </div>
    </footer>
  );
}
