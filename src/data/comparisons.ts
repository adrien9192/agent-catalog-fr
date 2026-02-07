export interface ComparisonRow {
  feature: string;
  values: string[];
}

export interface Comparison {
  slug: string;
  title: string;
  metaTitle: string;
  metaDescription: string;
  intro: string;
  options: { name: string; description: string }[];
  rows: ComparisonRow[];
  verdict: string;
  relatedUseCases: string[];
}

export const comparisons: Comparison[] = [
  {
    slug: "agent-ia-vs-chatbot",
    title: "Agent IA vs Chatbot : quelle différence ?",
    metaTitle: "Agent IA vs Chatbot — Comparatif Complet 2025 | AgentCatalog",
    metaDescription:
      "Quelle différence entre un agent IA et un chatbot classique ? Capacités, coûts, cas d'usage, ROI. Comparatif détaillé pour choisir la bonne solution.",
    intro:
      "Chatbot ou Agent IA ? Ces deux termes sont souvent confondus, mais ils désignent des technologies très différentes en termes de capacités, d'autonomie et de valeur ajoutée. Ce comparatif vous aide à choisir la bonne solution pour votre entreprise.",
    options: [
      { name: "Chatbot classique", description: "Programme de conversation basé sur des règles prédéfinies ou un arbre de décision. Répond à des questions simples." },
      { name: "Agent IA", description: "Programme autonome utilisant l'IA générative, capable de raisonner, d'exécuter des actions et de s'adapter au contexte." },
    ],
    rows: [
      { feature: "Autonomie", values: ["Suit un script prédéfini", "Raisonne et prend des décisions autonomes"] },
      { feature: "Compréhension du langage", values: ["Mots-clés / intentions limitées", "Compréhension contextuelle complète (LLM)"] },
      { feature: "Actions possibles", values: ["Réponse textuelle uniquement", "Lecture/écriture dans CRM, email, bases de données"] },
      { feature: "Personnalisation", values: ["Réponses génériques", "Réponses adaptées au contexte et à l'historique"] },
      { feature: "Gestion des cas complexes", values: ["Escalade vers un humain", "Résolution autonome + escalade intelligente"] },
      { feature: "Apprentissage", values: ["Mise à jour manuelle des règles", "Amélioration continue via feedback"] },
      { feature: "Temps de mise en place", values: ["1-2 semaines", "3-10 jours avec les bons outils"] },
      { feature: "Coût mensuel moyen", values: ["50-200 €/mois", "200-500 €/mois (API + orchestration)"] },
      { feature: "ROI typique", values: ["10-20% de tickets déviés", "40-60% de tâches automatisées"] },
      { feature: "Cas d'usage idéal", values: ["FAQ, horaires, statut commande", "Triage support, qualification leads, analyse documents"] },
    ],
    verdict:
      "Le chatbot reste pertinent pour des cas simples (FAQ, navigation) avec un budget limité. L'agent IA est le choix adapté dès que vous avez besoin d'automatiser des tâches complexes qui nécessitent du raisonnement, l'accès à vos outils métier, ou une personnalisation avancée. Pour la plupart des entreprises en 2025, l'agent IA offre un ROI nettement supérieur grâce à sa capacité à traiter des cas que le chatbot escalade systématiquement.",
    relatedUseCases: ["agent-triage-support-client", "agent-qualification-leads", "agent-knowledge-management"],
  },
  {
    slug: "n8n-vs-make-vs-zapier",
    title: "n8n vs Make vs Zapier : quel orchestrateur pour vos agents IA ?",
    metaTitle: "n8n vs Make vs Zapier — Comparatif Automatisation IA 2025 | AgentCatalog",
    metaDescription:
      "Comparatif complet n8n vs Make vs Zapier pour l'orchestration d'agents IA. Prix, fonctionnalités, intégrations, hébergement. Guide pour choisir le bon outil.",
    intro:
      "L'orchestration est le cœur de tout workflow d'agent IA. Ces trois plateformes permettent de connecter vos outils et d'automatiser des processus, mais avec des approches très différentes. Voici comment choisir.",
    options: [
      { name: "n8n", description: "Plateforme open-source d'automatisation. Self-hosted ou cloud. Interface visuelle par nœuds." },
      { name: "Make", description: "Plateforme SaaS d'automatisation visuelle (ex-Integromat). Interface par scénarios." },
      { name: "Zapier", description: "Plateforme SaaS la plus populaire. Interface simplifiée par Zaps (trigger → action)." },
    ],
    rows: [
      { feature: "Prix (démarrage)", values: ["Gratuit (self-hosted) / 20 €/mois (cloud)", "9 €/mois (1 000 ops)", "19,99 $/mois (750 tâches)"] },
      { feature: "Open-source", values: ["Oui (licence fair-code)", "Non", "Non"] },
      { feature: "Self-hosting", values: ["Oui (Docker, Kubernetes)", "Non", "Non"] },
      { feature: "Intégrations natives", values: ["400+", "1 500+", "6 000+"] },
      { feature: "Nœuds IA natifs", values: ["OpenAI, Anthropic, Ollama, LangChain", "OpenAI, Anthropic", "OpenAI (limité)"] },
      { feature: "Complexité des workflows", values: ["Illimitée (boucles, branches, sous-workflows)", "Avancée (routeurs, itérateurs)", "Basique (linéaire, quelques branches)"] },
      { feature: "Gestion des erreurs", values: ["Complète (retry, fallback, logs)", "Bonne (retry, break)", "Basique (retry)"] },
      { feature: "Exécution de code", values: ["JavaScript, Python natif", "JavaScript, Python (limité)", "JavaScript (limité)"] },
      { feature: "RGPD / Hébergement EU", values: ["Self-hosted = contrôle total", "Serveurs EU disponibles", "Serveurs US (EU sur demande)"] },
      { feature: "Courbe d'apprentissage", values: ["Moyenne (technique)", "Moyenne (visuelle)", "Faible (simplifié)"] },
      { feature: "Cas d'usage IA idéal", values: ["Agents IA complexes, multi-étapes", "Automatisations IA moyennes", "Automatisations simples avec IA"] },
    ],
    verdict:
      "Pour des agents IA en entreprise française, n8n est notre recommandation principale : open-source, self-hostable (RGPD), nœuds IA natifs puissants et workflows complexes sans limitation. Make est un excellent choix intermédiaire pour les équipes qui préfèrent le SaaS managé avec une bonne flexibilité. Zapier convient aux automatisations simples mais montre rapidement ses limites pour les workflows d'agents IA avancés.",
    relatedUseCases: ["agent-triage-support-client", "agent-gestion-incidents-it", "agent-redaction-contenu-marketing"],
  },
  {
    slug: "agent-ia-vs-consultant",
    title: "Agent IA vs Consultant externe : coût, vitesse, résultats",
    metaTitle: "Agent IA vs Consultant Externe — Comparatif Coût & ROI 2025 | AgentCatalog",
    metaDescription:
      "Faut-il embaucher un consultant ou déployer un agent IA ? Comparatif complet : coûts, délais, scalabilité, maintenance. Calculez le ROI de chaque option.",
    intro:
      "Quand une entreprise veut automatiser un processus, elle hésite souvent entre recruter un consultant (ou une ESN) et déployer un agent IA. Les deux approches ont des avantages, mais le rapport coût/efficacité est radicalement différent.",
    options: [
      { name: "Consultant / ESN", description: "Expertise humaine externalisée. Audit, recommandations, implémentation sur mesure. Facturation au jour/homme." },
      { name: "Agent IA (workflow)", description: "Solution technologique autonome. Déploiement via workflow documenté. Coût fixe mensuel." },
    ],
    rows: [
      { feature: "Coût initial", values: ["5 000 - 50 000 € (audit + implémentation)", "0 - 500 € (outils + configuration)"] },
      { feature: "Coût récurrent", values: ["1 000 - 5 000 €/mois (maintenance)", "200 - 500 €/mois (API + hébergement)"] },
      { feature: "Délai de déploiement", values: ["2 - 6 mois", "3 - 10 jours"] },
      { feature: "Scalabilité", values: ["Linéaire (plus de volume = plus de consultants)", "Quasi-gratuite (même agent, plus de volume)"] },
      { feature: "Disponibilité", values: ["Heures ouvrées", "24/7"] },
      { feature: "Personnalisation", values: ["Sur mesure total", "Configurable via prompts et règles"] },
      { feature: "Transfert de compétences", values: ["Dépend du consultant", "Documentation incluse dans chaque workflow"] },
      { feature: "Dépendance", values: ["Forte (expertise du consultant)", "Faible (workflow documenté et reproductible)"] },
      { feature: "Gestion du changement", values: ["Incluse (accompagnement humain)", "À gérer en interne (guides disponibles)"] },
      { feature: "ROI à 12 mois", values: ["Variable (dépend du livrable)", "5x - 25x (automatisation récurrente)"] },
    ],
    verdict:
      "L'agent IA est le choix optimal pour les processus répétitifs et scalables : support client, qualification de leads, traitement de factures, tri de CV. Le consultant reste pertinent pour les projets stratégiques uniques (transformation digitale, choix d'architecture, conduite du changement) ou quand l'expertise métier spécifique est critique. La meilleure approche combine souvent les deux : un consultant pour la stratégie initiale, puis des agents IA pour l'exécution quotidienne.",
    relatedUseCases: ["agent-triage-support-client", "agent-qualification-leads", "agent-rapports-financiers"],
  },
];
