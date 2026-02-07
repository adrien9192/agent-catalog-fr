import { UseCase } from "./types";

export const useCases: UseCase[] = [
  {
    slug: "agent-triage-support-client",
    title: "Agent de Triage Support Client",
    subtitle: "Classifiez et routez automatiquement les tickets de support grâce à l'IA",
    problem:
      "Les équipes support sont submergées par un volume croissant de tickets. Le triage manuel est lent, sujet aux erreurs de classification, et retarde la résolution des demandes critiques.",
    value:
      "Un agent IA analyse chaque ticket entrant, le classifie par catégorie et urgence, puis le route automatiquement vers l'équipe compétente. Le temps de première réponse chute drastiquement.",
    inputs: [
      "Contenu du ticket (texte, email, chat)",
      "Historique client (CRM)",
      "Base de connaissances interne",
      "Règles de routage métier",
    ],
    outputs: [
      "Catégorie du ticket (technique, facturation, etc.)",
      "Niveau d'urgence (P1-P4)",
      "Équipe assignée",
      "Suggestion de réponse pré-rédigée",
      "Score de confiance de la classification",
    ],
    risks: [
      "Mauvaise classification entraînant des SLA manqués",
      "Biais dans la priorisation des tickets",
      "Dépendance au LLM pour des décisions sensibles",
    ],
    roiIndicatif:
      "Réduction de 60% du temps de triage. Amélioration de 25% du taux de résolution au premier contact.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Pinecone", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Ticket    │────▶│  Agent LLM   │────▶│  Routage    │
│   entrant   │     │  (Classif.)  │     │  automatique│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Vector DB   │
                    │  (KB interne)│
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires et configurez vos clés API. Vous aurez besoin d'un compte OpenAI et d'une instance Pinecone (free tier suffisant pour le MVP).",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain openai pinecone-client python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=support-kb`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Indexation de la base de connaissances",
        content:
          "Créez un index vectoriel de votre base de connaissances interne. Cela permettra à l'agent de trouver les articles pertinents pour chaque ticket et d'améliorer la classification.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader

# Charger les documents de la KB
loader = DirectoryLoader("./kb_docs", glob="**/*.md")
docs = loader.load()

# Créer l'index vectoriel
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(
    docs, embeddings, index_name="support-kb"
)
print(f"{len(docs)} documents indexés.")`,
            filename: "index_kb.py",
          },
        ],
      },
      {
        title: "Agent de classification",
        content:
          "Construisez l'agent qui analyse le contenu du ticket, le compare à la KB, et produit une classification structurée avec catégorie, urgence et équipe cible.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class TicketClassification(BaseModel):
    category: str = Field(description="Catégorie: technique, facturation, commercial, autre")
    urgency: str = Field(description="Urgence: P1, P2, P3, P4")
    team: str = Field(description="Équipe cible")
    suggested_response: str = Field(description="Suggestion de réponse")
    confidence: float = Field(description="Score de confiance 0-1")

parser = PydanticOutputParser(pydantic_object=TicketClassification)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Tu es un agent de triage support client.
Analyse le ticket et classifie-le. Contexte KB: {context}
{format_instructions}"""),
    ("user", "{ticket_content}")
])

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
chain = prompt | llm | parser

def classify_ticket(content: str, context: str) -> TicketClassification:
    return chain.invoke({
        "ticket_content": content,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })`,
            filename: "agent_triage.py",
          },
        ],
      },
      {
        title: "API de routage",
        content:
          "Exposez l'agent via une API REST simple. Chaque appel reçoit un ticket, interroge la KB pour le contexte, puis retourne la classification.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TicketRequest(BaseModel):
    content: str
    customer_id: str | None = None

@app.post("/api/triage")
async def triage(req: TicketRequest):
    # Recherche de contexte dans la KB
    docs = vectorstore.similarity_search(req.content, k=3)
    context = "\\n".join([d.page_content for d in docs])

    # Classification
    result = classify_ticket(req.content, context)
    return result.model_dump()`,
            filename: "api.py",
          },
        ],
      },
      {
        title: "Tests et déploiement",
        content:
          "Testez avec des tickets réels anonymisés. Mesurez le taux de classification correcte avant mise en production. Déployez sur Railway ou Vercel pour le MVP.",
        codeSnippets: [
          {
            language: "python",
            code: `import pytest
from agent_triage import classify_ticket

def test_technical_ticket():
    result = classify_ticket(
        "Mon application plante quand je clique sur le bouton connexion",
        "Guide de dépannage: vérifier les logs serveur..."
    )
    assert result.category == "technique"
    assert result.confidence > 0.7

def test_billing_ticket():
    result = classify_ticket(
        "Je n'ai toujours pas reçu ma facture du mois dernier",
        "Facturation: les factures sont envoyées le 5 de chaque mois..."
    )
    assert result.category == "facturation"`,
            filename: "test_triage.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Anonymiser les données client (nom, email, téléphone) avant envoi au LLM. Utiliser un proxy de masquage PII comme Presidio ou regex custom.",
      auditLog: "Logger chaque classification avec timestamp, ticket ID, catégorie assignée, score de confiance, et agent humain notifié.",
      humanInTheLoop: "Tickets classifiés avec un score de confiance < 0.7 sont routés vers un agent humain pour validation manuelle.",
      monitoring: "Dashboard Grafana : volume de tickets/heure, taux de classification correcte, temps moyen de triage, alertes si le taux d'erreur dépasse 5%.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook reçoit le ticket → Node HTTP Request vers l'API LLM → Node Switch pour router selon la catégorie → Node HTTP Request vers le système de tickets (Zendesk/Freshdesk).",
      nodes: ["Webhook Trigger", "HTTP Request (LLM API)", "Switch (catégorie)", "HTTP Request (Zendesk)", "Slack Notification"],
      triggerType: "Webhook (nouveau ticket)",
    },
    estimatedTime: "2-4h",
    difficulty: "Facile",
    sectors: ["Services", "E-commerce", "Telecom"],
    metiers: ["Support Client"],
    functions: ["Support"],
    metaTitle: "Agent IA de Triage Support Client — Guide complet",
    metaDescription:
      "Implémentez un agent IA de triage automatique pour votre support client. Stack, tutoriel pas-à-pas et ROI détaillé.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-qualification-leads",
    title: "Agent de Qualification de Leads",
    subtitle: "Scorez et qualifiez automatiquement vos prospects entrants",
    problem:
      "Les commerciaux perdent un temps considérable à qualifier manuellement des leads dont la majorité ne convertira jamais. Les critères de qualification sont appliqués de manière inconsistante.",
    value:
      "L'agent analyse chaque lead entrant (formulaire, email, LinkedIn), le score selon vos critères BANT/MEDDIC, et enrichit la fiche prospect avec des données contextuelles. Seuls les leads qualifiés arrivent aux commerciaux.",
    inputs: [
      "Données du formulaire de contact",
      "Profil LinkedIn / site web du prospect",
      "Historique CRM",
      "Critères de scoring (BANT, MEDDIC)",
      "Données firmographiques",
    ],
    outputs: [
      "Score de qualification (0-100)",
      "Fiche prospect enrichie",
      "Recommandation d'action (qualifier, nurture, disqualifier)",
      "Points de discussion personnalisés",
    ],
    risks: [
      "Faux négatifs : rejet de leads à fort potentiel",
      "Données firmographiques obsolètes",
      "Non-conformité RGPD sur l'enrichissement automatique",
    ],
    roiIndicatif:
      "Augmentation de 35% du taux de conversion MQL→SQL. Réduction de 50% du temps de qualification par lead.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "Ollama + Mixtral", category: "LLM", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Formulaire │────▶│  Agent LLM   │────▶│    CRM      │
│  / Webhook  │     │  (Scoring)   │     │  (enrichi)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │ Enrichment   │
                    │ (LinkedIn/DB)│
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis",
        content:
          "Configurez l'accès à l'API Anthropic et préparez votre grille de scoring. Définissez vos critères BANT (Budget, Authority, Need, Timeline) avec des pondérations.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain psycopg2-binary",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modèle de scoring",
        content:
          "Définissez un modèle de données pour le scoring et les critères de qualification. Le modèle doit être configurable pour s'adapter à différents ICP (Ideal Customer Profile).",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from enum import Enum

class QualificationAction(str, Enum):
    QUALIFY = "qualifier"
    NURTURE = "nurture"
    DISQUALIFY = "disqualifier"

class LeadScore(BaseModel):
    score: int = Field(ge=0, le=100)
    budget_fit: int = Field(ge=0, le=25)
    authority_fit: int = Field(ge=0, le=25)
    need_fit: int = Field(ge=0, le=25)
    timeline_fit: int = Field(ge=0, le=25)
    action: QualificationAction
    reasoning: str
    talking_points: list[str]`,
            filename: "models.py",
          },
        ],
      },
      {
        title: "Agent de qualification",
        content:
          "Construisez l'agent qui analyse les données du lead et produit un score structuré. L'agent utilise le contexte de votre ICP pour évaluer chaque critère BANT.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic

client = anthropic.Anthropic()

def qualify_lead(lead_data: dict, icp_context: str) -> LeadScore:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Analyse ce lead selon nos critères BANT.
ICP: {icp_context}
Lead: {lead_data}
Retourne un JSON avec score, budget_fit, authority_fit,
need_fit, timeline_fit, action, reasoning, talking_points."""
        }]
    )
    return LeadScore.model_validate_json(message.content[0].text)`,
            filename: "qualify.py",
          },
        ],
      },
      {
        title: "Intégration webhook",
        content:
          "Connectez l'agent à votre CRM via webhook. Chaque nouveau lead déclenche automatiquement la qualification et met à jour la fiche dans le CRM.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, BackgroundTasks

app = FastAPI()

@app.post("/webhook/new-lead")
async def handle_new_lead(lead: dict, bg: BackgroundTasks):
    bg.add_task(process_lead, lead)
    return {"status": "processing"}

async def process_lead(lead: dict):
    score = qualify_lead(lead, ICP_CONTEXT)
    await update_crm(lead["id"], score)
    if score.action == "qualifier":
        await notify_sales_team(lead, score)`,
            filename: "webhook.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données de contact (email, téléphone, entreprise) restent dans le CRM. Seules les données anonymisées sont envoyées au LLM pour scoring.",
      auditLog: "Chaque scoring est enregistré : lead ID, score attribué, critères déclencheurs, action recommandée, horodatage.",
      humanInTheLoop: "Les leads scorés 'Hot' (>80) sont validés par un commercial avant ajout dans la séquence de nurturing automatisée.",
      monitoring: "Métriques : taux de conversion par score, précision du scoring vs résultat réel (deal gagné/perdu), volume traité/jour.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Trigger CRM (nouveau lead) → Enrichissement données (Clearbit/Societeinfo) → Appel LLM scoring → Update CRM avec score → Notification Slack si lead Hot.",
      nodes: ["CRM Trigger (HubSpot/Pipedrive)", "HTTP Request (enrichissement)", "HTTP Request (LLM scoring)", "CRM Update (score)", "IF (score > 80)", "Slack Notification"],
      triggerType: "CRM Trigger (nouveau lead créé)",
    },
    estimatedTime: "6-10h",
    difficulty: "Moyen",
    sectors: ["B2B SaaS", "Services"],
    metiers: ["Commercial"],
    functions: ["Sales"],
    metaTitle: "Agent IA de Qualification de Leads — Guide Sales",
    metaDescription:
      "Automatisez la qualification de vos leads avec un agent IA. Scoring BANT, enrichissement et intégration CRM.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-analyse-cv-preselection",
    title: "Agent d'Analyse de CVs et Pré-sélection",
    subtitle: "Filtrez et classez les candidatures automatiquement selon vos critères",
    problem:
      "Les recruteurs reçoivent des centaines de CVs par poste ouvert. Le tri manuel est chronophage, subjectif et laisse passer des profils pertinents noyés dans le volume.",
    value:
      "L'agent IA parse chaque CV, extrait les compétences clés, les compare au profil recherché et produit un classement objectif. Les recruteurs se concentrent sur les entretiens.",
    inputs: [
      "CV au format PDF/DOCX",
      "Fiche de poste avec critères requis",
      "Historique des recrutements réussis",
      "Grille d'évaluation pondérée",
    ],
    outputs: [
      "Score d'adéquation candidat/poste (0-100)",
      "Extraction structurée des compétences",
      "Points forts et points d'attention",
      "Classement des candidatures",
      "Email de pré-sélection personnalisé",
    ],
    risks: [
      "Biais algorithmiques reproduisant des discriminations historiques",
      "Non-conformité RGPD sur le traitement des données personnelles",
      "Rejet de profils atypiques mais à fort potentiel",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de pré-sélection. Amélioration de 20% de la qualité des shortlists.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LlamaIndex", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Unstructured.io", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "PyPDF2 + docx2txt", category: "Other", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  CV Upload  │────▶│  Parser      │────▶│  Agent LLM  │
│  (PDF/DOCX) │     │  (Extract)   │     │  (Scoring)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Dashboard  │◀────│  Classement  │◀────│  Matching   │
│  recruteur  │     │  candidats   │     │  poste/CV   │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et parsing de CVs",
        content:
          "Configurez le parser de documents pour extraire le texte des CVs. Unstructured.io gère nativement PDF, DOCX et images avec OCR.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai llama-index unstructured python-docx PyPDF2",
            filename: "terminal",
          },
          {
            language: "python",
            code: `from unstructured.partition.auto import partition

def extract_cv_text(file_path: str) -> str:
    elements = partition(filename=file_path)
    return "\\n".join([str(el) for el in elements])`,
            filename: "parser.py",
          },
        ],
      },
      {
        title: "Agent d'extraction de compétences",
        content:
          "L'agent extrait les compétences, l'expérience et la formation de chaque CV dans un format structuré pour faciliter le matching.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from pydantic import BaseModel

class CVProfile(BaseModel):
    name: str
    skills: list[str]
    years_experience: int
    education: list[str]
    languages: list[str]
    summary: str

client = OpenAI()

def extract_profile(cv_text: str) -> CVProfile:
    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "Extrais le profil structuré de ce CV."},
            {"role": "user", "content": cv_text}
        ],
        response_format=CVProfile,
    )
    return response.choices[0].message.parsed`,
            filename: "extract.py",
          },
        ],
      },
      {
        title: "Scoring et classement",
        content:
          "Comparez chaque profil extrait aux critères du poste. Le score pondéré permet un classement objectif des candidatures.",
        codeSnippets: [
          {
            language: "python",
            code: `def score_candidate(profile: CVProfile, job_requirements: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"""Évalue ce candidat par rapport au poste.
Candidat: {profile.model_dump_json()}
Poste: {job_requirements}
Score chaque critère sur 25 et donne un score total sur 100.
Liste les points forts et points d'attention."""
        }]
    )
    return response.choices[0].message.content`,
            filename: "scoring.py",
          },
        ],
      },
      {
        title: "Déploiement et RGPD",
        content:
          "Déployez l'agent avec un consentement explicite des candidats. Implémentez le droit à l'effacement et la transparence sur l'utilisation de l'IA dans le processus.",
        codeSnippets: [
          {
            language: "python",
            code: `# Middleware RGPD
def ensure_consent(candidate_id: str) -> bool:
    consent = db.get_consent(candidate_id)
    if not consent or not consent.ai_processing:
        raise PermissionError(
            "Consentement IA non obtenu pour ce candidat"
        )
    return True

def delete_candidate_data(candidate_id: str):
    """Droit à l'effacement RGPD"""
    db.delete_cv(candidate_id)
    db.delete_profile(candidate_id)
    db.delete_scores(candidate_id)
    vectorstore.delete(filter={"candidate_id": candidate_id})`,
            filename: "rgpd.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "RGPD strict : les CVs sont traités en mémoire, jamais stockés sur les serveurs LLM. Consentement explicite du candidat requis. Données supprimées après 30 jours.",
      auditLog: "Chaque analyse est tracée : CV hash, score attribué, critères de matching, décision (présélectionné/rejeté), recruteur validateur.",
      humanInTheLoop: "Tous les CVs rejetés par l'IA sont revus par un recruteur humain dans un délai de 48h. Aucune décision finale automatique.",
      monitoring: "Suivi du taux de faux positifs/négatifs, diversité des profils sélectionnés (biais), temps gagné par recrutement.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Email Trigger (CV reçu) → Extract Text (PDF parser) → HTTP Request LLM (analyse + scoring) → Google Sheets (résultats) → Email notification au recruteur.",
      nodes: ["Email Trigger (IMAP)", "Extract Binary Data", "HTTP Request (PDF to Text)", "HTTP Request (LLM analyse)", "Google Sheets (résultats)", "Send Email (notification)"],
      triggerType: "Email Trigger (nouveau CV reçu)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Tous secteurs"],
    metiers: ["Ressources Humaines"],
    functions: ["RH"],
    metaTitle: "Agent IA d'Analyse de CVs — Recrutement automatisé",
    metaDescription:
      "Automatisez le tri des CVs avec un agent IA. Extraction de compétences, scoring objectif et conformité RGPD.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-veille-concurrentielle",
    title: "Agent de Veille Concurrentielle",
    subtitle: "Surveillez vos concurrents et détectez les signaux faibles automatiquement",
    problem:
      "La veille concurrentielle manuelle est incomplète et toujours en retard. Les analystes passent des heures à scraper des sites, lire des articles et synthétiser des informations déjà obsolètes.",
    value:
      "L'agent monitore en continu les sources pertinentes (sites concurrents, presse, réseaux sociaux, brevets), détecte les changements significatifs et produit des synthèses actionnables.",
    inputs: [
      "Liste des concurrents à surveiller",
      "Sources à monitorer (URLs, RSS, réseaux sociaux)",
      "Critères d'alerte (lancement produit, levée de fonds, recrutement)",
      "Historique de veille précédent",
    ],
    outputs: [
      "Rapport de veille hebdomadaire structuré",
      "Alertes en temps réel sur signaux forts",
      "Analyse comparative (pricing, features, positionnement)",
      "Recommandations stratégiques",
    ],
    risks: [
      "Scraping non autorisé de sites concurrents",
      "Hallucinations dans l'analyse",
      "Surcharge d'alertes non pertinentes",
    ],
    roiIndicatif:
      "Gain de 15h/semaine d'analyse manuelle. Détection de signaux concurrentiels 3x plus rapide.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Firecrawl", category: "Other" },
      { name: "Supabase", category: "Database" },
      { name: "Resend", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "BeautifulSoup + Requests", category: "Other", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Scheduler  │────▶│  Scraper     │────▶│  Agent LLM  │
│  (CRON)     │     │  (Firecrawl) │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  Rapport +  │
                                         │  Alertes    │
                                         └─────────────┘`,
    tutorial: [
      {
        title: "Configuration des sources",
        content:
          "Définissez les sources à surveiller et configurez le scraper. Firecrawl simplifie l'extraction de contenu structuré depuis n'importe quel site web.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic firecrawl-py schedule resend",
            filename: "terminal",
          },
          {
            language: "python",
            code: `COMPETITORS = [
    {"name": "Concurrent A", "url": "https://concurrent-a.com", "rss": None},
    {"name": "Concurrent B", "url": "https://concurrent-b.com", "rss": "https://concurrent-b.com/feed"},
]

ALERT_KEYWORDS = ["levée de fonds", "nouveau produit", "partenariat", "recrutement massif"]`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Scraping et extraction",
        content:
          "Utilisez Firecrawl pour extraire le contenu des pages concurrentes. L'extraction en markdown facilite l'analyse par le LLM.",
        codeSnippets: [
          {
            language: "python",
            code: `from firecrawl import FirecrawlApp

app = FirecrawlApp(api_key="fc-...")

def scrape_competitor(url: str) -> str:
    result = app.scrape_url(url, params={"formats": ["markdown"]})
    return result.get("markdown", "")

def detect_changes(current: str, previous: str) -> bool:
    # Comparaison simple par hash ou diff
    return hash(current) != hash(previous)`,
            filename: "scraper.py",
          },
        ],
      },
      {
        title: "Analyse et synthèse",
        content:
          "L'agent analyse le contenu scrapé, détecte les signaux pertinents et génère une synthèse structurée.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic

client = anthropic.Anthropic()

def analyze_competitor(name: str, content: str, history: str) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Analyse la veille concurrentielle pour {name}.
Contenu actuel: {content[:3000]}
Historique: {history[:1000]}
Identifie: nouveautés produit, changements de pricing,
recrutements, partenariats, signaux faibles.
Format: synthèse structurée avec niveau d'importance."""
        }]
    )
    return message.content[0].text`,
            filename: "analyze.py",
          },
        ],
      },
      {
        title: "Automatisation et alertes",
        content:
          "Planifiez l'exécution quotidienne et configurez les alertes email pour les signaux importants.",
        codeSnippets: [
          {
            language: "python",
            code: `import schedule
import resend

resend.api_key = "re_..."

def daily_scan():
    for competitor in COMPETITORS:
        content = scrape_competitor(competitor["url"])
        if detect_changes(content, get_previous(competitor["name"])):
            analysis = analyze_competitor(
                competitor["name"], content, get_history(competitor["name"])
            )
            save_report(competitor["name"], analysis)
            if contains_alert_keywords(analysis):
                resend.Emails.send({
                    "from": "veille@monentreprise.com",
                    "to": ["strategie@monentreprise.com"],
                    "subject": f"Alerte veille: {competitor['name']}",
                    "html": format_alert_email(analysis)
                })

schedule.every().day.at("07:00").do(daily_scan)`,
            filename: "scheduler.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée. Les sources sont publiques (sites web, press releases, réseaux sociaux corporate).",
      auditLog: "Chaque rapport de veille est versionné : sources consultées, données extraites, synthèse générée, date, analyste destinataire.",
      humanInTheLoop: "Le rapport hebdomadaire est validé par un analyste avant diffusion aux stakeholders. L'IA propose, l'humain valide.",
      monitoring: "Alertes si une source devient inaccessible, tracking du nombre de signaux détectés/semaine, feedback des analystes sur la pertinence.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien) → HTTP Request (scraping URLs concurrents) → HTTP Request LLM (synthèse) → Notion/Google Docs (rapport) → Slack notification.",
      nodes: ["Cron Trigger (daily 8h)", "HTTP Request (scrape URL 1)", "HTTP Request (scrape URL 2)", "Merge", "HTTP Request (LLM synthèse)", "Notion Create Page", "Slack Notification"],
      triggerType: "Cron (quotidien à 8h)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Tous secteurs"],
    metiers: ["Marketing Stratégique"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Veille Concurrentielle — Guide Marketing",
    metaDescription:
      "Automatisez votre veille concurrentielle avec un agent IA. Monitoring continu, alertes et synthèses automatiques.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-rapports-financiers",
    title: "Agent de Génération de Rapports Financiers",
    subtitle: "Automatisez la production de rapports financiers structurés et commentés",
    problem:
      "La production de rapports financiers mensuels mobilise des équipes entières pendant plusieurs jours. Les erreurs de copier-coller et les incohérences entre sources de données sont fréquentes.",
    value:
      "L'agent collecte les données financières depuis vos systèmes, génère des rapports structurés avec commentaires analytiques, et détecte automatiquement les anomalies et écarts significatifs.",
    inputs: [
      "Données comptables (ERP, exports CSV)",
      "Budget prévisionnel",
      "Rapports N-1 pour comparaison",
      "Règles métier et seuils d'alerte",
      "Template de rapport",
    ],
    outputs: [
      "Rapport financier complet (PDF/Excel)",
      "Commentaires analytiques automatiques",
      "Détection d'anomalies et écarts",
      "Graphiques et tableaux de bord",
      "Recommandations d'actions correctives",
    ],
    risks: [
      "Erreurs de calcul si les données sources sont incorrectes",
      "Hallucinations dans les commentaires analytiques",
      "Confidentialité des données financières sensibles",
      "Non-conformité réglementaire des rapports générés",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de production des rapports. Détection de 95% des anomalies comptables.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Pandas + Plotly", category: "Other" },
      { name: "WeasyPrint", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + CodeLlama", category: "LLM", isFree: true },
      { name: "DuckDB", category: "Database", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
      { name: "Matplotlib", category: "Other", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ERP/CSV    │────▶│  ETL +       │────▶│  Agent LLM  │
│  (données)  │     │  Validation  │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  PDF/Excel  │◀────│  Générateur  │◀────│  Graphiques │
│  (rapport)  │     │  de rapport  │     │  + Tableaux │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et connexion aux données",
        content:
          "Configurez la connexion à vos sources de données financières. Pandas gère nativement les imports CSV, Excel et les connexions SQL.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai langchain pandas plotly weasyprint psycopg2-binary openpyxl",
            filename: "terminal",
          },
          {
            language: "python",
            code: `import pandas as pd

def load_financial_data(period: str) -> dict:
    revenue = pd.read_sql(
        f"SELECT * FROM revenue WHERE period = '{period}'", engine
    )
    expenses = pd.read_sql(
        f"SELECT * FROM expenses WHERE period = '{period}'", engine
    )
    budget = pd.read_sql(
        f"SELECT * FROM budget WHERE period = '{period}'", engine
    )
    return {"revenue": revenue, "expenses": expenses, "budget": budget}`,
            filename: "data_loader.py",
          },
        ],
      },
      {
        title: "Détection d'anomalies",
        content:
          "Implémentez la détection automatique d'écarts significatifs entre le réalisé et le budget, ainsi que les variations inhabituelles mois-sur-mois.",
        codeSnippets: [
          {
            language: "python",
            code: `def detect_anomalies(data: dict, threshold: float = 0.1) -> list:
    anomalies = []
    revenue = data["revenue"]
    budget = data["budget"]

    merged = revenue.merge(budget, on="category", suffixes=("_real", "_budget"))
    merged["ecart_pct"] = (
        (merged["amount_real"] - merged["amount_budget"]) / merged["amount_budget"]
    )

    for _, row in merged.iterrows():
        if abs(row["ecart_pct"]) > threshold:
            anomalies.append({
                "category": row["category"],
                "ecart": f"{row['ecart_pct']:.1%}",
                "reel": row["amount_real"],
                "budget": row["amount_budget"],
                "type": "dépassement" if row["ecart_pct"] > 0 else "sous-réalisation"
            })
    return anomalies`,
            filename: "anomalies.py",
          },
        ],
      },
      {
        title: "Génération des commentaires",
        content:
          "L'agent LLM analyse les chiffres et les anomalies pour produire des commentaires analytiques pertinents, dans le style des rapports financiers professionnels.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI

client = OpenAI()

def generate_commentary(data_summary: str, anomalies: list) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Tu es un analyste financier senior.
Génère des commentaires professionnels pour un rapport financier mensuel.
Style: factuel, concis, orienté décision. Langue: français."""
        }, {
            "role": "user",
            "content": f"""Données: {data_summary}
Anomalies détectées: {anomalies}
Génère: 1) Synthèse exécutive 2) Analyse par poste
3) Anomalies commentées 4) Recommandations"""
        }],
        temperature=0.3,
    )
    return response.choices[0].message.content`,
            filename: "commentary.py",
          },
        ],
      },
      {
        title: "Génération PDF et déploiement",
        content:
          "Assemblez le rapport final en PDF avec graphiques Plotly et commentaires générés. Automatisez l'envoi mensuel.",
        codeSnippets: [
          {
            language: "python",
            code: `import plotly.graph_objects as go
from weasyprint import HTML

def generate_report_pdf(data: dict, commentary: str, period: str):
    # Graphique revenus vs budget
    fig = go.Figure(data=[
        go.Bar(name="Réalisé", x=data["categories"], y=data["actual"]),
        go.Bar(name="Budget", x=data["categories"], y=data["budget"]),
    ])
    fig.update_layout(barmode="group", title=f"Résultats {period}")
    chart_html = fig.to_html(include_plotlyjs="cdn")

    html_content = f"""<html>
    <h1>Rapport Financier — {period}</h1>
    {chart_html}
    <div>{commentary}</div>
    </html>"""

    HTML(string=html_content).write_pdf(f"rapport_{period}.pdf")`,
            filename: "report_generator.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Données financières sensibles : chiffrement en transit (TLS) et au repos. Accès restreint par rôle (RBAC). Aucune donnée envoyée à des LLM cloud — utiliser un modèle on-premise ou Azure OpenAI avec data residency EU.",
      auditLog: "Chaque rapport généré est tracé : sources de données, requêtes SQL exécutées, calculs effectués, version du template, approbateur.",
      humanInTheLoop: "Tout rapport financier est relu et approuvé par le DAF ou un contrôleur de gestion avant publication. Double validation pour les montants > 100K€.",
      monitoring: "Alertes si écart > 5% avec les données du mois précédent, monitoring des temps de génération, audit trail complet pour les commissaires aux comptes.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (mensuel J+5) → Postgres Query (données financières) → Code Node (calculs KPIs) → HTTP Request LLM (rédaction narrative) → Google Slides (mise en forme) → Email au DAF.",
      nodes: ["Cron Trigger (mensuel)", "Postgres (requêtes financières)", "Code Node (KPIs)", "HTTP Request (LLM rédaction)", "Google Slides (template)", "Send Email (DAF)"],
      triggerType: "Cron (mensuel, J+5 ouvré)",
    },
    estimatedTime: "20-30h",
    difficulty: "Expert",
    sectors: ["Banque", "Assurance", "Audit"],
    metiers: ["Finance", "Comptabilité"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Rapports Financiers — Guide Expert",
    metaDescription:
      "Automatisez vos rapports financiers avec un agent IA. Analyse, détection d'anomalies et commentaires automatiques.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-gestion-incidents-it",
    title: "Agent de Gestion des Incidents IT",
    subtitle: "Diagnostiquez et résolvez les incidents IT automatiquement",
    problem:
      "Les équipes IT sont submergées par les tickets d'incidents répétitifs. Le diagnostic manuel est lent et dépend de l'expertise individuelle, créant des goulots d'étranglement.",
    value:
      "L'agent analyse les logs, corrèle les événements, propose un diagnostic et exécute les runbooks de résolution automatiquement. Les incidents L1/L2 sont résolus sans intervention humaine.",
    inputs: [
      "Logs applicatifs et système",
      "Alertes monitoring (Datadog, Grafana)",
      "Base de connaissances incidents (runbooks)",
      "Topologie infrastructure",
    ],
    outputs: [
      "Diagnostic de la cause racine",
      "Actions de remédiation suggérées/exécutées",
      "Post-mortem automatique",
      "Mise à jour du ticket ITSM",
      "Notification aux parties prenantes",
    ],
    risks: [
      "Exécution de remédiation incorrecte aggravant l'incident",
      "Faux positifs dans la corrélation d'événements",
      "Latence dans le diagnostic retardant la résolution",
    ],
    roiIndicatif:
      "Résolution automatique de 45% des incidents L1. Réduction du MTTR de 60%.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Elasticsearch", category: "Database" },
      { name: "PagerDuty", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "Loki + Grafana", category: "Monitoring", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "SQLite FTS", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Monitoring │────▶│  Corrélation │────▶│  Agent LLM  │
│  (Alertes)  │     │  événements  │     │  (Diagnostic)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  ITSM       │◀────│  Exécution   │◀────│  Runbook    │
│  (Ticket)   │     │  remédiation │     │  sélection  │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Configuration du pipeline de logs",
        content:
          "Connectez vos sources de logs à l'agent. Elasticsearch stocke et indexe les logs pour permettre la recherche et la corrélation rapides.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai langchain elasticsearch pagerduty-api",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Corrélation d'événements",
        content:
          "L'agent corrèle les alertes et les logs dans une fenêtre temporelle pour identifier les patterns d'incidents connus.",
        codeSnippets: [
          {
            language: "python",
            code: `from elasticsearch import Elasticsearch
from datetime import datetime, timedelta

es = Elasticsearch(["http://localhost:9200"])

def correlate_events(alert: dict, window_minutes: int = 15) -> list:
    end_time = datetime.fromisoformat(alert["timestamp"])
    start_time = end_time - timedelta(minutes=window_minutes)

    result = es.search(index="logs-*", body={
        "query": {
            "bool": {
                "must": [
                    {"range": {"@timestamp": {"gte": start_time, "lte": end_time}}},
                    {"terms": {"level": ["ERROR", "CRITICAL", "FATAL"]}}
                ]
            }
        },
        "sort": [{"@timestamp": "desc"}],
        "size": 50
    })
    return [hit["_source"] for hit in result["hits"]["hits"]]`,
            filename: "correlate.py",
          },
        ],
      },
      {
        title: "Diagnostic automatique",
        content:
          "L'agent LLM analyse les événements corrélés et les compare aux runbooks existants pour produire un diagnostic structuré.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI

client = OpenAI()

def diagnose_incident(events: list, runbooks: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": f"""Tu es un ingénieur SRE senior.
Runbooks disponibles: {runbooks}"""
        }, {
            "role": "user",
            "content": f"""Événements corrélés: {events}
Diagnostic: cause racine probable, runbook applicable,
actions de remédiation recommandées, niveau de sévérité."""
        }],
        temperature=0,
    )
    return {"diagnosis": response.choices[0].message.content}`,
            filename: "diagnose.py",
          },
        ],
      },
      {
        title: "Remédiation et notification",
        content:
          "Exécutez automatiquement les runbooks approuvés et notifiez les équipes via PagerDuty ou Slack.",
        codeSnippets: [
          {
            language: "python",
            code: `import subprocess

APPROVED_RUNBOOKS = {
    "restart_service": "systemctl restart {service}",
    "clear_cache": "redis-cli FLUSHDB",
    "scale_up": "kubectl scale deployment {deployment} --replicas={count}",
}

def execute_runbook(runbook_id: str, params: dict) -> dict:
    if runbook_id not in APPROVED_RUNBOOKS:
        return {"status": "blocked", "reason": "Runbook non approuvé"}

    cmd = APPROVED_RUNBOOKS[runbook_id].format(**params)
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "output": result.stdout.decode()
    }`,
            filename: "remediate.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les logs peuvent contenir des IPs et identifiants utilisateur — anonymiser avant analyse LLM. Pas de credentials dans les logs envoyés.",
      auditLog: "Chaque incident traité : ID incident, source d'alerte, diagnostic IA, actions prises, temps de résolution, escalade éventuelle.",
      humanInTheLoop: "Les incidents critiques (P1/P2) déclenchent une escalade automatique vers l'ingénieur d'astreinte. L'IA propose un diagnostic, l'humain exécute.",
      monitoring: "MTTR (Mean Time To Resolve), taux de résolution automatique, faux positifs d'alertes, SLA compliance par criticité.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (alerte PagerDuty/Datadog) → HTTP Request LLM (diagnostic) → Switch (criticité) → Jira Create Issue → Slack Alert → IF P1: PagerDuty escalade.",
      nodes: ["Webhook Trigger (PagerDuty)", "HTTP Request (LLM diagnostic)", "Switch (criticité P1-P4)", "Jira Create Issue", "Slack Notification", "IF P1: PagerDuty Escalade"],
      triggerType: "Webhook (alerte monitoring)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Tous secteurs"],
    metiers: ["IT", "DevOps"],
    functions: ["IT"],
    metaTitle: "Agent IA de Gestion des Incidents IT — Guide DevOps",
    metaDescription:
      "Automatisez le diagnostic et la résolution des incidents IT avec un agent IA. Corrélation de logs et runbooks automatiques.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-redaction-contenu-marketing",
    title: "Agent de Rédaction de Contenu Marketing",
    subtitle: "Générez du contenu marketing de qualité à grande échelle",
    problem:
      "Les équipes marketing doivent produire un volume croissant de contenu (articles, posts sociaux, newsletters) tout en maintenant la cohérence de la marque et la qualité éditoriale.",
    value:
      "L'agent génère des drafts de contenu alignés avec votre charte éditoriale, vos personas et votre calendrier marketing. Les rédacteurs se concentrent sur la validation et l'affinement.",
    inputs: [
      "Brief éditorial (sujet, angle, persona cible)",
      "Charte éditoriale et tone of voice",
      "Mots-clés SEO cibles",
      "Contenus existants (pour éviter les répétitions)",
    ],
    outputs: [
      "Draft d'article de blog (1000-2000 mots)",
      "Variantes de posts réseaux sociaux",
      "Objet et corps d'email / newsletter",
      "Méta-descriptions SEO",
    ],
    risks: [
      "Contenu trop générique ou non-différenciant",
      "Plagiat involontaire de sources externes",
      "Ton incohérent avec la marque",
    ],
    roiIndicatif:
      "Production de contenu 4x plus rapide. Réduction de 60% du coût par article.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Mistral API (gratuit 1M tokens)", category: "LLM", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Brief     │────▶│  Agent LLM   │────▶│  Draft      │
│  éditorial  │     │  (Rédaction) │     │  contenu    │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Charte +    │
                    │  SEO + KB    │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Configuration de la charte éditoriale",
        content:
          "Encodez votre charte éditoriale dans un prompt système réutilisable. C'est la clé pour des contenus cohérents avec votre marque.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain",
            filename: "terminal",
          },
          {
            language: "python",
            code: `EDITORIAL_CHARTER = """
# Charte éditoriale — Mon Entreprise
## Tone of voice: professionnel mais accessible
## Tutoiement/Vouvoiement: vouvoiement
## Style: phrases courtes, verbes d'action, exemples concrets
## À éviter: jargon technique non expliqué, superlatifs
## Persona principal: Directeur Marketing, 35-50 ans, PME
"""`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Agent de rédaction",
        content:
          "L'agent prend un brief et produit un draft structuré respectant la charte, les mots-clés SEO et le persona cible.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic

client = anthropic.Anthropic()

def generate_article(brief: dict) -> str:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Rédige un article de blog selon ce brief.
{EDITORIAL_CHARTER}
Sujet: {brief['topic']}
Angle: {brief['angle']}
Persona: {brief['persona']}
Mots-clés SEO: {', '.join(brief['keywords'])}
Longueur: {brief.get('word_count', 1500)} mots
Structure: titre H1, intro, 3-5 sections H2, conclusion, CTA."""
        }]
    )
    return message.content[0].text`,
            filename: "writer.py",
          },
        ],
      },
      {
        title: "Déclinaison multi-canal",
        content:
          "À partir de l'article, générez automatiquement les déclinaisons pour les réseaux sociaux et la newsletter.",
        codeSnippets: [
          {
            language: "python",
            code: `def generate_social_posts(article: str, platforms: list) -> dict:
    posts = {}
    for platform in platforms:
        limits = {"linkedin": 3000, "twitter": 280, "instagram": 2200}
        message = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""Décline cet article en post {platform}.
Max {limits[platform]} caractères. Ton engageant, avec CTA.
Article: {article[:2000]}"""
            }]
        )
        posts[platform] = message.content[0].text
    return posts`,
            filename: "social.py",
          },
        ],
      },
      {
        title: "Pipeline de publication",
        content:
          "Automatisez le workflow : brief → draft → review → publication. Le rédacteur valide et ajuste avant publication.",
        codeSnippets: [
          {
            language: "python",
            code: `async def content_pipeline(brief: dict):
    # 1. Génération du draft
    article = generate_article(brief)

    # 2. Déclinaisons sociales
    posts = generate_social_posts(article, ["linkedin", "twitter"])

    # 3. Méta-description SEO
    meta = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": f"Méta-description SEO (155 chars max) pour: {article[:500]}"}]
    )

    return {
        "article": article,
        "social_posts": posts,
        "meta_description": meta.content[0].text,
        "status": "draft_ready"
    }`,
            filename: "pipeline.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle. Le brief marketing et le contenu généré sont des données corporate non sensibles.",
      auditLog: "Chaque contenu généré : brief initial, prompt utilisé, version générée, modifications humaines, publication finale, performance (vues/clics).",
      humanInTheLoop: "Tout contenu est relu par un rédacteur senior ou le brand manager avant publication. L'IA rédige un premier jet, l'humain affine et valide.",
      monitoring: "Volume de contenu produit/semaine, taux d'acceptation sans modification, performance SEO des contenus générés vs rédigés manuellement.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Google Sheets Trigger (nouveau brief) → HTTP Request LLM (rédaction) → Google Docs (brouillon) → Slack notification au rédacteur → Attente validation → WordPress Publish.",
      nodes: ["Google Sheets Trigger", "HTTP Request (LLM rédaction)", "Google Docs (créer brouillon)", "Slack Notification", "Wait (validation)", "WordPress Create Post"],
      triggerType: "Google Sheets (nouveau brief ajouté)",
    },
    estimatedTime: "2-4h",
    difficulty: "Facile",
    sectors: ["E-commerce", "B2B SaaS", "Média"],
    metiers: ["Marketing Digital"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Rédaction Marketing — Guide Content",
    metaDescription:
      "Générez du contenu marketing de qualité avec un agent IA. Articles, posts sociaux et newsletters automatisés.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-onboarding-collaborateurs",
    title: "Agent d'Onboarding Collaborateurs",
    subtitle: "Accompagnez les nouveaux collaborateurs avec un assistant IA personnalisé",
    problem:
      "L'onboarding des nouveaux collaborateurs est souvent désorganisé : documents éparpillés, interlocuteurs multiples, questions répétitives. Le résultat est une montée en compétence lente et une mauvaise expérience employé.",
    value:
      "L'agent IA guide chaque nouveau collaborateur à travers un parcours d'onboarding personnalisé, répond à leurs questions 24/7 et s'assure que toutes les étapes administratives et formation sont complétées.",
    inputs: [
      "Base de connaissances RH (politiques, procédures)",
      "Parcours d'onboarding par poste/département",
      "Informations du nouveau collaborateur",
      "FAQ existante",
    ],
    outputs: [
      "Parcours d'onboarding personnalisé (checklist)",
      "Réponses aux questions fréquentes",
      "Rappels automatiques pour les étapes à compléter",
      "Rapport de progression pour les RH",
    ],
    risks: [
      "Informations RH obsolètes dans la base de connaissances",
      "Réponses incorrectes sur des sujets légaux/contractuels",
      "Manque de contact humain dans l'accueil",
    ],
    roiIndicatif:
      "Réduction de 50% des sollicitations RH répétitives. Time-to-productivity amélioré de 30%.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1-mini", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Pinecone", category: "Database" },
      { name: "Slack API", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Phi-3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Discord Bot", category: "Other", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Nouveau    │────▶│  Chatbot     │────▶│  Agent LLM  │
│  collabor.  │     │  (Slack/Web) │     │  (RAG)      │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                         ┌──────▼──────┐
                                         │  KB RH      │
                                         │  (Vector DB)│
                                         └─────────────┘`,
    tutorial: [
      {
        title: "Indexation de la base RH",
        content:
          "Indexez tous les documents RH (politiques, procédures, FAQ) dans un vector store pour permettre le RAG (Retrieval Augmented Generation).",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai langchain pinecone-client slack-sdk",
            filename: "terminal",
          },
          {
            language: "python",
            code: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader

loader = DirectoryLoader("./rh_docs", glob="**/*.{md,pdf,docx}")
docs = loader.load()

vectorstore = Pinecone.from_documents(
    docs, OpenAIEmbeddings(), index_name="onboarding-kb"
)`,
            filename: "index_rh.py",
          },
        ],
      },
      {
        title: "Agent conversationnel RAG",
        content:
          "Créez un agent qui répond aux questions en s'appuyant sur la base de connaissances RH. Le RAG garantit des réponses factuelles basées sur vos documents internes.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4.1-mini", temperature=0.3),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    verbose=True,
)

def ask_onboarding_bot(question: str) -> str:
    result = qa_chain.invoke({"question": question})
    return result["answer"]`,
            filename: "bot.py",
          },
        ],
      },
      {
        title: "Intégration Slack",
        content:
          "Connectez l'agent à Slack pour que les nouveaux collaborateurs puissent poser leurs questions directement dans un canal dédié.",
        codeSnippets: [
          {
            language: "python",
            code: `from slack_bolt import App

app = App(token="xoxb-...", signing_secret="...")

@app.message("")
def handle_message(message, say):
    user_question = message["text"]
    response = ask_onboarding_bot(user_question)
    say(response)

if __name__ == "__main__":
    app.start(port=3000)`,
            filename: "slack_bot.py",
          },
        ],
      },
      {
        title: "Checklist et suivi",
        content:
          "Générez automatiquement une checklist d'onboarding personnalisée et suivez la progression de chaque nouveau collaborateur.",
        codeSnippets: [
          {
            language: "python",
            code: `ONBOARDING_STEPS = {
    "engineering": [
        "Configurer l'environnement de dev",
        "Accéder aux repos GitHub",
        "Lire la documentation d'architecture",
        "Faire un premier commit",
        "Participer au standup",
    ],
    "marketing": [
        "Accéder aux outils (HubSpot, Figma, GA4)",
        "Lire la charte éditoriale",
        "Rencontrer l'équipe produit",
        "Publier un premier contenu",
    ],
}

def create_onboarding_checklist(department: str, name: str) -> dict:
    steps = ONBOARDING_STEPS.get(department, ONBOARDING_STEPS["engineering"])
    return {
        "employee": name,
        "department": department,
        "steps": [{"task": s, "completed": False} for s in steps],
        "created_at": datetime.now().isoformat()
    }`,
            filename: "checklist.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Données RH sensibles (contrat, salaire, identité) : accès restreint, chiffrement, conformité RGPD. Le chatbot ne stocke pas les conversations sensibles.",
      auditLog: "Chaque interaction chatbot loggée : collaborateur ID, question posée, réponse donnée, source documentaire, satisfaction (pouce haut/bas).",
      humanInTheLoop: "Questions non résolues par le chatbot sont escaladées au RH référent. Le chatbot ne prend aucune décision contractuelle.",
      monitoring: "Taux de résolution chatbot, questions les plus fréquentes, NPS collaborateurs onboardés, temps d'onboarding moyen (avant/après IA).",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Slack Trigger (message nouveau collaborateur) → HTTP Request LLM (RAG sur docs RH) → Slack Reply → IF non résolu: Créer ticket RH dans Notion.",
      nodes: ["Slack Trigger (channel onboarding)", "HTTP Request (LLM + RAG)", "Slack Reply", "IF (confiance < 0.7)", "Notion Create Page (ticket RH)"],
      triggerType: "Slack message (channel #onboarding)",
    },
    estimatedTime: "2-4h",
    difficulty: "Facile",
    sectors: ["Tous secteurs"],
    metiers: ["Ressources Humaines"],
    functions: ["RH"],
    metaTitle: "Agent IA d'Onboarding — Guide RH",
    metaDescription:
      "Créez un assistant IA d'onboarding pour vos nouveaux collaborateurs. Chatbot RAG, Slack et checklist automatique.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-detection-fraude",
    title: "Agent de Détection de Fraude",
    subtitle: "Détectez les transactions frauduleuses en temps réel grâce à l'IA",
    problem:
      "Les systèmes de détection de fraude basés sur des règles statiques laissent passer des fraudes sophistiquées et génèrent trop de faux positifs, mobilisant les analystes sur des alertes non pertinentes.",
    value:
      "L'agent combine des modèles ML de scoring avec un LLM pour analyser le contexte de chaque transaction suspecte, réduisant les faux positifs de 70% et détectant des patterns de fraude inconnus.",
    inputs: [
      "Flux de transactions en temps réel",
      "Profil comportemental du client",
      "Historique des fraudes connues",
      "Données contextuelles (géolocalisation, device)",
      "Règles réglementaires (LCB-FT)",
    ],
    outputs: [
      "Score de risque par transaction (0-100)",
      "Classification (légitime, suspecte, frauduleuse)",
      "Explication détaillée du diagnostic",
      "Décision automatique (approuver, bloquer, escalader)",
      "Rapport de conformité SAR",
    ],
    risks: [
      "Faux positifs bloquant des transactions légitimes",
      "Latence inacceptable sur les transactions temps réel",
      "Biais dans le scoring pénalisant certains profils",
      "Non-conformité réglementaire des décisions automatisées",
    ],
    roiIndicatif:
      "Réduction de 70% des faux positifs. Détection de 30% de fraudes supplémentaires non capturées par les règles.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "Apache Kafka", category: "Other" },
      { name: "PostgreSQL + TimescaleDB", category: "Database" },
      { name: "scikit-learn / XGBoost", category: "Other" },
      { name: "Grafana", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Redis Streams", category: "Other", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "DuckDB", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Transaction │────▶│  ML Scoring  │────▶│  Agent LLM  │
│  (stream)   │     │  (XGBoost)   │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Dashboard  │◀────│  Décision    │◀────│  Règles +   │
│  analyste   │     │  (auto/esc.) │     │  Contexte   │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Modèle ML de scoring",
        content:
          "Entraînez un modèle XGBoost sur votre historique de transactions pour le scoring initial. Le ML gère le volume en temps réel, le LLM intervient pour l'analyse contextuelle des cas suspects.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai xgboost scikit-learn pandas kafka-python",
            filename: "terminal",
          },
          {
            language: "python",
            code: `import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_fraud_model(data_path: str):
    df = pd.read_csv(data_path)
    features = ["amount", "hour", "merchant_category",
                "distance_from_home", "transaction_frequency"]
    X = df[features]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier(scale_pos_weight=50, max_depth=6)
    model.fit(X_train, y_train)
    return model`,
            filename: "train_model.py",
          },
        ],
      },
      {
        title: "Pipeline temps réel",
        content:
          "Consommez le flux de transactions via Kafka. Chaque transaction est d'abord scorée par le modèle ML, puis les cas suspects sont analysés par le LLM.",
        codeSnippets: [
          {
            language: "python",
            code: `from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    "transactions",
    bootstrap_servers=["localhost:9092"],
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

SUSPECT_THRESHOLD = 0.6

for message in consumer:
    transaction = message.value
    ml_score = model.predict_proba([extract_features(transaction)])[0][1]

    if ml_score > SUSPECT_THRESHOLD:
        # Analyse approfondie par LLM
        analysis = analyze_with_llm(transaction, ml_score)
        decision = make_decision(analysis)
        execute_decision(transaction, decision)
    else:
        approve_transaction(transaction)`,
            filename: "pipeline.py",
          },
        ],
      },
      {
        title: "Analyse contextuelle LLM",
        content:
          "Le LLM analyse le contexte complet de la transaction suspecte : profil client, historique, géolocalisation, patterns comportementaux.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI

client = OpenAI()

def analyze_with_llm(transaction: dict, ml_score: float) -> dict:
    customer_profile = get_customer_profile(transaction["customer_id"])
    recent_transactions = get_recent_transactions(transaction["customer_id"])

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Analyste fraude expert. Évalue cette transaction.
Critères: montant inhabituel, localisation, fréquence, merchant."""
        }, {
            "role": "user",
            "content": f"""Transaction: {transaction}
Score ML: {ml_score:.2f}
Profil client: {customer_profile}
Transactions récentes: {recent_transactions}
Verdict: légitime/suspecte/frauduleuse + explication."""
        }],
        temperature=0,
    )
    return {"analysis": response.choices[0].message.content}`,
            filename: "llm_analysis.py",
          },
        ],
      },
      {
        title: "Conformité et reporting",
        content:
          "Générez automatiquement les rapports de conformité (SAR) pour les transactions signalées comme frauduleuses, conformément aux obligations LCB-FT.",
        codeSnippets: [
          {
            language: "python",
            code: `def generate_sar_report(transaction: dict, analysis: dict) -> str:
    """Génère un rapport SAR (Suspicious Activity Report)"""
    report = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"""Génère un rapport SAR conforme LCB-FT.
Transaction: {transaction}
Analyse: {analysis}
Format: narratif structuré avec dates, montants,
parties impliquées, indicateurs de suspicion."""
        }],
        temperature=0.1,
    )
    return report.choices[0].message.content

def log_decision(transaction_id: str, decision: str, analysis: str):
    """Traçabilité complète pour audit réglementaire"""
    db.insert("fraud_decisions", {
        "transaction_id": transaction_id,
        "decision": decision,
        "analysis": analysis,
        "timestamp": datetime.now(),
        "model_version": "v1.2",
    })`,
            filename: "compliance.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Données bancaires hautement sensibles : tokenisation des numéros de carte, chiffrement AES-256, conformité PCI-DSS. Aucune donnée en clair dans les logs.",
      auditLog: "Audit trail complet exigé par les régulateurs : chaque transaction analysée, score de risque, décision (approuver/bloquer/escalader), justification IA, timestamp.",
      humanInTheLoop: "Les transactions bloquées par l'IA sont systématiquement revues par un analyste fraude dans les 2h. Droit de recours client garanti.",
      monitoring: "Taux de détection (recall), précision (precision), faux positifs par jour, temps moyen de review humain, conformité réglementaire LCB-FT.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouvelle transaction) → Code Node (feature engineering) → HTTP Request (modèle ML scoring) → HTTP Request LLM (analyse contexte) → Switch (décision) → DB Update + Alerte.",
      nodes: ["Webhook Trigger (transaction)", "Code Node (features)", "HTTP Request (ML scoring)", "HTTP Request (LLM contexte)", "Switch (approuver/bloquer/escalader)", "Postgres Update", "Slack Alert (si fraude)"],
      triggerType: "Webhook (nouvelle transaction temps réel)",
    },
    estimatedTime: "30-50h",
    difficulty: "Expert",
    sectors: ["Banque", "Assurance", "E-commerce"],
    metiers: ["Conformité", "Risk Management"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Détection de Fraude — Guide Expert",
    metaDescription:
      "Implémentez un agent IA de détection de fraude temps réel. ML + LLM, conformité LCB-FT et réduction des faux positifs.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-automatisation-achats",
    title: "Agent d'Automatisation des Achats",
    subtitle: "Optimisez vos processus achats de la demande à la commande",
    problem:
      "Les processus achats sont lents et manuels : comparaison de devis fastidieuse, validation multi-niveaux chronophage, suivi fournisseurs fragmenté. Les acheteurs passent plus de temps sur l'administratif que sur la négociation.",
    value:
      "L'agent automatise la comparaison de devis, la sélection fournisseurs, le workflow de validation et le suivi de commande. Les acheteurs se concentrent sur la négociation stratégique et la relation fournisseur.",
    inputs: [
      "Demandes d'achat internes",
      "Catalogue fournisseurs et historique prix",
      "Devis reçus (PDF, email)",
      "Règles de validation (seuils, hiérarchie)",
      "Contrats cadres existants",
    ],
    outputs: [
      "Comparatif de devis structuré",
      "Recommandation fournisseur argumentée",
      "Bon de commande pré-rempli",
      "Suivi de livraison automatisé",
      "Rapport d'économies réalisées",
    ],
    risks: [
      "Erreurs dans l'extraction de données de devis",
      "Biais vers les fournisseurs historiques",
      "Non-respect des contrats cadres",
    ],
    roiIndicatif:
      "Réduction de 25% des coûts d'achat. Cycle d'achat raccourci de 40%.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Unstructured.io", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "PyPDF2", category: "Other", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Demande    │────▶│  Parser      │────▶│  Agent LLM  │
│  d'achat    │     │  (Devis PDF) │     │  (Comparaif)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Commande   │◀────│  Validation  │◀────│  Recommand. │
│  fourniss.  │     │  workflow    │     │  fournisseur│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Extraction de devis",
        content:
          "Parsez automatiquement les devis PDF reçus des fournisseurs pour extraire les informations clés : lignes de produit, prix unitaires, conditions.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain unstructured psycopg2-binary",
            filename: "terminal",
          },
          {
            language: "python",
            code: `import anthropic
from unstructured.partition.pdf import partition_pdf

client = anthropic.Anthropic()

def extract_quote_data(pdf_path: str) -> dict:
    elements = partition_pdf(pdf_path)
    text = "\\n".join([str(el) for el in elements])

    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Extrais les données de ce devis:
{text}
Retourne un JSON: fournisseur, date, lignes (ref, desc,
qté, prix_unitaire, total), conditions, délai livraison."""
        }]
    )
    return message.content[0].text`,
            filename: "extract_quote.py",
          },
        ],
      },
      {
        title: "Comparaison et recommandation",
        content:
          "Comparez automatiquement les devis extraits selon vos critères pondérés : prix, qualité, délai, fiabilité fournisseur.",
        codeSnippets: [
          {
            language: "python",
            code: `def compare_quotes(quotes: list, criteria_weights: dict) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Compare ces devis fournisseurs:
{quotes}
Critères pondérés: {criteria_weights}
Produis: 1) Tableau comparatif
2) Score par fournisseur
3) Recommandation argumentée
4) Risques identifiés"""
        }]
    )
    return {"comparison": message.content[0].text}`,
            filename: "compare.py",
          },
        ],
      },
      {
        title: "Workflow de validation",
        content:
          "Implémentez un workflow de validation multi-niveaux basé sur les seuils de montant. L'agent route automatiquement les demandes vers les bons validateurs.",
        codeSnippets: [
          {
            language: "python",
            code: `APPROVAL_THRESHOLDS = [
    {"max_amount": 1000, "approvers": ["manager"]},
    {"max_amount": 10000, "approvers": ["manager", "direction_achats"]},
    {"max_amount": float("inf"), "approvers": ["manager", "direction_achats", "dg"]},
]

def get_approval_chain(amount: float) -> list:
    for threshold in APPROVAL_THRESHOLDS:
        if amount <= threshold["max_amount"]:
            return threshold["approvers"]
    return APPROVAL_THRESHOLDS[-1]["approvers"]

async def submit_for_approval(purchase_request: dict):
    amount = purchase_request["total_amount"]
    chain = get_approval_chain(amount)
    for approver_role in chain:
        approver = get_approver(approver_role, purchase_request["department"])
        await send_approval_request(approver, purchase_request)`,
            filename: "workflow.py",
          },
        ],
      },
      {
        title: "Suivi et reporting",
        content:
          "Suivez automatiquement l'exécution des commandes et générez des rapports d'économies pour mesurer l'impact de l'automatisation.",
        codeSnippets: [
          {
            language: "python",
            code: `def generate_savings_report(period: str) -> dict:
    purchases = db.get_purchases(period=period)
    total_spent = sum(p["amount"] for p in purchases)
    total_savings = sum(p.get("savings", 0) for p in purchases)

    report = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Rapport achats {period}:
Total dépensé: {total_spent}€
Économies: {total_savings}€
Détail: {purchases[:20]}
Génère une analyse avec tendances et recommandations."""
        }]
    )
    return {
        "total_spent": total_spent,
        "savings": total_savings,
        "savings_pct": f"{total_savings/total_spent*100:.1f}%",
        "analysis": report.content[0].text
    }`,
            filename: "reporting.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Données fournisseurs et prix : confidentialité commerciale stricte. Accès restreint aux acheteurs autorisés. Pas de partage avec des LLM cloud publics sans accord NDA.",
      auditLog: "Chaque comparaison et recommandation tracée : fournisseurs analysés, critères de scoring, prix comparés, recommandation finale, décision acheteur.",
      humanInTheLoop: "L'IA recommande le meilleur fournisseur mais l'acheteur prend la décision finale. Validation managériale requise au-dessus de 50K€.",
      monitoring: "Économies réalisées vs prix moyen historique, temps de traitement des demandes d'achat, satisfaction des demandeurs internes, nombre de fournisseurs analysés/semaine.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Form Trigger (demande d'achat) → HTTP Request (APIs fournisseurs) → Code Node (comparaison prix) → HTTP Request LLM (analyse qualitative) → Google Sheets (tableau comparatif) → Email à l'acheteur.",
      nodes: ["Form Trigger (demande achat)", "HTTP Request (API fournisseur 1)", "HTTP Request (API fournisseur 2)", "Code Node (comparaison)", "HTTP Request (LLM analyse)", "Google Sheets", "Send Email (acheteur)"],
      triggerType: "Formulaire n8n (demande d'achat interne)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Industrie", "Retail", "Distribution"],
    metiers: ["Achats", "Supply Chain"],
    functions: ["Supply Chain"],
    metaTitle: "Agent IA d'Automatisation des Achats — Guide Supply Chain",
    metaDescription:
      "Automatisez vos processus achats avec un agent IA. Comparaison de devis, validation workflow et suivi fournisseurs.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-analyse-contrats",
    title: "Agent d'Analyse de Contrats",
    subtitle: "Analysez automatiquement vos contrats, détectez les clauses à risque et générez des redlines",
    problem:
      "Les juristes passent des heures à relire chaque contrat ligne par ligne pour identifier les clauses à risque, les écarts par rapport aux standards et les obligations cachées. Ce processus est lent, coûteux et sujet aux oublis humains, surtout lors de pics d'activité.",
    value:
      "Un agent IA scanne l'intégralité du contrat en quelques minutes, détecte les clauses problématiques en les comparant à vos standards internes, génère des redlines avec suggestions de reformulation, et produit un rapport de risque synthétique pour accélérer la négociation.",
    inputs: [
      "Document contractuel (PDF, DOCX)",
      "Bibliothèque de clauses standards internes",
      "Grille de risque juridique par type de clause",
      "Historique des négociations précédentes",
    ],
    outputs: [
      "Rapport d'analyse clause par clause avec niveau de risque",
      "Redlines générées avec suggestions de reformulation",
      "Score de risque global du contrat (0-100)",
      "Liste des obligations et échéances extraites",
      "Comparaison avec les standards internes",
    ],
    risks: [
      "Hallucination sur l'interprétation juridique d'une clause ambiguë",
      "Omission de clauses à risque dans des formulations inhabituelles",
      "Confidentialité des contrats envoyés à un LLM cloud",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de revue contractuelle. Détection de 40% de clauses à risque supplémentaires par rapport à la relecture manuelle.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Pinecone", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Contrat    │────▶│  Extraction  │────▶│  Agent LLM  │
│  (PDF/DOCX) │     │  de clauses  │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Rapport    │◀────│  Générateur  │◀────│  Vector DB  │
│  + Redlines │     │  de redlines │     │  (Standards)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires pour l'extraction de texte depuis des PDF/DOCX et la connexion au LLM Anthropic. Configurez vos clés API.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain pinecone-client pymupdf python-docx python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "clauses-standards"`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Indexation des clauses standards",
        content:
          "Créez un index vectoriel de vos clauses standards internes. Chaque clause est associée à un type (limitation de responsabilité, confidentialité, résiliation, etc.) et un niveau de risque accepté.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.document_loaders import DirectoryLoader
import json

# Charger les clauses standards
with open("clauses_standards.json", "r") as f:
    clauses = json.load(f)

docs = []
for clause in clauses:
    docs.append({
        "page_content": clause["texte"],
        "metadata": {
            "type": clause["type"],
            "risque_max": clause["risque_max"],
            "version": clause["version"]
        }
    })

embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_documents(
    docs, embeddings, index_name="clauses-standards"
)
print(f"{len(docs)} clauses standards indexées.")`,
            filename: "index_clauses.py",
          },
        ],
      },
      {
        title: "Agent d'analyse et génération de redlines",
        content:
          "Construisez l'agent principal qui extrait les clauses du contrat, les compare aux standards internes via la base vectorielle, et génère un rapport d'analyse avec des redlines pour chaque clause à risque.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from typing import List

class ClauseAnalysis(BaseModel):
    clause_text: str = Field(description="Texte original de la clause")
    risk_level: str = Field(description="Risque: faible, moyen, élevé, critique")
    issues: List[str] = Field(description="Problèmes identifiés")
    redline_suggestion: str = Field(description="Reformulation suggérée")

class ContractReport(BaseModel):
    overall_risk_score: int = Field(ge=0, le=100)
    clauses_analyzed: int
    high_risk_clauses: List[ClauseAnalysis]
    obligations: List[str]
    key_dates: List[str]

client = anthropic.Anthropic()

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    return "\\n".join([page.get_text() for page in doc])

def analyze_contract(contract_text: str, standards_context: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyse ce contrat clause par clause.
Compare chaque clause aux standards internes fournis.
Pour chaque clause à risque, génère une redline.

Standards internes:
{standards_context}

Contrat à analyser:
{contract_text}

Retourne un JSON structuré avec le rapport complet."""
        }]
    )
    return response.content[0].text`,
            filename: "agent_contrats.py",
          },
        ],
      },
      {
        title: "API et intégration",
        content:
          "Exposez l'agent via une API REST pour l'intégrer à votre workflow juridique. Le juriste upload un contrat et reçoit le rapport d'analyse en retour.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, UploadFile, File
import tempfile

app = FastAPI()

@app.post("/api/analyze-contract")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    contract_text = extract_text_from_pdf(tmp_path)
    standards = vectorstore.similarity_search(contract_text, k=10)
    context = "\\n".join([s.page_content for s in standards])
    report = analyze_contract(contract_text, context)
    return {"report": report}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contrats contiennent des données commerciales sensibles. Utiliser un LLM on-premise ou un accord de traitement des données (DPA) avec le fournisseur cloud. Chiffrement AES-256 au repos et en transit.",
      auditLog: "Chaque analyse tracée : contrat analysé (hash SHA-256), clauses détectées, scores de risque, redlines générées, juriste destinataire, horodatage complet.",
      humanInTheLoop: "Toute redline générée doit être validée par un juriste avant envoi au cocontractant. Les contrats avec un score de risque > 80 nécessitent une revue senior.",
      monitoring: "Temps moyen d'analyse par contrat, taux de clauses à risque détectées vs manquées (feedback juristes), volume de contrats traités/semaine, taux d'adoption par les équipes.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (upload contrat) → Code Node (extraction texte PDF) → HTTP Request LLM (analyse clause par clause) → Google Sheets (rapport structuré) → Email au juriste avec le rapport.",
      nodes: ["Webhook Trigger (upload)", "Code Node (extraction PDF)", "HTTP Request (LLM analyse)", "Google Sheets (rapport)", "Send Email (juriste)"],
      triggerType: "Webhook (upload de contrat)",
    },
    estimatedTime: "12-18h",
    difficulty: "Expert",
    sectors: ["Banque", "Assurance", "B2B SaaS", "Services"],
    metiers: ["Juridique", "Direction Générale"],
    functions: ["Legal"],
    metaTitle: "Agent IA d'Analyse de Contrats — Guide Juridique Complet",
    metaDescription:
      "Automatisez l'analyse de vos contrats avec un agent IA. Détection de clauses à risque, génération de redlines et rapport de conformité.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-prevision-demande",
    title: "Agent de Prévision de Demande",
    subtitle: "Anticipez la demande grâce à l'IA combinant données historiques et signaux externes",
    problem:
      "Les prévisions de demande traditionnelles reposent sur des modèles statistiques rigides qui ignorent les signaux faibles (météo, événements, réseaux sociaux). Les ruptures de stock et les surstocks coûtent des millions chaque année.",
    value:
      "Un agent IA combine vos données de vente historiques avec des signaux externes (météo, tendances Google, événements locaux, réseaux sociaux) pour produire des prévisions de demande granulaires et ajustées en temps réel. Les réapprovisionnements sont optimisés automatiquement.",
    inputs: [
      "Historique de ventes (ERP, POS)",
      "Données météorologiques par zone",
      "Calendrier événementiel et promotionnel",
      "Tendances Google Trends et réseaux sociaux",
    ],
    outputs: [
      "Prévision de demande à 7/30/90 jours par produit et zone",
      "Intervalle de confiance et scénarios (optimiste, pessimiste, médian)",
      "Alertes de rupture de stock anticipées",
      "Recommandations de réapprovisionnement automatiques",
      "Rapport d'impact des signaux externes détectés",
    ],
    risks: [
      "Données historiques incomplètes faussant les prédictions",
      "Événements exceptionnels (crise, pandémie) non modélisables",
      "Sur-confiance dans les prédictions IA sans validation terrain",
    ],
    roiIndicatif:
      "Réduction de 30% des ruptures de stock. Diminution de 20% des surstocks. Amélioration de 25% de la précision des prévisions vs méthodes traditionnelles.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL + TimescaleDB", category: "Database" },
      { name: "AWS EC2", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "DuckDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ERP/POS    │────▶│  Pipeline    │────▶│  Agent LLM  │
│  (ventes)   │     │  ETL + ML    │     │  (Analyse)  │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
┌─────────────┐     ┌──────▼───────┐     ┌──────▼──────┐
│  Signaux    │────▶│  TimescaleDB │     │  Dashboard  │
│  externes   │     │  (historique)│     │  + Alertes  │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour le traitement de séries temporelles, l'accès aux APIs de données externes et la connexion au LLM. Configurez votre base TimescaleDB.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain pandas prophet requests psycopg2-binary python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte et enrichissement des données",
        content:
          "Construisez le pipeline de collecte qui récupère les ventes historiques et les enrichit avec les signaux externes (météo, événements, tendances).",
        codeSnippets: [
          {
            language: "python",
            code: `import pandas as pd
import requests
from datetime import datetime, timedelta

def get_sales_history(product_id: str, days: int = 365) -> pd.DataFrame:
    query = f"""
    SELECT date, quantity, revenue, zone
    FROM sales
    WHERE product_id = '{product_id}'
    AND date >= NOW() - INTERVAL '{days} days'
    ORDER BY date
    """
    return pd.read_sql(query, DB_URL)

def get_weather_forecast(zone: str, days: int = 30) -> pd.DataFrame:
    resp = requests.get(
        f"https://api.openweathermap.org/data/2.5/forecast",
        params={"q": zone, "appid": WEATHER_API_KEY, "units": "metric"}
    )
    data = resp.json()
    records = [{"date": item["dt_txt"], "temp": item["main"]["temp"],
                "weather": item["weather"][0]["main"]} for item in data["list"]]
    return pd.DataFrame(records)

def enrich_with_signals(sales_df: pd.DataFrame, zone: str) -> pd.DataFrame:
    weather = get_weather_forecast(zone)
    merged = sales_df.merge(weather, on="date", how="left")
    return merged`,
            filename: "data_pipeline.py",
          },
        ],
      },
      {
        title: "Modèle de prévision hybride",
        content:
          "Combinez Prophet pour les prévisions statistiques de base avec un agent LLM qui ajuste les prédictions en tenant compte des signaux qualitatifs externes.",
        codeSnippets: [
          {
            language: "python",
            code: `from prophet import Prophet
from openai import OpenAI

client = OpenAI()

def statistical_forecast(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    prophet_df = df.rename(columns={"date": "ds", "quantity": "y"})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

def llm_adjust_forecast(forecast_data: str, signals: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Tu es un expert en prévision de demande.
Ajuste les prévisions statistiques en tenant compte des signaux externes.
Retourne un JSON avec les ajustements par date."""
        }, {
            "role": "user",
            "content": f"""Prévisions statistiques:
{forecast_data}

Signaux externes détectés:
{signals}

Ajuste les prévisions et explique chaque ajustement."""
        }]
    )
    return response.choices[0].message.content`,
            filename: "forecast_engine.py",
          },
        ],
      },
      {
        title: "API et alertes automatiques",
        content:
          "Exposez les prévisions via une API et configurez des alertes automatiques en cas de risque de rupture de stock.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class ForecastRequest(BaseModel):
    product_id: str
    zone: str
    horizon_days: int = 30

@app.post("/api/forecast")
async def forecast(req: ForecastRequest):
    sales = get_sales_history(req.product_id)
    enriched = enrich_with_signals(sales, req.zone)
    stat_forecast = statistical_forecast(enriched, req.horizon_days)
    signals = get_external_signals(req.zone, req.horizon_days)
    adjusted = llm_adjust_forecast(stat_forecast.to_json(), signals)

    stock_level = get_current_stock(req.product_id, req.zone)
    if stock_level < stat_forecast["yhat"].sum() * 0.8:
        send_restock_alert(req.product_id, req.zone, stock_level)

    return {"forecast": adjusted, "stock_alert": stock_level < stat_forecast["yhat"].sum() * 0.8}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée directement. Les données de ventes sont agrégées par produit et zone. Conformité RGPD assurée par l'anonymisation des transactions individuelles.",
      auditLog: "Chaque prévision tracée : données d'entrée (hash), signaux externes utilisés, prévision statistique brute, ajustements LLM, prévision finale, date de génération.",
      humanInTheLoop: "Les prévisions sont proposées au supply chain manager qui valide avant déclenchement des commandes. Alertes de rupture transmises pour décision humaine.",
      monitoring: "MAPE (Mean Absolute Percentage Error) par produit/zone, taux de rupture de stock, taux de surstock, précision des signaux externes, temps de génération des prévisions.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien) → HTTP Request (API ERP ventes) → HTTP Request (API météo) → Code Node (agrégation) → HTTP Request LLM (ajustement prévisions) → Google Sheets (tableau prévisionnel) → Slack alerte si rupture.",
      nodes: ["Cron Trigger (daily 6h)", "HTTP Request (ERP ventes)", "HTTP Request (API météo)", "Code Node (agrégation)", "HTTP Request (LLM ajustement)", "Google Sheets", "IF Node (seuil rupture)", "Slack Notification"],
      triggerType: "Cron (quotidien à 6h)",
    },
    estimatedTime: "16-24h",
    difficulty: "Expert",
    sectors: ["Retail", "E-commerce", "Distribution", "Industrie"],
    metiers: ["Supply Chain", "Logistique", "Direction des Opérations"],
    functions: ["Supply Chain"],
    metaTitle: "Agent IA de Prévision de Demande — Guide Supply Chain",
    metaDescription:
      "Optimisez vos prévisions de demande avec un agent IA combinant données historiques et signaux externes. Réduction des ruptures et surstocks.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-analyse-retours-clients",
    title: "Agent d'Analyse des Retours Clients",
    subtitle: "Agrégez et analysez automatiquement les retours clients pour guider vos décisions produit",
    problem:
      "Les retours clients sont éparpillés entre les avis en ligne, les tickets support, les enquêtes NPS et les réseaux sociaux. Les équipes produit n'ont pas le temps de tout lire et passent à côté de signaux critiques sur l'expérience utilisateur.",
    value:
      "Un agent IA collecte et agrège les retours clients de toutes les sources, effectue une analyse de sentiment fine, identifie les thèmes récurrents et génère des recommandations produit priorisées par impact business.",
    inputs: [
      "Avis clients (App Store, Google, Trustpilot)",
      "Tickets support et conversations chat",
      "Réponses aux enquêtes NPS/CSAT",
      "Mentions sur les réseaux sociaux",
    ],
    outputs: [
      "Dashboard de sentiment par thème et par période",
      "Top 10 des irritants clients classés par fréquence et impact",
      "Recommandations produit priorisées avec justification",
      "Alertes en temps réel sur les baisses de sentiment",
      "Rapport hebdomadaire synthétique pour le comité produit",
    ],
    risks: [
      "Biais d'échantillonnage (clients mécontents surreprésentés)",
      "Mauvaise détection du sarcasme ou de l'ironie dans les avis",
      "Recommandations produit basées sur une minorité vocale",
    ],
    roiIndicatif:
      "Réduction de 80% du temps d'analyse des retours clients. Amélioration de 15% du NPS grâce à des actions produit ciblées en 6 mois.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Avis /     │────▶│  Collecteur  │────▶│  Agent LLM  │
│  Tickets    │     │  multi-source│     │  (Sentiment) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Dashboard  │◀────│  Agrégateur  │◀────│  PostgreSQL │
│  + Alertes  │     │  de thèmes   │     │  (stockage) │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte multi-source, l'analyse de sentiment et le stockage. Configurez vos accès API aux plateformes d'avis.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary requests beautifulsoup4 python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
TRUSTPILOT_API_KEY = os.getenv("TRUSTPILOT_API_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte multi-source des retours",
        content:
          "Construisez les connecteurs pour récupérer les retours clients depuis les différentes sources : avis en ligne, tickets support, enquêtes NPS.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
from bs4 import BeautifulSoup
from datetime import datetime
import psycopg2

def collect_trustpilot_reviews(domain: str, pages: int = 5) -> list:
    reviews = []
    for page in range(1, pages + 1):
        resp = requests.get(
            f"https://api.trustpilot.com/v1/business-units/find",
            params={"name": domain},
            headers={"apikey": TRUSTPILOT_API_KEY}
        )
        for review in resp.json().get("reviews", []):
            reviews.append({
                "source": "trustpilot",
                "text": review["text"],
                "rating": review["stars"],
                "date": review["createdAt"],
            })
    return reviews

def collect_support_tickets(days: int = 30) -> list:
    query = f"""
    SELECT id, content, satisfaction_score, created_at
    FROM tickets WHERE created_at >= NOW() - INTERVAL '{days} days'
    AND status = 'resolved'
    """
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute(query)
    return [{"source": "support", "text": row[1], "rating": row[2],
             "date": str(row[3])} for row in cur.fetchall()]`,
            filename: "collectors.py",
          },
        ],
      },
      {
        title: "Analyse de sentiment et extraction de thèmes",
        content:
          "Utilisez l'agent LLM pour effectuer une analyse de sentiment fine et extraire les thèmes récurrents à partir de l'ensemble des retours collectés.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from pydantic import BaseModel, Field
from typing import List

class FeedbackAnalysis(BaseModel):
    sentiment: str = Field(description="positif, neutre, négatif")
    score: float = Field(ge=-1, le=1, description="Score de sentiment -1 à 1")
    themes: List[str] = Field(description="Thèmes identifiés")
    pain_points: List[str] = Field(description="Irritants détectés")
    suggestions: List[str] = Field(description="Suggestions d'amélioration")

client = anthropic.Anthropic()

def analyze_feedback_batch(feedbacks: list) -> list:
    batch_text = "\\n---\\n".join([f"[{f['source']}] {f['text']}" for f in feedbacks])
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyse ces retours clients. Pour chaque retour :
1. Détermine le sentiment (positif/neutre/négatif) et un score (-1 à 1)
2. Extrais les thèmes principaux
3. Identifie les irritants concrets
4. Propose des suggestions d'amélioration produit

Retours clients:
{batch_text}

Retourne un JSON structuré avec l'analyse de chaque retour."""
        }]
    )
    return response.content[0].text`,
            filename: "sentiment_analyzer.py",
          },
        ],
      },
      {
        title: "Génération de recommandations et dashboard",
        content:
          "Agrégez les analyses individuelles en un rapport synthétique avec des recommandations produit priorisées. Exposez les résultats via une API pour le dashboard.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from collections import Counter

app = FastAPI()

def generate_product_recommendations(analyses: list) -> str:
    all_themes = []
    all_pain_points = []
    for a in analyses:
        all_themes.extend(a.get("themes", []))
        all_pain_points.extend(a.get("pain_points", []))

    theme_counts = Counter(all_themes).most_common(10)
    pain_counts = Counter(all_pain_points).most_common(10)

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""En tant que Product Manager, génère des recommandations produit priorisées.

Top thèmes: {theme_counts}
Top irritants: {pain_counts}
Nombre total de retours analysés: {len(analyses)}

Priorise par impact business et fréquence. Format: tableau priorisé."""
        }]
    )
    return response.content[0].text

@app.get("/api/feedback-report")
async def get_report(days: int = 30):
    feedbacks = collect_all_sources(days)
    analyses = analyze_feedback_batch(feedbacks)
    recommendations = generate_product_recommendations(analyses)
    return {"total_feedbacks": len(feedbacks), "recommendations": recommendations}`,
            filename: "report_api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les retours clients peuvent contenir des données personnelles (noms, emails). Anonymisation automatique via regex et NER avant analyse par le LLM. Conformité RGPD avec droit à l'oubli.",
      auditLog: "Sources collectées, nombre de retours analysés, distribution de sentiment, thèmes extraits, recommandations générées, date d'exécution, destinataires du rapport.",
      humanInTheLoop: "Les recommandations produit sont soumises au Product Manager pour validation. Aucune action automatique sur le produit sans validation humaine.",
      monitoring: "Volume de retours collectés/jour par source, distribution de sentiment (tendance), précision du sentiment (échantillon validé manuellement), adoption des recommandations.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (hebdomadaire) → HTTP Request (API Trustpilot) → HTTP Request (API support) → Merge Node → HTTP Request LLM (analyse sentiment) → Code Node (agrégation) → Google Sheets (rapport) → Slack notification.",
      nodes: ["Cron Trigger (weekly)", "HTTP Request (Trustpilot)", "HTTP Request (Support API)", "Merge Node", "HTTP Request (LLM sentiment)", "Code Node (agrégation)", "Google Sheets", "Slack Notification"],
      triggerType: "Cron (hebdomadaire lundi 9h)",
    },
    estimatedTime: "6-10h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "B2B SaaS", "Retail", "Services"],
    metiers: ["Product Management", "Expérience Client"],
    functions: ["Product"],
    metaTitle: "Agent IA d'Analyse des Retours Clients — Guide Produit",
    metaDescription:
      "Analysez automatiquement vos retours clients avec un agent IA. Sentiment, thèmes récurrents et recommandations produit priorisées.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-veille-reglementaire",
    title: "Agent de Veille Réglementaire",
    subtitle: "Surveillez les évolutions réglementaires et évaluez leur impact sur votre organisation",
    problem:
      "Le paysage réglementaire évolue en permanence (EU AI Act, RGPD, DORA, NIS2). Les équipes juridiques et conformité peinent à suivre toutes les publications officielles, à interpréter leur impact et à identifier les écarts de conformité à temps.",
    value:
      "Un agent IA surveille en continu les sources réglementaires officielles (Journal Officiel, EUR-Lex, CNIL), détecte les nouvelles obligations pertinentes pour votre secteur, effectue une analyse de gap par rapport à vos pratiques actuelles et génère un plan d'action priorisé.",
    inputs: [
      "Sources réglementaires officielles (EUR-Lex, JOUE, CNIL)",
      "Registre de conformité actuel de l'entreprise",
      "Cartographie des processus métier impactés",
      "Secteur d'activité et périmètre géographique",
    ],
    outputs: [
      "Alertes en temps réel sur les nouvelles réglementations pertinentes",
      "Synthèse vulgarisée de chaque nouveau texte avec impacts identifiés",
      "Analyse de gap : écarts entre pratiques actuelles et nouvelles obligations",
      "Plan d'action priorisé avec échéances de mise en conformité",
      "Tableau de bord de conformité global avec score par domaine",
    ],
    risks: [
      "Mauvaise interprétation d'un texte juridique par le LLM",
      "Omission d'une réglementation sectorielle spécifique",
      "Faux sentiment de conformité basé sur une analyse IA incomplète",
    ],
    roiIndicatif:
      "Réduction de 60% du temps de veille réglementaire. Détection 3x plus rapide des nouvelles obligations. Économie estimée de 50K-200K€/an en évitant les sanctions pour non-conformité.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Pinecone", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  EUR-Lex /  │────▶│  Scraper +   │────▶│  Agent LLM  │
│  CNIL / JO  │     │  Détection   │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Plan       │◀────│  Gap         │◀────│  Vector DB  │
│  d'action   │     │  Analysis    │     │  (Registre) │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour le scraping des sources réglementaires, l'indexation vectorielle de votre registre de conformité et la connexion au LLM Anthropic.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain pinecone-client requests beautifulsoup4 feedparser python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
SECTORS = ["banque", "assurance", "fintech"]
MONITORED_SOURCES = [
    "https://eur-lex.europa.eu/collection/eu-law.html",
    "https://www.cnil.fr/fr/les-textes-officiels",
    "https://www.legifrance.gouv.fr/eli/jo",
]`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte et détection de nouvelles réglementations",
        content:
          "Construisez le système de surveillance qui scrape les sources réglementaires officielles, détecte les nouveaux textes et filtre ceux pertinents pour votre secteur.",
        codeSnippets: [
          {
            language: "python",
            code: `import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import hashlib

def monitor_eurlex_feed(sector_keywords: list) -> list:
    feed = feedparser.parse(
        "https://eur-lex.europa.eu/collection/eu-law/rss.xml"
    )
    new_texts = []
    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published > datetime.now() - timedelta(days=1):
            content = entry.summary.lower()
            if any(kw in content for kw in sector_keywords):
                new_texts.append({
                    "title": entry.title,
                    "url": entry.link,
                    "summary": entry.summary,
                    "date": str(published),
                    "source": "EUR-Lex",
                    "hash": hashlib.sha256(entry.link.encode()).hexdigest()
                })
    return new_texts

def monitor_cnil_publications() -> list:
    resp = requests.get("https://www.cnil.fr/fr/les-textes-officiels")
    soup = BeautifulSoup(resp.text, "html.parser")
    articles = soup.select(".article-item")
    return [{"title": a.select_one("h3").text.strip(),
             "url": "https://www.cnil.fr" + a.select_one("a")["href"],
             "source": "CNIL"} for a in articles[:10]]`,
            filename: "regulatory_monitor.py",
          },
        ],
      },
      {
        title: "Analyse de gap et génération du plan d'action",
        content:
          "Utilisez l'agent LLM pour comparer chaque nouveau texte réglementaire au registre de conformité actuel, identifier les écarts et générer un plan d'action priorisé avec des échéances.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic

client = anthropic.Anthropic()

def analyze_regulatory_gap(new_regulation: dict, current_practices: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en conformité réglementaire.

Nouveau texte réglementaire:
Titre: {new_regulation['title']}
Contenu: {new_regulation['summary']}
Source: {new_regulation['source']}

Pratiques actuelles de l'entreprise:
{current_practices}

Effectue une analyse de gap complète:
1. Résumé vulgarisé du texte (3-5 phrases)
2. Obligations nouvelles identifiées
3. Écarts par rapport aux pratiques actuelles
4. Impact (faible/moyen/élevé/critique)
5. Plan d'action avec échéances recommandées
6. Estimation du coût de mise en conformité

Retourne un JSON structuré."""
        }]
    )
    return response.content[0].text

def generate_compliance_dashboard(gap_analyses: list) -> dict:
    total = len(gap_analyses)
    critical = sum(1 for g in gap_analyses if g.get("impact") == "critique")
    high = sum(1 for g in gap_analyses if g.get("impact") == "élevé")
    return {
        "total_regulations_tracked": total,
        "critical_gaps": critical,
        "high_impact_gaps": high,
        "compliance_score": round((1 - (critical + high) / max(total, 1)) * 100),
        "next_deadlines": sorted(
            [g["deadline"] for g in gap_analyses if g.get("deadline")],
            key=lambda x: x
        )[:5]
    }`,
            filename: "gap_analyzer.py",
          },
        ],
      },
      {
        title: "API et alertes automatiques",
        content:
          "Exposez le système de veille via une API REST et configurez des alertes email automatiques pour les nouvelles réglementations à impact élevé ou critique.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import resend

app = FastAPI()

@app.get("/api/compliance-dashboard")
async def dashboard():
    return generate_compliance_dashboard(get_all_gap_analyses())

@app.post("/api/scan-regulations")
async def scan():
    new_regs = monitor_eurlex_feed(SECTORS)
    new_regs += monitor_cnil_publications()
    results = []
    for reg in new_regs:
        practices = vectorstore.similarity_search(reg["summary"], k=5)
        context = "\\n".join([p.page_content for p in practices])
        analysis = analyze_regulatory_gap(reg, context)
        results.append(analysis)
        if analysis.get("impact") in ["critique", "élevé"]:
            resend.Emails.send({
                "from": "compliance@entreprise.com",
                "to": ["legal@entreprise.com"],
                "subject": f"[ALERTE] Nouvelle réglementation: {reg['title']}",
                "html": format_alert_email(analysis)
            })
    return {"scanned": len(new_regs), "alerts_sent": len(results)}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée. Les sources sont des textes réglementaires publics. Le registre de conformité interne reste sur l'infrastructure de l'entreprise.",
      auditLog: "Chaque scan tracé : sources consultées, textes détectés, analyses de gap générées, alertes envoyées, actions de mise en conformité déclenchées, horodatage complet.",
      humanInTheLoop: "Chaque analyse de gap est validée par le responsable conformité avant communication aux équipes. Les plans d'action nécessitent une approbation de la direction juridique.",
      monitoring: "Nombre de sources surveillées, délai de détection (publication vs alerte), taux de faux positifs, avancement des plans d'action, score de conformité global.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien 7h) → HTTP Request (EUR-Lex RSS) → HTTP Request (CNIL scrape) → Merge → HTTP Request LLM (analyse de gap) → Notion (fiche réglementaire) → IF Node (impact critique) → Email alerte → Slack notification.",
      nodes: ["Cron Trigger (daily 7h)", "HTTP Request (EUR-Lex)", "HTTP Request (CNIL)", "Merge Node", "HTTP Request (LLM analyse)", "Notion Create Page", "IF Node (impact critique)", "Send Email (alerte)", "Slack Notification"],
      triggerType: "Cron (quotidien à 7h)",
    },
    estimatedTime: "14-20h",
    difficulty: "Expert",
    sectors: ["Banque", "Assurance", "B2B SaaS", "Santé", "Telecom"],
    metiers: ["Conformité", "Juridique"],
    functions: ["Legal"],
    metaTitle: "Agent IA de Veille Réglementaire — Guide Conformité",
    metaDescription:
      "Automatisez votre veille réglementaire avec un agent IA. Surveillance EU AI Act, RGPD, DORA et analyse de gap automatique.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-compte-rendu-reunion",
    title: "Agent de Compte-Rendu de Réunion",
    subtitle: "Transcrivez vos réunions, extrayez les décisions et assignez les actions automatiquement",
    problem:
      "Les comptes-rendus de réunion sont rarement rédigés, souvent incomplets et publiés trop tard. Les décisions prises et les actions assignées se perdent, entraînant des suivis inefficaces et des responsabilités floues.",
    value:
      "Un agent IA transcrit automatiquement vos réunions (audio/vidéo), identifie les décisions clés, extrait les actions avec leurs responsables et échéances, et distribue un compte-rendu structuré dans les 5 minutes suivant la fin de la réunion.",
    inputs: [
      "Enregistrement audio/vidéo de la réunion",
      "Liste des participants et leurs rôles",
      "Ordre du jour préparé en amont",
      "Comptes-rendus précédents (suivi des actions)",
    ],
    outputs: [
      "Transcription complète horodatée par intervenant",
      "Synthèse structurée de la réunion (5-10 points clés)",
      "Liste des décisions prises avec contexte",
      "Actions assignées avec responsable, échéance et priorité",
      "Email de diffusion automatique aux participants",
    ],
    risks: [
      "Erreurs de transcription sur les noms propres et termes techniques",
      "Mauvaise attribution des propos à un participant",
      "Confidentialité des discussions envoyées à un service de transcription cloud",
    ],
    roiIndicatif:
      "Gain de 30 minutes par réunion sur la rédaction. 100% des réunions documentées vs 30% avant. Suivi des actions amélioré de 50%.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "OpenAI Whisper", category: "Other" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Whisper.cpp (local)", category: "Other", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Audio /    │────▶│  Whisper     │────▶│  Agent LLM  │
│  Vidéo      │     │  (Transcr.)  │     │  (Synthèse) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Email      │◀────│  Formatteur  │◀────│  Extraction │
│  diffusion  │     │  CR structuré│     │  décisions  │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez Whisper pour la transcription audio et les dépendances pour l'analyse par LLM. Configurez vos clés API OpenAI.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain psycopg2-binary python-dotenv pydub resend`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
RESEND_API_KEY = os.getenv("RESEND_API_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Transcription audio avec Whisper",
        content:
          "Utilisez l'API Whisper d'OpenAI pour transcrire l'enregistrement audio en texte horodaté. Le modèle détecte automatiquement la langue et fournit des timestamps.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from pydub import AudioSegment
import os

client = OpenAI()

def transcribe_meeting(audio_path: str) -> dict:
    # Découper en segments de 25MB max (limite API)
    audio = AudioSegment.from_file(audio_path)
    chunk_ms = 10 * 60 * 1000  # 10 minutes
    chunks = [audio[i:i+chunk_ms] for i in range(0, len(audio), chunk_ms)]

    full_transcript = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"/tmp/chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        with open(chunk_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        for segment in result.segments:
            full_transcript.append({
                "start": segment["start"] + i * 600,
                "end": segment["end"] + i * 600,
                "text": segment["text"]
            })
        os.remove(chunk_path)
    return {"segments": full_transcript, "full_text": " ".join([s["text"] for s in full_transcript])}`,
            filename: "transcriber.py",
          },
        ],
      },
      {
        title: "Extraction des décisions et actions",
        content:
          "Utilisez le LLM pour analyser la transcription, identifier les décisions prises et extraire les actions avec responsables et échéances.",
        codeSnippets: [
          {
            language: "python",
            code: `def extract_meeting_insights(transcript: str, participants: list, agenda: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Tu es un assistant de réunion expert.
Analyse la transcription et produis un compte-rendu structuré en JSON."""
        }, {
            "role": "user",
            "content": f"""Transcription de la réunion:
{transcript}

Participants: {', '.join(participants)}
Ordre du jour: {agenda}

Extrais:
1. Synthèse en 5-10 points clés
2. Décisions prises (avec contexte et votants)
3. Actions: pour chaque action, indique le responsable, l'échéance et la priorité (haute/moyenne/basse)
4. Points en suspens à traiter lors de la prochaine réunion
5. Prochaine réunion suggérée (date, sujets)

Retourne un JSON structuré."""
        }]
    )
    return response.choices[0].message.content`,
            filename: "meeting_analyzer.py",
          },
        ],
      },
      {
        title: "Diffusion automatique du compte-rendu",
        content:
          "Formatez le compte-rendu et envoyez-le automatiquement par email à tous les participants dans les minutes suivant la fin de la réunion.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, UploadFile, File, Form
import resend
import json

app = FastAPI()
resend.api_key = RESEND_API_KEY

@app.post("/api/process-meeting")
async def process_meeting(
    audio: UploadFile = File(...),
    participants: str = Form(...),
    agenda: str = Form("")
):
    audio_path = f"/tmp/{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    transcript = transcribe_meeting(audio_path)
    participant_list = json.loads(participants)
    insights = extract_meeting_insights(
        transcript["full_text"], [p["name"] for p in participant_list], agenda
    )

    # Envoi du CR par email
    for p in participant_list:
        resend.Emails.send({
            "from": "reunions@entreprise.com",
            "to": [p["email"]],
            "subject": f"CR Réunion - {agenda[:50]}",
            "html": format_meeting_report(insights, p["name"])
        })

    return {"status": "sent", "participants": len(participant_list)}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les enregistrements audio contiennent des voix identifiables. Consentement RGPD de tous les participants requis avant enregistrement. Suppression automatique de l'audio après transcription. Stockage des CR sur infrastructure interne uniquement.",
      auditLog: "Chaque réunion traitée : participants, durée, date, nombre de décisions extraites, actions assignées, emails envoyés, horodatage de chaque étape.",
      humanInTheLoop: "Le compte-rendu peut être relu et corrigé par l'organisateur avant diffusion (mode brouillon optionnel). Les actions critiques nécessitent une confirmation du responsable assigné.",
      monitoring: "Temps de traitement par réunion, précision de la transcription (feedback utilisateurs), taux de complétion des actions assignées, satisfaction des participants sur la qualité des CR.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (fin de réunion Zoom/Teams) → HTTP Request (download audio) → HTTP Request (Whisper transcription) → HTTP Request LLM (extraction décisions) → Google Docs (CR formaté) → Send Email (participants) → Notion (suivi actions).",
      nodes: ["Webhook Trigger (fin réunion)", "HTTP Request (download audio)", "HTTP Request (Whisper API)", "HTTP Request (LLM extraction)", "Google Docs (CR)", "Send Email (tous participants)", "Notion Create (actions)"],
      triggerType: "Webhook (événement fin de réunion Zoom/Teams)",
    },
    estimatedTime: "4-6h",
    difficulty: "Facile",
    sectors: ["Tous secteurs"],
    metiers: ["Management", "Chef de Projet", "Direction"],
    functions: ["Operations"],
    metaTitle: "Agent IA de Compte-Rendu de Réunion — Guide Opérationnel",
    metaDescription:
      "Automatisez vos comptes-rendus de réunion avec un agent IA. Transcription, extraction de décisions et suivi des actions en temps réel.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-surveillance-sla",
    title: "Agent de Surveillance des SLA",
    subtitle: "Surveillez vos SLA en temps réel, prédisez les dépassements et déclenchez des escalades automatiques",
    problem:
      "Les équipes IT et support découvrent souvent les violations de SLA après coup. Le suivi manuel des dizaines de métriques contractuelles est fastidieux et les escalades arrivent trop tard, entraînant des pénalités et une insatisfaction client.",
    value:
      "Un agent IA surveille en continu les métriques de performance liées à vos SLA, prédit les risques de dépassement avant qu'ils ne surviennent, et déclenche automatiquement des escalades graduées pour prévenir les violations.",
    inputs: [
      "Métriques de performance en temps réel (API monitoring)",
      "Contrats SLA avec seuils et pénalités",
      "Historique des incidents et résolutions",
      "Planning des équipes et disponibilités",
    ],
    outputs: [
      "Dashboard temps réel du statut de chaque SLA",
      "Prédiction de risque de dépassement (score 0-100)",
      "Alertes graduées (warning, critical, breach)",
      "Escalades automatiques vers les bons interlocuteurs",
      "Rapport mensuel de performance SLA avec tendances",
    ],
    risks: [
      "Faux positifs générant une fatigue d'alerte",
      "Données de monitoring incomplètes faussant les prédictions",
      "Escalade automatique inadaptée au contexte réel de l'incident",
    ],
    roiIndicatif:
      "Réduction de 45% des violations de SLA. Détection anticipée de 80% des dépassements 2h avant la breach. Économie de 30-100K€/an en pénalités évitées.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL + TimescaleDB", category: "Database" },
      { name: "Grafana", category: "Other" },
      { name: "AWS Lambda", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Monitoring │────▶│  Collecteur  │────▶│  Agent LLM  │
│  (APIs)     │     │  métriques   │     │  (Prédiction)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Escalade   │◀────│  Moteur de   │◀────│  TimescaleDB│
│  auto       │     │  règles SLA  │     │  (historique)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de métriques, l'analyse prédictive et les notifications. Configurez vos connexions aux systèmes de monitoring.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain psycopg2-binary requests schedule python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("TIMESCALEDB_URL")
PAGERDUTY_API_KEY = os.getenv("PAGERDUTY_API_KEY")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des métriques et définition des SLA",
        content:
          "Définissez vos SLA sous forme de données structurées et construisez le système de collecte des métriques en temps réel depuis vos outils de monitoring.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

class SLADefinition(BaseModel):
    name: str
    metric: str
    threshold: float
    unit: str
    penalty_per_breach: float
    escalation_contacts: List[str]

SLA_DEFINITIONS = [
    SLADefinition(name="Uptime API", metric="availability_pct",
                  threshold=99.9, unit="%", penalty_per_breach=5000,
                  escalation_contacts=["cto@entreprise.com"]),
    SLADefinition(name="Temps de réponse P1", metric="p1_resolution_hours",
                  threshold=4, unit="heures", penalty_per_breach=2000,
                  escalation_contacts=["support-lead@entreprise.com"]),
]

def collect_metrics() -> dict:
    # Exemple: collecte depuis Datadog / Prometheus
    resp = requests.get("http://prometheus:9090/api/v1/query",
                        params={"query": "up{job='api'}"})
    metrics = resp.json()
    return {
        "availability_pct": calculate_availability(metrics),
        "p1_resolution_hours": get_avg_p1_resolution(),
        "timestamp": datetime.now().isoformat()
    }`,
            filename: "sla_monitor.py",
          },
        ],
      },
      {
        title: "Prédiction de dépassement et analyse LLM",
        content:
          "Utilisez le LLM pour analyser les tendances des métriques, prédire les risques de dépassement SLA et recommander des actions préventives.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
import json

client = OpenAI()

def predict_sla_breach(sla: SLADefinition, metrics_history: list) -> dict:
    history_str = json.dumps(metrics_history[-48:])  # 48 dernières heures
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Tu es un expert SLA/SRE.
Analyse l'historique des métriques et prédit le risque de dépassement SLA.
Retourne un JSON avec: risk_score (0-100), predicted_breach_time, root_cause_hypothesis, recommended_actions."""
        }, {
            "role": "user",
            "content": f"""SLA: {sla.name}
Seuil: {sla.threshold} {sla.unit}
Historique des métriques (48h):
{history_str}

Prédit le risque de dépassement dans les 4 prochaines heures."""
        }]
    )
    return json.loads(response.choices[0].message.content)

def escalation_engine(risk: dict, sla: SLADefinition):
    if risk["risk_score"] >= 90:
        trigger_pagerduty(sla.escalation_contacts, "CRITICAL", risk)
        send_slack_alert(f"🚨 SLA CRITICAL: {sla.name} - Breach imminente")
    elif risk["risk_score"] >= 70:
        send_slack_alert(f"⚠️ SLA WARNING: {sla.name} - Risque élevé ({risk['risk_score']}%)")`,
            filename: "breach_predictor.py",
          },
        ],
      },
      {
        title: "API et boucle de surveillance continue",
        content:
          "Mettez en place la boucle de surveillance continue qui collecte les métriques, analyse les risques et déclenche les escalades automatiquement toutes les 5 minutes.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import schedule
import threading

app = FastAPI()

def monitoring_loop():
    metrics = collect_metrics()
    store_metrics(metrics)
    for sla in SLA_DEFINITIONS:
        history = get_metrics_history(sla.metric, hours=48)
        risk = predict_sla_breach(sla, history)
        store_prediction(sla.name, risk)
        escalation_engine(risk, sla)

schedule.every(5).minutes.do(monitoring_loop)

def run_scheduler():
    while True:
        schedule.run_pending()

threading.Thread(target=run_scheduler, daemon=True).start()

@app.get("/api/sla-dashboard")
async def sla_dashboard():
    return {
        "slas": [{
            "name": sla.name,
            "current_value": get_latest_metric(sla.metric),
            "threshold": sla.threshold,
            "risk_score": get_latest_prediction(sla.name),
            "status": get_sla_status(sla)
        } for sla in SLA_DEFINITIONS]
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les métriques SLA sont des données techniques sans PII. Les contacts d'escalade sont des emails professionnels internes. Stockage sur infrastructure interne uniquement.",
      auditLog: "Chaque cycle de monitoring tracé : métriques collectées, prédictions générées, scores de risque, escalades déclenchées, temps de résolution, résultat final (breach ou non).",
      humanInTheLoop: "Les escalades critiques (risk > 90) notifient un humain qui décide de l'action. L'agent ne modifie jamais l'infrastructure directement. Les SLA sont configurés manuellement par le responsable IT.",
      monitoring: "Précision des prédictions (breach prédite vs réelle), taux de faux positifs, délai moyen entre alerte et résolution, nombre de breaches évitées/mois, coût des pénalités évitées.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 5 min) → HTTP Request (API monitoring) → Code Node (calcul métriques) → HTTP Request LLM (prédiction risque) → IF Node (seuil dépassé) → PagerDuty / Slack (escalade) → Google Sheets (log).",
      nodes: ["Cron Trigger (5 min)", "HTTP Request (Prometheus/Datadog)", "Code Node (calcul)", "HTTP Request (LLM prédiction)", "IF Node (risk > 70)", "PagerDuty Node", "Slack Notification", "Google Sheets (log)"],
      triggerType: "Cron (toutes les 5 minutes)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["B2B SaaS", "Telecom", "Services", "Banque"],
    metiers: ["SRE", "Support IT", "Infrastructure"],
    functions: ["IT"],
    metaTitle: "Agent IA de Surveillance des SLA — Guide IT/Support",
    metaDescription:
      "Surveillez vos SLA en temps réel avec un agent IA. Prédiction de dépassement, escalade automatique et reporting de performance.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-generation-propositions-commerciales",
    title: "Agent de Génération de Propositions Commerciales",
    subtitle: "Générez automatiquement des propositions commerciales personnalisées et des devis sur mesure",
    problem:
      "Les commerciaux passent 3 à 5 heures par proposition commerciale, assemblant manuellement des contenus depuis différentes sources. Les propositions manquent de personnalisation, les prix sont parfois incohérents et les délais de réponse aux appels d'offres sont trop longs.",
    value:
      "Un agent IA génère des propositions commerciales complètes et personnalisées en 30 minutes. Il adapte le contenu au contexte du prospect (secteur, taille, enjeux), calcule le pricing optimal et produit un document professionnel prêt à envoyer.",
    inputs: [
      "Fiche prospect (CRM, secteur, taille, enjeux)",
      "Catalogue produits/services avec grille tarifaire",
      "Historique des propositions gagnées/perdues",
      "Template de proposition commerciale",
    ],
    outputs: [
      "Proposition commerciale personnalisée (PDF)",
      "Devis détaillé avec pricing optimisé",
      "Arguments de vente adaptés au contexte du prospect",
      "Analyse concurrentielle ciblée",
      "Email d'accompagnement personnalisé",
    ],
    risks: [
      "Pricing incorrect ou incohérent avec la politique commerciale",
      "Promesses ou engagements non validés par la direction",
      "Données prospect obsolètes menant à une personnalisation erronée",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de rédaction des propositions. Augmentation de 20% du taux de conversion grâce à une meilleure personnalisation. Capacité de réponse x3 sur les appels d'offres.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "WeasyPrint", category: "Other" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  CRM /      │────▶│  Agent LLM   │────▶│  Générateur │
│  Prospect   │     │  (Rédaction) │     │  PDF/DOCX   │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
┌─────────────┐     ┌──────▼───────┐
│  Catalogue  │────▶│  Moteur de   │
│  Produits   │     │  pricing     │
└─────────────┘     └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la génération de documents, l'accès au CRM et la connexion au LLM Anthropic. Préparez vos templates de proposition.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary weasyprint jinja2 python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
CRM_API_KEY = os.getenv("HUBSPOT_API_KEY")
TEMPLATE_DIR = "./templates/proposals"`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des données prospect et catalogue",
        content:
          "Récupérez les informations du prospect depuis le CRM et les produits/services pertinents depuis le catalogue. L'agent utilisera ces données pour personnaliser la proposition.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
from pydantic import BaseModel, Field
from typing import List, Optional

class ProspectInfo(BaseModel):
    company: str
    sector: str
    size: str
    revenue: Optional[str]
    pain_points: List[str]
    decision_makers: List[str]
    budget_range: Optional[str]

class Product(BaseModel):
    name: str
    description: str
    base_price: float
    discount_rules: dict

def get_prospect_from_crm(deal_id: str) -> ProspectInfo:
    resp = requests.get(
        f"https://api.hubapi.com/crm/v3/objects/deals/{deal_id}",
        headers={"Authorization": f"Bearer {CRM_API_KEY}"},
        params={"associations": "companies,contacts"}
    )
    data = resp.json()
    return ProspectInfo(
        company=data["properties"]["dealname"],
        sector=data["properties"].get("industry", ""),
        size=data["properties"].get("company_size", ""),
        pain_points=data["properties"].get("pain_points", "").split(";"),
        decision_makers=[c["id"] for c in data.get("associations", {}).get("contacts", [])]
    )

def get_relevant_products(sector: str, pain_points: list) -> List[Product]:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""SELECT name, description, base_price, discount_rules
                   FROM products WHERE active = true""")
    return [Product(name=r[0], description=r[1], base_price=r[2],
                    discount_rules=r[3]) for r in cur.fetchall()]`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Génération de la proposition et du pricing",
        content:
          "Utilisez l'agent LLM pour rédiger le contenu personnalisé de la proposition, calculer le pricing optimal et assembler le document final.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def generate_proposal_content(prospect: ProspectInfo, products: list) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert commercial.
Génère une proposition commerciale personnalisée.

Prospect:
- Entreprise: {prospect.company}
- Secteur: {prospect.sector}
- Taille: {prospect.size}
- Pain points: {', '.join(prospect.pain_points)}

Produits disponibles: {json.dumps([p.model_dump() for p in products])}

Génère en JSON:
1. executive_summary: résumé exécutif personnalisé (3-4 paragraphes)
2. pain_point_analysis: analyse des enjeux du prospect
3. proposed_solution: solution recommandée avec justification
4. pricing: tableau de prix avec options (standard, premium, enterprise)
5. roi_projection: projection de ROI sur 12 mois
6. next_steps: prochaines étapes proposées
7. cover_email: email d'accompagnement personnalisé"""
        }]
    )
    return json.loads(response.content[0].text)

def calculate_optimal_pricing(products: list, prospect: ProspectInfo) -> dict:
    base_total = sum(p.base_price for p in products)
    discount = 0.1 if prospect.size == "enterprise" else 0.05
    return {
        "standard": base_total * (1 - discount),
        "premium": base_total * 1.3 * (1 - discount),
        "enterprise": base_total * 1.8 * (1 - discount),
    }`,
            filename: "proposal_generator.py",
          },
        ],
      },
      {
        title: "Génération PDF et API",
        content:
          "Assemblez le contenu dans un template HTML/CSS professionnel et convertissez-le en PDF via WeasyPrint. Exposez le tout via une API REST.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from fastapi.responses import FileResponse
from weasyprint import HTML
from jinja2 import Environment, FileSystemLoader
import tempfile

app = FastAPI()
jinja_env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

@app.post("/api/generate-proposal")
async def generate_proposal(deal_id: str):
    prospect = get_prospect_from_crm(deal_id)
    products = get_relevant_products(prospect.sector, prospect.pain_points)
    content = generate_proposal_content(prospect, products)
    pricing = calculate_optimal_pricing(products, prospect)

    template = jinja_env.get_template("proposal_template.html")
    html_content = template.render(
        prospect=prospect.model_dump(),
        content=content,
        pricing=pricing,
        date=datetime.now().strftime("%d/%m/%Y")
    )

    pdf_path = tempfile.mktemp(suffix=".pdf")
    HTML(string=html_content).write_pdf(pdf_path)

    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"proposition_{prospect.company}.pdf")`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données prospect proviennent du CRM interne. Les prix et conditions commerciales sont confidentiels. Aucune donnée de pricing ne doit être stockée dans les logs du LLM. Accès restreint aux commerciaux autorisés.",
      auditLog: "Chaque proposition tracée : prospect, produits sélectionnés, pricing généré, contenu produit, version PDF, commercial demandeur, horodatage, statut (envoyée, gagnée, perdue).",
      humanInTheLoop: "Le commercial relit et valide chaque proposition avant envoi. Les remises > 15% nécessitent une approbation du directeur commercial. Le pricing est vérifié automatiquement contre la politique commerciale.",
      monitoring: "Nombre de propositions générées/semaine, temps moyen de génération, taux de conversion des propositions IA vs manuelles, feedback des commerciaux sur la qualité, revenus générés.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouvelle opportunité CRM) → HTTP Request (données prospect HubSpot) → HTTP Request (catalogue produits) → HTTP Request LLM (génération contenu) → Code Node (pricing) → HTTP Request (génération PDF) → Email au commercial.",
      nodes: ["Webhook Trigger (opportunité CRM)", "HTTP Request (HubSpot API)", "HTTP Request (catalogue)", "HTTP Request (LLM contenu)", "Code Node (pricing)", "HTTP Request (PDF API)", "Send Email (commercial)"],
      triggerType: "Webhook (nouvelle opportunité dans le CRM)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["B2B SaaS", "Services", "Industrie", "Telecom"],
    metiers: ["Commercial", "Avant-Vente", "Business Development"],
    functions: ["Sales"],
    metaTitle: "Agent IA de Génération de Propositions Commerciales — Guide Sales",
    metaDescription:
      "Automatisez vos propositions commerciales avec un agent IA. Personnalisation, pricing optimal et génération PDF en 30 minutes.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-engagement-collaborateur",
    title: "Agent d'Analyse de l'Engagement Collaborateur",
    subtitle: "Mesurez l'engagement de vos équipes, détectez les signaux faibles et anticipez l'attrition",
    problem:
      "Les enquêtes d'engagement annuelles arrivent trop tard pour agir. Les RH manquent de visibilité sur le moral des équipes au quotidien et découvrent les départs après la démission. Les signaux faibles de désengagement passent inaperçus.",
    value:
      "Un agent IA analyse en continu les signaux d'engagement collaborateur (pulse surveys, données RH, patterns de communication) pour détecter les tendances de désengagement, prédire les risques d'attrition et recommander des actions RH ciblées en temps réel.",
    inputs: [
      "Réponses aux pulse surveys hebdomadaires (anonymisées)",
      "Données RH (ancienneté, évolution, formation, absences)",
      "Métriques de collaboration (participation réunions, activité outils)",
      "Entretiens annuels et feedbacks 360 (anonymisés)",
    ],
    outputs: [
      "Score d'engagement par équipe et département (tendance mensuelle)",
      "Détection de signaux faibles de désengagement",
      "Prédiction de risque d'attrition par segment (0-100)",
      "Recommandations RH personnalisées par manager",
      "Rapport mensuel d'engagement avec benchmarks sectoriels",
    ],
    risks: [
      "Biais algorithmique dans la prédiction d'attrition (âge, genre)",
      "Surveillance perçue comme intrusive par les collaborateurs",
      "Faux positifs générant des interventions RH non justifiées",
    ],
    roiIndicatif:
      "Réduction de 25% du turnover volontaire. Détection anticipée de 70% des départs dans les 3 mois précédents. Économie de 15-30K€ par départ évité (coût de remplacement).",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Pulse      │────▶│  Collecteur  │────▶│  Agent LLM  │
│  Surveys    │     │  + Anonymis. │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Dashboard  │◀────│  Moteur de   │◀────│  PostgreSQL │
│  RH         │     │  prédiction  │     │  (historique)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour l'analyse de sentiment, la prédiction d'attrition et le stockage des données anonymisées. Configurez les accès aux sources de données RH.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary scikit-learn pandas python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
SURVEY_TOOL_API = os.getenv("TYPEFORM_API_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte et anonymisation des données",
        content:
          "Collectez les réponses aux pulse surveys et les données RH en les anonymisant rigoureusement. Le système ne doit jamais permettre d'identifier un collaborateur individuel.",
        codeSnippets: [
          {
            language: "python",
            code: `import pandas as pd
import hashlib
import psycopg2

def anonymize_employee_id(emp_id: str, salt: str) -> str:
    return hashlib.sha256(f"{emp_id}{salt}".encode()).hexdigest()[:16]

def collect_pulse_surveys(period_days: int = 7) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    query = f"""
    SELECT anonymized_id, department, team, tenure_months,
           survey_date, engagement_score, workload_score,
           manager_score, growth_score, open_comment
    FROM pulse_surveys
    WHERE survey_date >= NOW() - INTERVAL '{period_days} days'
    """
    return pd.read_sql(query, conn)

def collect_hr_signals(department: str = None) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    query = """
    SELECT department, team, avg_tenure_months,
           absence_rate_30d, training_hours_90d,
           internal_mobility_rate, avg_meeting_participation
    FROM hr_aggregated_metrics
    WHERE aggregation_level = 'team'
    """
    if department:
        query += f" AND department = '{department}'"
    return pd.read_sql(query, conn)`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Analyse de sentiment et prédiction d'attrition",
        content:
          "Utilisez le LLM pour analyser le sentiment des commentaires anonymisés et un modèle ML pour prédire les risques d'attrition par équipe.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import json

client = anthropic.Anthropic()

def analyze_survey_sentiment(comments: list) -> list:
    batch = "\\n---\\n".join(comments)
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyse ces commentaires anonymisés de pulse survey.
Pour chaque commentaire, identifie:
1. Le sentiment (positif/neutre/négatif)
2. Les thèmes (management, charge de travail, évolution, rémunération, ambiance)
3. Les signaux faibles de désengagement
4. Urgence d'action (faible/moyenne/haute)

Commentaires:
{batch}

Retourne un JSON structuré avec l'analyse de chaque commentaire."""
        }]
    )
    return json.loads(response.content[0].text)

def predict_attrition_risk(team_metrics: pd.DataFrame) -> dict:
    features = team_metrics[["avg_tenure_months", "absence_rate_30d",
                             "training_hours_90d", "avg_meeting_participation"]].values
    # Modèle pré-entraîné sur données historiques
    model = load_attrition_model()
    risk_scores = model.predict_proba(features)[:, 1]
    return {
        "team_risks": dict(zip(team_metrics["team"].tolist(),
                               (risk_scores * 100).round(1).tolist())),
        "high_risk_teams": team_metrics[risk_scores > 0.7]["team"].tolist()
    }`,
            filename: "engagement_analyzer.py",
          },
        ],
      },
      {
        title: "Dashboard et recommandations pour les managers",
        content:
          "Exposez les résultats via une API qui alimente le dashboard RH. Générez des recommandations personnalisées pour chaque manager dont l'équipe présente des signaux de désengagement.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI

app = FastAPI()

def generate_manager_recommendations(team: str, metrics: dict, sentiment: dict) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""En tant que coach RH, génère des recommandations pour le manager.

Équipe: {team}
Métriques d'engagement: {json.dumps(metrics)}
Analyse de sentiment: {json.dumps(sentiment)}

Génère 3-5 actions concrètes et réalisables cette semaine.
Priorise par impact sur l'engagement. Sois spécifique et actionnable."""
        }]
    )
    return response.content[0].text

@app.get("/api/engagement-dashboard")
async def engagement_dashboard(department: str = None):
    surveys = collect_pulse_surveys()
    hr_data = collect_hr_signals(department)
    sentiments = analyze_survey_sentiment(surveys["open_comment"].tolist())
    attrition = predict_attrition_risk(hr_data)
    return {
        "overall_engagement_score": surveys["engagement_score"].mean(),
        "trend": calculate_trend(surveys),
        "sentiment_distribution": summarize_sentiments(sentiments),
        "attrition_risks": attrition,
        "alerts": get_high_risk_alerts(attrition, sentiments)
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Anonymisation stricte obligatoire : aucune donnée nominative ne transite par le LLM. Les pulse surveys sont anonymisés à la source. Agrégation minimale par équipe de 5+ personnes pour empêcher la ré-identification. Conformité RGPD et accord du CSE requis.",
      auditLog: "Chaque analyse tracée : période couverte, nombre de réponses analysées, scores d'engagement par département, alertes générées, recommandations produites, sans aucune donnée individuelle.",
      humanInTheLoop: "Les recommandations sont transmises aux RH qui décident des actions. Les alertes d'attrition élevée sont traitées par le HRBP en entretien confidentiel. Aucune décision automatisée impactant un collaborateur.",
      monitoring: "Taux de participation aux pulse surveys, évolution du score d'engagement, précision des prédictions d'attrition (validation à 6 mois), satisfaction des managers sur les recommandations, turnover réel vs prédit.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (hebdomadaire lundi 8h) → HTTP Request (API surveys Typeform) → HTTP Request (données RH) → Code Node (anonymisation) → HTTP Request LLM (analyse sentiment) → Google Sheets (dashboard) → Slack notification HRBP si alerte.",
      nodes: ["Cron Trigger (weekly lundi 8h)", "HTTP Request (Typeform API)", "HTTP Request (API RH)", "Code Node (anonymisation)", "HTTP Request (LLM sentiment)", "Google Sheets (dashboard)", "IF Node (alerte)", "Slack Notification (HRBP)"],
      triggerType: "Cron (hebdomadaire lundi 8h)",
    },
    estimatedTime: "10-14h",
    difficulty: "Moyen",
    sectors: ["Tous secteurs"],
    metiers: ["Ressources Humaines", "HRBP", "Direction RH"],
    functions: ["RH"],
    metaTitle: "Agent IA d'Analyse de l'Engagement Collaborateur — Guide RH",
    metaDescription:
      "Mesurez l'engagement collaborateur en continu avec un agent IA. Pulse surveys, prédiction d'attrition et recommandations RH ciblées.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-evaluation-fournisseurs",
    title: "Agent d'Évaluation des Fournisseurs",
    subtitle: "Évaluez vos fournisseurs automatiquement avec des scorecards, une détection de risques et des notations ESG",
    problem:
      "L'évaluation des fournisseurs est manuelle, chronophage et souvent incomplète. Les acheteurs jonglent entre des dizaines de critères (qualité, délais, prix, conformité, ESG) sans vision consolidée. Les risques supply chain sont détectés trop tard.",
    value:
      "Un agent IA génère des scorecards fournisseurs automatiques en agrégeant les données de performance, les audits qualité, les actualités et les notations ESG. Il détecte les risques supply chain en temps réel et recommande des actions correctives.",
    inputs: [
      "Données de performance fournisseur (ERP, qualité, délais)",
      "Rapports d'audit et certifications",
      "Actualités et presse économique du fournisseur",
      "Bases de données ESG et conformité (EcoVadis, CDP)",
    ],
    outputs: [
      "Scorecard fournisseur consolidé (qualité, coût, délai, ESG)",
      "Détection de risques supply chain avec niveau de sévérité",
      "Notation ESG actualisée avec sources vérifiées",
      "Recommandations d'action par fournisseur (reconduire, surveiller, remplacer)",
      "Rapport comparatif multi-fournisseurs par catégorie d'achat",
    ],
    risks: [
      "Données ESG incomplètes ou obsolètes faussant les notations",
      "Biais géographique dans l'évaluation des fournisseurs internationaux",
      "Mauvaise interprétation d'une actualité négative sans contexte",
    ],
    roiIndicatif:
      "Réduction de 50% du temps d'évaluation fournisseur. Détection anticipée de 60% des risques supply chain. Amélioration de 15% du score qualité moyen du panel fournisseurs en 12 mois.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Pinecone", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ERP /      │────▶│  Agrégateur  │────▶│  Agent LLM  │
│  Qualité    │     │  multi-source│     │  (Scoring)  │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
┌─────────────┐     ┌──────▼───────┐     ┌──────▼──────┐
│  ESG /      │────▶│  Vector DB   │     │  Scorecard  │
│  Actualités │     │  (historique)│     │  + Alertes  │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de données fournisseur multi-sources, l'analyse par LLM et le stockage des scorecards. Configurez vos accès aux APIs de données ESG et presse.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain pinecone-client psycopg2-binary requests beautifulsoup4 python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des données fournisseur multi-sources",
        content:
          "Construisez les connecteurs pour récupérer les données de performance depuis l'ERP, les résultats d'audit qualité, les actualités du fournisseur et les scores ESG disponibles.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional

class SupplierData(BaseModel):
    name: str
    siret: Optional[str]
    category: str
    quality_score: float
    delivery_on_time_pct: float
    avg_lead_time_days: float
    defect_rate_pct: float
    certifications: List[str]
    audit_results: List[dict]

def get_supplier_performance(supplier_id: str) -> SupplierData:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT s.name, s.siret, s.category,
               AVG(p.quality_score) as quality_score,
               AVG(p.on_time_delivery_pct) as delivery_pct,
               AVG(p.lead_time_days) as avg_lead_time,
               AVG(p.defect_rate) as defect_rate
        FROM suppliers s
        JOIN supplier_performance p ON s.id = p.supplier_id
        WHERE s.id = %s AND p.date >= NOW() - INTERVAL '12 months'
        GROUP BY s.name, s.siret, s.category
    """, (supplier_id,))
    row = cur.fetchone()
    return SupplierData(name=row[0], siret=row[1], category=row[2],
                        quality_score=row[3], delivery_on_time_pct=row[4],
                        avg_lead_time_days=row[5], defect_rate_pct=row[6],
                        certifications=get_certifications(supplier_id),
                        audit_results=get_audit_results(supplier_id))

def get_supplier_news(company_name: str, days: int = 30) -> list:
    resp = requests.get("https://newsapi.org/v2/everything",
        params={"q": company_name, "from": get_date_n_days_ago(days),
                "sortBy": "relevancy", "apiKey": NEWS_API_KEY})
    return [{"title": a["title"], "description": a["description"],
             "source": a["source"]["name"], "date": a["publishedAt"]}
            for a in resp.json().get("articles", [])[:10]]`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Scoring et analyse ESG par l'agent LLM",
        content:
          "Utilisez le LLM pour analyser l'ensemble des données collectées, générer un scorecard consolidé et évaluer les risques ESG du fournisseur.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
import json

client = OpenAI()

def generate_supplier_scorecard(supplier: SupplierData, news: list, esg_data: dict) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "system",
            "content": """Tu es un expert en évaluation fournisseurs et achats responsables.
Génère un scorecard complet et objectif basé sur les données fournies."""
        }, {
            "role": "user",
            "content": f"""Fournisseur: {supplier.name}
Catégorie: {supplier.category}

Données de performance (12 mois):
- Qualité: {supplier.quality_score}/100
- Livraison à temps: {supplier.delivery_on_time_pct}%
- Délai moyen: {supplier.avg_lead_time_days} jours
- Taux de défaut: {supplier.defect_rate_pct}%
- Certifications: {', '.join(supplier.certifications)}
- Audits: {json.dumps(supplier.audit_results)}

Actualités récentes: {json.dumps(news)}
Données ESG disponibles: {json.dumps(esg_data)}

Génère un scorecard JSON avec:
1. overall_score (0-100)
2. quality_score, delivery_score, cost_score, esg_score (0-100 chacun)
3. risk_level (faible/moyen/élevé/critique)
4. detected_risks avec sévérité et source
5. esg_rating (A/B/C/D/E) avec justification
6. recommendation (reconduire/surveiller/remplacer)
7. action_items priorisés"""
        }]
    )
    return json.loads(response.choices[0].message.content)

def detect_supply_risks(supplier: SupplierData, news: list) -> list:
    risk_keywords = ["faillite", "liquidation", "grève", "rappel produit",
                     "sanction", "pollution", "cyberattaque", "pénurie"]
    risks = []
    for article in news:
        text = f"{article['title']} {article['description']}".lower()
        matched = [kw for kw in risk_keywords if kw in text]
        if matched:
            risks.append({
                "source": article["source"],
                "title": article["title"],
                "risk_type": matched,
                "date": article["date"]
            })
    return risks`,
            filename: "supplier_scorer.py",
          },
        ],
      },
      {
        title: "API et rapport comparatif",
        content:
          "Exposez les scorecards via une API REST et générez des rapports comparatifs multi-fournisseurs pour faciliter les décisions d'achat.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from typing import List

app = FastAPI()

@app.get("/api/supplier-scorecard/{supplier_id}")
async def get_scorecard(supplier_id: str):
    supplier = get_supplier_performance(supplier_id)
    news = get_supplier_news(supplier.name)
    esg_data = get_esg_data(supplier.siret)
    scorecard = generate_supplier_scorecard(supplier, news, esg_data)
    risks = detect_supply_risks(supplier, news)
    return {"scorecard": scorecard, "risks": risks}

@app.post("/api/supplier-comparison")
async def compare_suppliers(supplier_ids: List[str], category: str):
    scorecards = []
    for sid in supplier_ids:
        supplier = get_supplier_performance(sid)
        news = get_supplier_news(supplier.name)
        esg = get_esg_data(supplier.siret)
        sc = generate_supplier_scorecard(supplier, news, esg)
        scorecards.append({"supplier": supplier.name, **sc})

    ranked = sorted(scorecards, key=lambda x: x["overall_score"], reverse=True)
    return {
        "category": category,
        "comparison": ranked,
        "recommended": ranked[0]["supplier"],
        "total_evaluated": len(ranked)
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données fournisseurs sont des données commerciales confidentielles. Accès restreint aux acheteurs autorisés par catégorie. Les scores ESG peuvent provenir de sources publiques mais les prix et conditions restent internes. Accord NDA avec le fournisseur LLM.",
      auditLog: "Chaque évaluation tracée : fournisseur évalué, sources consultées, scores calculés, risques détectés, recommandation émise, acheteur responsable, date d'évaluation, décision finale.",
      humanInTheLoop: "Les scorecards sont validés par l'acheteur catégorie avant diffusion. Les recommandations de remplacement de fournisseur nécessitent une validation du directeur achats. Les alertes risques critiques sont traitées en comité.",
      monitoring: "Nombre de fournisseurs évalués/mois, corrélation entre scores IA et performance réelle, détection de risques confirmés vs faux positifs, évolution du score qualité moyen du panel, temps gagné par évaluation.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (mensuel) → HTTP Request (ERP données fournisseurs) → HTTP Request (NewsAPI actualités) → HTTP Request (API ESG) → HTTP Request LLM (génération scorecard) → Google Sheets (tableau comparatif) → Email aux acheteurs.",
      nodes: ["Cron Trigger (monthly)", "HTTP Request (ERP fournisseurs)", "HTTP Request (NewsAPI)", "HTTP Request (API ESG)", "HTTP Request (LLM scorecard)", "Google Sheets (comparatif)", "Send Email (acheteurs)"],
      triggerType: "Cron (mensuel, 1er du mois)",
    },
    estimatedTime: "10-14h",
    difficulty: "Moyen",
    sectors: ["Industrie", "Retail", "Distribution", "Audit"],
    metiers: ["Achats", "Supply Chain", "RSE"],
    functions: ["Operations"],
    metaTitle: "Agent IA d'Évaluation des Fournisseurs — Guide Achats",
    metaDescription:
      "Automatisez l'évaluation de vos fournisseurs avec un agent IA. Scorecards, détection de risques supply chain et notation ESG automatique.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-surveillance-reputation",
    title: "Agent de Surveillance de Réputation",
    subtitle: "Surveillez votre réputation de marque en temps réel, détectez les crises et préparez des réponses",
    problem:
      "Les mentions de marque sur le web et les réseaux sociaux sont impossibles à suivre manuellement. Les crises réputationnelles se propagent en quelques heures et les équipes marketing réagissent souvent trop tard, amplifiant les dégâts d'image.",
    value:
      "Un agent IA surveille en continu les mentions de votre marque sur le web, les réseaux sociaux, les forums et les sites d'avis. Il détecte les signaux de crise en temps réel, évalue la tonalité globale et génère des templates de réponse adaptés au contexte.",
    inputs: [
      "Mentions de marque (réseaux sociaux, presse, forums, avis)",
      "Mots-clés et hashtags de surveillance configurés",
      "Historique des crises et réponses passées",
      "Charte de communication et tone of voice de la marque",
    ],
    outputs: [
      "Dashboard en temps réel des mentions avec tonalité",
      "Score de réputation global avec tendance (quotidien/hebdomadaire)",
      "Alertes de crise détectées avec niveau de sévérité",
      "Templates de réponse pré-générés adaptés au contexte",
      "Rapport hebdomadaire de veille réputationnelle",
    ],
    risks: [
      "Fausse alerte de crise sur un sujet sans impact réel",
      "Réponse automatique inadaptée au contexte émotionnel",
      "Non-détection d'une crise sur un canal non surveillé",
    ],
    roiIndicatif:
      "Détection des crises 4x plus rapide (30 min vs 2h en moyenne). Réduction de 40% de l'impact négatif grâce à une réponse rapide. Couverture de surveillance x10 vs monitoring manuel.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Réseaux    │────▶│  Collecteur  │────▶│  Agent LLM  │
│  sociaux    │     │  multi-canal │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Templates  │◀────│  Détecteur   │◀────│  PostgreSQL │
│  de réponse │     │  de crise    │     │  (historique)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de mentions multi-plateformes, l'analyse de sentiment et les notifications d'alerte. Configurez vos accès aux APIs des réseaux sociaux.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary requests tweepy python-dotenv schedule`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
NEWS_API_KEY = os.getenv("NEWSAPI_KEY")
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_URL")

BRAND_KEYWORDS = ["MaMarque", "@mamarque", "#mamarque"]
CRISIS_THRESHOLD = -0.5  # Score de sentiment déclenchant une alerte`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des mentions multi-plateformes",
        content:
          "Construisez les connecteurs pour surveiller les mentions de votre marque sur Twitter/X, les sites d'actualités, les forums et les plateformes d'avis en temps quasi-réel.",
        codeSnippets: [
          {
            language: "python",
            code: `import tweepy
import requests
from datetime import datetime, timedelta

def collect_twitter_mentions(keywords: list, hours: int = 1) -> list:
    client = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN)
    query = " OR ".join(keywords) + " -is:retweet lang:fr"
    tweets = client.search_recent_tweets(
        query=query,
        max_results=100,
        tweet_fields=["created_at", "public_metrics", "author_id"],
        start_time=datetime.utcnow() - timedelta(hours=hours)
    )
    return [{
        "platform": "twitter",
        "text": tweet.text,
        "date": str(tweet.created_at),
        "engagement": tweet.public_metrics["like_count"] + tweet.public_metrics["retweet_count"],
        "author_id": tweet.author_id
    } for tweet in (tweets.data or [])]

def collect_news_mentions(brand: str, hours: int = 24) -> list:
    resp = requests.get("https://newsapi.org/v2/everything",
        params={"q": brand, "language": "fr", "sortBy": "publishedAt",
                "from": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "apiKey": NEWS_API_KEY})
    return [{
        "platform": "news",
        "text": f"{a['title']}. {a['description']}",
        "date": a["publishedAt"],
        "source": a["source"]["name"],
        "url": a["url"]
    } for a in resp.json().get("articles", [])]

def collect_all_mentions(keywords: list) -> list:
    mentions = []
    mentions.extend(collect_twitter_mentions(keywords))
    mentions.extend(collect_news_mentions(keywords[0]))
    return sorted(mentions, key=lambda x: x["date"], reverse=True)`,
            filename: "mention_collector.py",
          },
        ],
      },
      {
        title: "Analyse de sentiment et détection de crise",
        content:
          "Utilisez l'agent LLM pour analyser le sentiment de chaque mention, détecter les patterns de crise et évaluer le niveau de risque réputationnel en temps réel.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def analyze_mentions_sentiment(mentions: list) -> dict:
    mentions_text = "\\n---\\n".join([f"[{m['platform']}] {m['text']}" for m in mentions])
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Analyse ces mentions de marque pour la veille réputationnelle.

Mentions:
{mentions_text}

Pour chaque mention, évalue:
1. sentiment (-1 à 1)
2. thème (produit, service, prix, communication, RSE, autre)
3. reach_impact (faible/moyen/élevé) basé sur l'engagement
4. requires_response (true/false)

Puis synthétise:
- overall_sentiment_score (-1 à 1)
- crisis_detected (true/false)
- crisis_level (null/faible/moyen/élevé/critique)
- trending_topics (top 5 sujets)
- volume_negative_pct (% de mentions négatives)

Retourne un JSON structuré."""
        }]
    )
    return json.loads(response.content[0].text)

def detect_crisis(analysis: dict, threshold: float = -0.3) -> dict:
    is_crisis = (
        analysis.get("crisis_detected", False) or
        analysis.get("overall_sentiment_score", 0) < threshold or
        analysis.get("volume_negative_pct", 0) > 40
    )
    return {
        "is_crisis": is_crisis,
        "level": analysis.get("crisis_level", "aucun"),
        "trigger_topics": analysis.get("trending_topics", []),
        "recommended_urgency": "immédiate" if analysis.get("crisis_level") in ["élevé", "critique"] else "standard"
    }`,
            filename: "sentiment_crisis.py",
          },
        ],
      },
      {
        title: "Génération de réponses et alertes automatiques",
        content:
          "Générez des templates de réponse adaptés au contexte de la crise ou de la mention, et mettez en place les alertes automatiques vers l'équipe communication.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import requests as http_requests
import schedule
import threading

app = FastAPI()

def generate_response_templates(crisis_context: dict, brand_voice: str) -> list:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Génère des templates de réponse de crise pour notre marque.

Contexte de crise: {json.dumps(crisis_context)}
Tone of voice de la marque: {brand_voice}

Génère 3 templates de réponse adaptés:
1. Réponse réseaux sociaux (court, empathique, 280 caractères max)
2. Communiqué de presse (formel, factuel, 3 paragraphes)
3. Réponse FAQ client (rassurante, solution-oriented)

Chaque template doit être personnalisable (placeholders entre crochets)."""
        }]
    )
    return json.loads(response.content[0].text)

def send_crisis_alert(crisis: dict, templates: list):
    http_requests.post(SLACK_WEBHOOK, json={
        "text": f"ALERTE REPUTATION - Niveau: {crisis['level']}\\n"
                f"Sujets: {', '.join(crisis['trigger_topics'])}\\n"
                f"Urgence: {crisis['recommended_urgency']}\\n"
                f"Templates de réponse disponibles dans le dashboard."
    })

def monitoring_loop():
    mentions = collect_all_mentions(BRAND_KEYWORDS)
    if not mentions:
        return
    analysis = analyze_mentions_sentiment(mentions)
    store_analysis(analysis)
    crisis = detect_crisis(analysis)
    if crisis["is_crisis"]:
        templates = generate_response_templates(crisis, get_brand_voice())
        store_templates(templates)
        send_crisis_alert(crisis, templates)

schedule.every(15).minutes.do(monitoring_loop)
threading.Thread(target=lambda: [schedule.run_pending() or __import__('time').sleep(60) for _ in iter(int, 1)], daemon=True).start()

@app.get("/api/reputation-dashboard")
async def reputation_dashboard(days: int = 7):
    return {
        "reputation_score": get_avg_sentiment(days),
        "mentions_count": get_mention_count(days),
        "sentiment_trend": get_sentiment_trend(days),
        "top_topics": get_trending_topics(days),
        "active_crisis": get_active_crises(),
        "response_templates": get_latest_templates()
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les mentions publiques sur les réseaux sociaux ne contiennent pas de PII traité par l'entreprise. Les auteurs des mentions ne sont pas stockés nominativement sauf pour les réponses directes. Conformité avec les CGU de chaque plateforme.",
      auditLog: "Chaque cycle de surveillance tracé : plateformes scannées, nombre de mentions collectées, score de sentiment, crises détectées, templates générés, alertes envoyées, réponses publiées, horodatage complet.",
      humanInTheLoop: "Les templates de réponse sont proposés à l'équipe communication qui les adapte et les valide avant publication. Aucune réponse publiée automatiquement sans validation humaine. Les alertes de crise sont traitées par le directeur communication.",
      monitoring: "Temps de détection de crise (mention -> alerte), volume de mentions/jour par plateforme, évolution du score de réputation, taux d'utilisation des templates générés, NPS de l'équipe communication sur l'outil.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 15 min) → HTTP Request (Twitter API) → HTTP Request (NewsAPI) → Merge Node → HTTP Request LLM (analyse sentiment) → IF Node (crise détectée) → HTTP Request LLM (templates réponse) → Slack alerte → Google Sheets (log).",
      nodes: ["Cron Trigger (15 min)", "HTTP Request (Twitter API)", "HTTP Request (NewsAPI)", "Merge Node", "HTTP Request (LLM sentiment)", "IF Node (crise détectée)", "HTTP Request (LLM templates)", "Slack Notification", "Google Sheets (log)"],
      triggerType: "Cron (toutes les 15 minutes)",
    },
    estimatedTime: "4-8h",
    difficulty: "Facile",
    sectors: ["Retail", "E-commerce", "Média", "B2B SaaS", "Tous secteurs"],
    metiers: ["Communication", "Marketing Digital", "Relations Publiques"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Surveillance de Réputation — Guide Marketing",
    metaDescription:
      "Surveillez votre réputation de marque en temps réel avec un agent IA. Détection de crise, analyse de sentiment et templates de réponse automatiques.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
];
