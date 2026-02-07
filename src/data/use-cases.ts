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
  {
    slug: "agent-maintenance-predictive",
    title: "Agent de Maintenance Prédictive Industrielle",
    subtitle: "Surveillez vos équipements via IoT, anticipez les pannes et planifiez la maintenance automatiquement",
    problem:
      "Les arrêts non planifiés d'équipements industriels coûtent des dizaines de milliers d'euros par heure. La maintenance préventive traditionnelle est soit trop fréquente (coûteuse) soit insuffisante (pannes imprévues). Les données des capteurs IoT sont sous-exploitées.",
    value:
      "Un agent IA connecté aux capteurs IoT surveille les équipements en temps réel, détecte les anomalies vibratoires, thermiques et acoustiques, prédit les pannes avant qu'elles ne surviennent et planifie automatiquement les interventions de maintenance au moment optimal.",
    inputs: [
      "Flux de données capteurs IoT (vibrations, température, pression, acoustique)",
      "Historique de maintenance et pannes passées (GMAO)",
      "Fiches techniques et durées de vie des composants",
      "Planning de production et contraintes d'arrêt",
    ],
    outputs: [
      "Score de santé de chaque équipement en temps réel (0-100)",
      "Prédiction de panne avec probabilité et horizon temporel",
      "Diagnostic de la cause racine probable",
      "Ordre de travail de maintenance généré automatiquement",
      "Rapport hebdomadaire de santé du parc machines",
    ],
    risks: [
      "Faux positifs générant des interventions inutiles et coûteuses",
      "Capteurs défaillants faussant les données d'entrée du modèle",
      "Sous-estimation du risque de panne critique menant à un arrêt non planifié",
    ],
    roiIndicatif:
      "Réduction de 30-40% des arrêts non planifiés, ROI sous 12 mois. Allongement de 15-20% de la durée de vie des équipements. Réduction de 25% des coûts de maintenance.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "TimescaleDB", category: "Database" },
      { name: "AWS IoT Core", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "InfluxDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Mosquitto MQTT", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Capteurs   │────▶│  Broker MQTT │────▶│  Agent LLM  │
│  IoT        │     │  (ingestion) │     │  (Prédiction)│
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  GMAO       │◀────│  Planificateur│◀────│  TimescaleDB│
│  (OT maint.)│     │  maintenance │     │  (historique)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de données IoT, l'analyse de séries temporelles et la connexion au LLM. Configurez le broker MQTT et la base de données TimescaleDB.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic paho-mqtt psycopg2-binary pandas numpy scikit-learn python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
DB_URL = os.getenv("TIMESCALEDB_URL")
GMAO_API_URL = os.getenv("GMAO_API_URL")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Ingestion des données capteurs IoT",
        content:
          "Connectez-vous au broker MQTT pour recevoir les flux de données des capteurs en temps réel et stockez-les dans TimescaleDB pour l'analyse historique.",
        codeSnippets: [
          {
            language: "python",
            code: `import paho.mqtt.client as mqtt
import psycopg2
import json
from datetime import datetime

conn = psycopg2.connect(DB_URL)

def on_message(client, userdata, msg):
    payload = json.loads(msg.payload.decode())
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sensor_readings
        (equipment_id, sensor_type, value, unit, timestamp)
        VALUES (%s, %s, %s, %s, %s)
    """, (
        payload["equipment_id"],
        payload["sensor_type"],
        payload["value"],
        payload["unit"],
        datetime.fromisoformat(payload["timestamp"])
    ))
    conn.commit()

mqtt_client = mqtt.Client()
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, MQTT_PORT)
mqtt_client.subscribe("usine/+/capteurs/#")
mqtt_client.loop_start()
print("Collecte des données capteurs en cours...")`,
            filename: "iot_collector.py",
          },
        ],
      },
      {
        title: "Détection d'anomalies et prédiction de pannes",
        content:
          "Analysez les séries temporelles des capteurs pour détecter les anomalies et utilisez le LLM pour interpréter les patterns et prédire les pannes avec leur cause racine probable.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import pandas as pd
import numpy as np
import json

client = anthropic.Anthropic()

def get_equipment_readings(equipment_id: str, hours: int = 72) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    query = f"""
        SELECT sensor_type, value, timestamp
        FROM sensor_readings
        WHERE equipment_id = %s
          AND timestamp >= NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp
    """
    return pd.read_sql(query, conn, params=(equipment_id,))

def detect_anomalies(readings: pd.DataFrame) -> dict:
    anomalies = {}
    for sensor in readings["sensor_type"].unique():
        data = readings[readings["sensor_type"] == sensor]["value"]
        mean_val = data.mean()
        std_val = data.std()
        latest = data.iloc[-1]
        z_score = abs((latest - mean_val) / std_val) if std_val > 0 else 0
        anomalies[sensor] = {
            "current": round(latest, 2),
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "z_score": round(z_score, 2),
            "is_anomaly": z_score > 2.5
        }
    return anomalies

def predict_failure(equipment_id: str, anomalies: dict, readings: pd.DataFrame) -> dict:
    stats_summary = readings.groupby("sensor_type")["value"].describe().to_string()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en maintenance prédictive industrielle.
Analyse les données capteurs de l'équipement {equipment_id}.

Anomalies détectées: {json.dumps(anomalies)}
Statistiques des capteurs (72h): {stats_summary}

Évalue:
1. health_score (0-100, 100 = parfait état)
2. failure_probability (0-1) dans les 7 prochains jours
3. estimated_failure_window (ex: "3-5 jours")
4. root_cause_hypothesis (cause racine probable)
5. recommended_action (intervention recommandée)
6. urgency (faible/moyenne/haute/critique)
7. affected_components (composants à vérifier)

Retourne un JSON structuré."""
        }]
    )
    return json.loads(response.content[0].text)`,
            filename: "failure_predictor.py",
          },
        ],
      },
      {
        title: "Planification automatique et API",
        content:
          "Planifiez automatiquement les interventions de maintenance en fonction des prédictions et des contraintes de production. Exposez les résultats via une API REST pour le dashboard et la GMAO.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import schedule
import threading
import requests

app = FastAPI()

def create_maintenance_order(equipment_id: str, prediction: dict) -> dict:
    order = {
        "equipment_id": equipment_id,
        "type": "predictive",
        "urgency": prediction["urgency"],
        "description": prediction["root_cause_hypothesis"],
        "components": prediction["affected_components"],
        "recommended_action": prediction["recommended_action"],
        "deadline": prediction["estimated_failure_window"],
        "created_by": "agent-maintenance-predictive"
    }
    # Envoi vers la GMAO
    resp = requests.post(f"{GMAO_API_URL}/work-orders", json=order)
    return resp.json()

def predictive_scan():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT equipment_id FROM sensor_readings WHERE timestamp >= NOW() - INTERVAL '1 hour'")
    equipment_ids = [row[0] for row in cur.fetchall()]

    for eq_id in equipment_ids:
        readings = get_equipment_readings(eq_id)
        anomalies = detect_anomalies(readings)
        has_anomaly = any(a["is_anomaly"] for a in anomalies.values())
        if has_anomaly:
            prediction = predict_failure(eq_id, anomalies, readings)
            store_prediction(eq_id, prediction)
            if prediction.get("urgency") in ["haute", "critique"]:
                create_maintenance_order(eq_id, prediction)
                send_alert(eq_id, prediction)

schedule.every(30).minutes.do(predictive_scan)
threading.Thread(target=lambda: [schedule.run_pending() or __import__('time').sleep(60) for _ in iter(int, 1)], daemon=True).start()

@app.get("/api/equipment-health")
async def equipment_health():
    return get_all_equipment_health_scores()

@app.get("/api/equipment/{equipment_id}/prediction")
async def get_prediction(equipment_id: str):
    readings = get_equipment_readings(equipment_id)
    anomalies = detect_anomalies(readings)
    prediction = predict_failure(equipment_id, anomalies, readings)
    return {"equipment_id": equipment_id, "anomalies": anomalies, "prediction": prediction}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée. Les données sont exclusivement des mesures de capteurs industriels (vibrations, température, pression). Stockage sur infrastructure interne ou cloud privé industriel.",
      auditLog: "Chaque cycle de scan tracé : équipements analysés, anomalies détectées, prédictions générées, ordres de maintenance créés, alertes envoyées, résultat post-intervention (panne confirmée ou non).",
      humanInTheLoop: "Les ordres de maintenance prédictive sont validés par le responsable maintenance avant exécution. Les interventions critiques nécessitent une validation du directeur de production pour planifier l'arrêt machine.",
      monitoring: "Précision des prédictions (panne prédite vs réelle), taux de faux positifs, temps moyen entre alerte et intervention, réduction des arrêts non planifiés, coût de maintenance évité par prédiction correcte.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 30 min) → MQTT Subscribe (données capteurs) → Code Node (détection anomalies) → HTTP Request LLM (prédiction panne) → IF Node (urgence haute) → HTTP Request GMAO (ordre de travail) → Slack alerte maintenance.",
      nodes: ["Cron Trigger (30 min)", "MQTT Subscribe (capteurs)", "Code Node (anomalies)", "HTTP Request (LLM prédiction)", "IF Node (urgence haute)", "HTTP Request (GMAO)", "Slack Notification"],
      triggerType: "Cron (toutes les 30 minutes)",
    },
    estimatedTime: "14-20h",
    difficulty: "Expert",
    sectors: ["Industrie", "Energie"],
    metiers: ["Maintenance", "Ingénierie", "Production"],
    functions: ["Operations"],
    metaTitle: "Agent IA de Maintenance Prédictive Industrielle — Guide Opérations",
    metaDescription:
      "Anticipez les pannes industrielles avec un agent IA connecté à vos capteurs IoT. Maintenance prédictive, détection d'anomalies et planification automatique.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-tarification-dynamique",
    title: "Agent de Tarification Dynamique",
    subtitle: "Ajustez automatiquement vos prix en fonction du stock, de la demande et de la concurrence",
    problem:
      "Les équipes pricing passent des heures à analyser manuellement les prix concurrents, les niveaux de stock et la saisonnalité. Les ajustements de prix sont lents, souvent réactifs plutôt que proactifs, et le stock dormant s'accumule faute de prix attractifs au bon moment.",
    value:
      "Un agent IA analyse en temps réel les niveaux de stock, la demande client, les prix concurrents et les tendances saisonnières pour ajuster automatiquement les prix produit par produit. La marge nette est maximisée tout en réduisant le stock dormant.",
    inputs: [
      "Niveaux de stock en temps réel (ERP/WMS)",
      "Historique de ventes et courbes de demande",
      "Prix concurrents (scraping ou API comparateurs)",
      "Calendrier saisonnier et événements commerciaux",
    ],
    outputs: [
      "Prix optimal recommandé par produit et canal de vente",
      "Score de confiance de la recommandation de prix",
      "Simulation d'impact sur la marge et le volume de ventes",
      "Alertes de prix concurrents significativement différents",
      "Rapport hebdomadaire de performance pricing (marge, rotation stock)",
    ],
    risks: [
      "Guerre des prix déclenchée par des baisses trop agressives",
      "Perception client négative sur des hausses de prix fréquentes",
      "Données concurrentielles obsolètes menant à un mauvais positionnement",
    ],
    roiIndicatif:
      "+5 à 15% de marge nette, réduction du stock dormant de 20-30%. Augmentation de 10-20% de la rotation des stocks. ROI sous 6 mois.",
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
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  ERP /      │────▶│  Agrégateur  │────▶│  Agent LLM  │
│  Stock      │     │  données     │     │  (Pricing)  │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
┌─────────────┐     ┌──────▼───────┐     ┌──────▼──────┐
│  Concurrents│────▶│  Historique  │     │  Mise à jour│
│  (scraping) │     │  ventes/prix │     │  prix (API) │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de prix concurrents, l'analyse de données et la connexion au LLM. Configurez vos accès aux systèmes ERP et e-commerce.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic psycopg2-binary pandas requests beautifulsoup4 python-dotenv schedule`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
ECOMMERCE_API_URL = os.getenv("SHOPIFY_API_URL")
ECOMMERCE_API_KEY = os.getenv("SHOPIFY_API_KEY")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des données de stock, ventes et prix concurrents",
        content:
          "Construisez les connecteurs pour récupérer les niveaux de stock, l'historique des ventes et les prix pratiqués par les concurrents. Ces données alimenteront le moteur de tarification.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
import pandas as pd
from datetime import datetime, timedelta

def get_stock_levels() -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    return pd.read_sql("""
        SELECT product_id, product_name, current_stock,
               avg_daily_sales_30d, days_of_stock,
               cost_price, current_price
        FROM inventory_dashboard
        WHERE active = true
    """, conn)

def get_sales_history(product_id: str, days: int = 90) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    return pd.read_sql("""
        SELECT sale_date, quantity, unit_price, channel
        FROM sales
        WHERE product_id = %s AND sale_date >= NOW() - INTERVAL '%s days'
        ORDER BY sale_date
    """, conn, params=(product_id, days))

def get_competitor_prices(product_name: str) -> list:
    # Exemple via une API de comparateur de prix
    resp = requests.get("https://api.comparateur.example.com/search",
        params={"q": product_name, "country": "FR"})
    results = resp.json().get("results", [])
    return [{
        "competitor": r["merchant"],
        "price": r["price"],
        "shipping": r.get("shipping_cost", 0),
        "in_stock": r.get("availability", True),
        "url": r["url"]
    } for r in results[:10]]`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Moteur de tarification dynamique avec le LLM",
        content:
          "Utilisez l'agent LLM pour analyser l'ensemble des données collectées, calculer le prix optimal pour chaque produit et simuler l'impact sur la marge et les volumes.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def calculate_optimal_price(product: dict, sales: pd.DataFrame, competitors: list) -> dict:
    sales_summary = sales.groupby("sale_date").agg(
        {"quantity": "sum", "unit_price": "mean"}
    ).tail(30).to_string()

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en pricing e-commerce et retail.
Analyse les données et recommande le prix optimal.

Produit: {product['product_name']}
- Prix actuel: {product['current_price']}EUR
- Prix de revient: {product['cost_price']}EUR
- Stock actuel: {product['current_stock']} unités
- Ventes moyennes/jour: {product['avg_daily_sales_30d']}
- Jours de stock restants: {product['days_of_stock']}

Historique des ventes (30j):
{sales_summary}

Prix concurrents: {json.dumps(competitors)}

Calcule:
1. optimal_price (prix recommandé en EUR)
2. price_range (min-max acceptable)
3. confidence_score (0-100)
4. strategy (pénétration/alignement/premium/déstockage)
5. expected_margin_pct (marge brute attendue)
6. expected_volume_change_pct (impact volume vs prix actuel)
7. reasoning (justification en 2-3 phrases)

Retourne un JSON structuré."""
        }]
    )
    return json.loads(response.content[0].text)

def apply_price_update(product_id: str, new_price: float):
    resp = requests.put(
        f"{ECOMMERCE_API_URL}/products/{product_id}.json",
        headers={"X-Shopify-Access-Token": ECOMMERCE_API_KEY},
        json={"product": {"variants": [{"price": str(new_price)}]}}
    )
    return resp.json()`,
            filename: "pricing_engine.py",
          },
        ],
      },
      {
        title: "Boucle de repricing automatique et API",
        content:
          "Mettez en place la boucle de repricing automatique qui analyse le catalogue, calcule les prix optimaux et les applique selon les règles de validation configurées.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import schedule
import threading

app = FastAPI()

PRICE_CHANGE_THRESHOLD = 0.15  # Max 15% de variation par cycle

def repricing_loop():
    stock = get_stock_levels()
    results = []
    for _, product in stock.iterrows():
        sales = get_sales_history(product["product_id"])
        competitors = get_competitor_prices(product["product_name"])
        recommendation = calculate_optimal_price(product.to_dict(), sales, competitors)

        price_change_pct = abs(recommendation["optimal_price"] - product["current_price"]) / product["current_price"]
        auto_apply = price_change_pct <= PRICE_CHANGE_THRESHOLD

        if auto_apply and recommendation["confidence_score"] >= 70:
            apply_price_update(product["product_id"], recommendation["optimal_price"])
            recommendation["applied"] = True
        else:
            recommendation["applied"] = False
            recommendation["requires_review"] = True

        results.append({"product_id": product["product_id"], **recommendation})
    store_repricing_results(results)
    return results

schedule.every(6).hours.do(repricing_loop)
threading.Thread(target=lambda: [schedule.run_pending() or __import__('time').sleep(60) for _ in iter(int, 1)], daemon=True).start()

@app.get("/api/pricing-dashboard")
async def pricing_dashboard():
    return {
        "last_run": get_last_repricing_run(),
        "products_repriced": get_repriced_count_today(),
        "avg_margin_improvement": get_avg_margin_delta(),
        "pending_reviews": get_pending_price_reviews()
    }

@app.get("/api/pricing/{product_id}")
async def get_product_pricing(product_id: str):
    product = get_product_details(product_id)
    sales = get_sales_history(product_id)
    competitors = get_competitor_prices(product["product_name"])
    return calculate_optimal_price(product, sales, competitors)`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée. Les données manipulées sont des prix, des stocks et des volumes de vente agrégés. Les prix concurrents proviennent de sources publiques. Accès restreint à l'équipe pricing.",
      auditLog: "Chaque cycle de repricing tracé : produit concerné, prix avant/après, justification de la recommandation, score de confiance, application automatique ou manuelle, impact observé sur les ventes 48h après.",
      humanInTheLoop: "Les variations de prix supérieures à 15% nécessitent une validation du responsable pricing. Les prix sous le coût de revient sont bloqués automatiquement. Le directeur commercial valide la stratégie de pricing globale.",
      monitoring: "Marge nette moyenne avant/après, taux de rotation du stock, nombre de produits repricés/semaine, taux d'acceptation des recommandations, impact sur le chiffre d'affaires, évolution du stock dormant.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 6h) → HTTP Request (ERP stocks) → HTTP Request (scraping prix concurrents) → HTTP Request LLM (calcul prix optimal) → IF Node (variation > 15%) → HTTP Request (mise à jour prix e-commerce) → Google Sheets (log repricing).",
      nodes: ["Cron Trigger (6h)", "HTTP Request (ERP stocks)", "HTTP Request (prix concurrents)", "HTTP Request (LLM pricing)", "IF Node (variation > 15%)", "HTTP Request (update prix)", "Google Sheets (log)"],
      triggerType: "Cron (toutes les 6 heures)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "Retail"],
    metiers: ["Pricing", "Category Management", "E-commerce"],
    functions: ["Sales"],
    metaTitle: "Agent IA de Tarification Dynamique — Guide Sales & E-commerce",
    metaDescription:
      "Optimisez vos prix automatiquement avec un agent IA. Analyse concurrentielle, gestion du stock dormant et maximisation de la marge nette en temps réel.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-knowledge-management",
    title: "Agent de Gestion de Base de Connaissances",
    subtitle: "Structurez votre documentation interne, détectez les contenus obsolètes et répondez aux questions des collaborateurs",
    problem:
      "La documentation interne est dispersée sur plusieurs outils (Confluence, SharePoint, Google Docs, Notion), souvent obsolète et difficile à trouver. Les collaborateurs perdent en moyenne 1h30 par jour à chercher de l'information, et les mêmes questions sont posées des dizaines de fois.",
    value:
      "Un agent IA ingère, structure et maintient automatiquement la base de connaissances interne. Il détecte les contenus obsolètes ou contradictoires, répond aux questions des collaborateurs en citant ses sources, et suggère les mises à jour nécessaires.",
    inputs: [
      "Documentation interne (Confluence, Notion, SharePoint, Google Docs)",
      "FAQ et tickets de support interne résolus",
      "Procédures et processus métier documentés",
      "Organigramme et référentiel de compétences",
    ],
    outputs: [
      "Réponses contextualisées aux questions avec sources citées",
      "Détection de contenus obsolètes avec date de dernière mise à jour",
      "Identification de contenus contradictoires entre documents",
      "Suggestions de nouveaux articles à créer (questions sans réponse)",
      "Rapport mensuel de santé de la base de connaissances",
    ],
    risks: [
      "Réponse incorrecte basée sur un document obsolète non encore détecté",
      "Hallucination du LLM inventant des procédures inexistantes",
      "Accès à des documents confidentiels par des collaborateurs non autorisés",
    ],
    roiIndicatif:
      "-40 à 60% du temps de recherche d'information, ROI sous 6 mois. Réduction de 50% des questions répétitives au support interne. Amélioration de 30% de la qualité documentaire.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
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
│  Confluence │────▶│  Indexeur     │────▶│  Vector DB  │
│  Notion...  │     │  (embedding) │     │  (Pinecone) │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Question   │────▶│  Agent LLM   │────▶│  Réponse    │
│  collabor.  │     │  (RAG)       │     │  + sources  │
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour l'indexation documentaire, le RAG (Retrieval-Augmented Generation) et la connexion aux sources de documentation. Configurez vos clés API et accès.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain pinecone-client python-dotenv requests beautifulsoup4 tiktoken`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "knowledge-base")
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Indexation de la documentation interne",
        content:
          "Connectez-vous aux sources de documentation, découpez les contenus en chunks et indexez-les dans la base vectorielle. Stockez les métadonnées pour la traçabilité des sources.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from datetime import datetime

def fetch_confluence_pages(space_key: str) -> list:
    pages = []
    url = f"{CONFLUENCE_URL}/rest/api/content"
    params = {"spaceKey": space_key, "expand": "body.storage,version", "limit": 50}
    headers = {"Authorization": f"Bearer {CONFLUENCE_TOKEN}"}
    resp = requests.get(url, params=params, headers=headers)
    for page in resp.json().get("results", []):
        pages.append({
            "title": page["title"],
            "content": page["body"]["storage"]["value"],
            "url": f"{CONFLUENCE_URL}{page['_links']['webui']}",
            "last_updated": page["version"]["when"],
            "author": page["version"]["by"]["displayName"],
            "page_id": page["id"]
        })
    return pages

def index_documents(pages: list):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = []
    for page in pages:
        chunks = splitter.split_text(page["content"])
        for i, chunk in enumerate(chunks):
            docs.append({
                "text": chunk,
                "metadata": {
                    "title": page["title"],
                    "url": page["url"],
                    "last_updated": page["last_updated"],
                    "author": page["author"],
                    "chunk_index": i
                }
            })
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone.from_texts(
        [d["text"] for d in docs],
        embeddings,
        metadatas=[d["metadata"] for d in docs],
        index_name=PINECONE_INDEX
    )
    print(f"{len(docs)} chunks indexés depuis {len(pages)} pages.")
    return vectorstore`,
            filename: "indexer.py",
          },
        ],
      },
      {
        title: "Agent RAG et détection de contenus obsolètes",
        content:
          "Implémentez l'agent de questions-réponses avec RAG qui cite ses sources, et le système de détection automatique des contenus obsolètes ou contradictoires.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
import json
from datetime import datetime, timedelta

client = anthropic.Anthropic()
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)

def answer_question(question: str, user_role: str = "collaborateur") -> dict:
    relevant_docs = vectorstore.similarity_search(question, k=5)
    context = "\\n\\n".join([
        f"[Source: {doc.metadata['title']} - Mis à jour: {doc.metadata['last_updated']}]\\n{doc.page_content}"
        for doc in relevant_docs
    ])
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Tu es un assistant de base de connaissances interne.
Réponds à la question en utilisant UNIQUEMENT les sources fournies.
Si l'information n'est pas dans les sources, dis-le clairement.
Cite toujours tes sources entre crochets.

Sources disponibles:
{context}

Question: {question}

Réponds en JSON:
1. answer: réponse détaillée avec citations [Source: titre]
2. sources: liste des sources utilisées avec URL
3. confidence: score de confiance (0-100)
4. outdated_warning: true si une source date de plus de 6 mois"""
        }]
    )
    return json.loads(response.content[0].text)

def detect_obsolete_content() -> list:
    threshold = datetime.now() - timedelta(days=180)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT title, url, last_updated, author
        FROM indexed_documents
        WHERE last_updated < %s
        ORDER BY last_updated ASC
    """, (threshold.isoformat(),))
    obsolete = [{"title": r[0], "url": r[1], "last_updated": r[2],
                 "author": r[3]} for r in cur.fetchall()]
    return obsolete`,
            filename: "knowledge_agent.py",
          },
        ],
      },
      {
        title: "API et intégration Slack",
        content:
          "Exposez l'agent de connaissances via une API REST et intégrez-le à Slack pour que les collaborateurs puissent poser leurs questions directement depuis leur messagerie.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, Request
import json

app = FastAPI()

@app.post("/api/ask")
async def ask_question(request: Request):
    body = await request.json()
    question = body["question"]
    user_role = body.get("role", "collaborateur")
    result = answer_question(question, user_role)
    store_qa_log(question, result)
    return result

@app.get("/api/obsolete-content")
async def get_obsolete_content():
    obsolete = detect_obsolete_content()
    return {"count": len(obsolete), "documents": obsolete}

@app.post("/api/slack/ask")
async def slack_command(request: Request):
    form = await request.form()
    question = form.get("text", "")
    result = answer_question(question)
    return {
        "response_type": "in_channel",
        "text": result["answer"],
        "attachments": [{
            "text": "Sources: " + ", ".join([s["title"] for s in result["sources"]]),
            "color": "#36a64f" if result["confidence"] >= 70 else "#ff9900"
        }]
    }

@app.post("/api/reindex")
async def trigger_reindex(space_key: str = "ALL"):
    pages = fetch_confluence_pages(space_key)
    index_documents(pages)
    return {"status": "indexed", "pages_count": len(pages)}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "La documentation interne peut contenir des données sensibles. Contrôle d'accès basé sur les rôles (RBAC) pour filtrer les résultats selon les permissions de l'utilisateur. Aucune donnée personnelle stockée dans l'index vectoriel. Conformité RGPD sur les données auteurs.",
      auditLog: "Chaque question tracée : question posée (anonymisée), sources consultées, réponse générée, score de confiance, feedback utilisateur (utile/non utile), documents obsolètes détectés, réindexations effectuées.",
      humanInTheLoop: "Les réponses avec un score de confiance inférieur à 50% affichent un avertissement et proposent de contacter un expert humain. Les suggestions de contenus obsolètes sont validées par le propriétaire du document avant archivage.",
      monitoring: "Nombre de questions/jour, taux de résolution (réponse utile), temps de réponse moyen, couverture documentaire (questions sans réponse), nombre de documents obsolètes détectés/mois, satisfaction utilisateur (NPS).",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (question Slack/API) → HTTP Request (recherche vectorielle Pinecone) → HTTP Request LLM (génération réponse RAG) → Slack Reply (réponse) → Google Sheets (log QA). Cron hebdomadaire → HTTP Request (scan obsolescence) → Email (rapport).",
      nodes: ["Webhook Trigger (question)", "HTTP Request (Pinecone search)", "HTTP Request (LLM RAG)", "Slack Reply", "Google Sheets (log)", "Cron Trigger (weekly)", "HTTP Request (scan obsolescence)", "Send Email (rapport)"],
      triggerType: "Webhook (question Slack ou API) + Cron (hebdomadaire)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Services", "Banque", "Assurance"],
    metiers: ["Knowledge Manager", "Support Interne", "IT"],
    functions: ["IT"],
    metaTitle: "Agent IA de Gestion de Base de Connaissances — Guide IT & Knowledge",
    metaDescription:
      "Structurez et maintenez votre documentation interne avec un agent IA. RAG, détection de contenus obsolètes et réponses instantanées aux collaborateurs.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-optimisation-energetique",
    title: "Agent d'Optimisation Énergétique",
    subtitle: "Analysez et réduisez vos consommations énergétiques en temps réel grâce à l'IA",
    problem:
      "Les entreprises industrielles et de distribution peinent à maîtriser leurs consommations énergétiques. Les factures augmentent, les réglementations se durcissent (décret tertiaire, taxonomie EU) et les données de consommation sont sous-exploitées. Les ajustements manuels sont trop lents face aux variations de prix de l'énergie.",
    value:
      "Un agent IA analyse les consommations énergétiques en temps réel (électricité, gaz, eau), identifie les gaspillages, ajuste automatiquement les usages (HVAC, éclairage, process) et optimise les achats d'énergie en fonction des tarifs dynamiques. L'empreinte carbone est réduite sans impact sur la productivité.",
    inputs: [
      "Données de consommation en temps réel (compteurs intelligents, sous-compteurs)",
      "Tarifs énergétiques dynamiques (RTE, marché spot)",
      "Données météo et prévisions (température, ensoleillement)",
      "Planning de production et occupation des bâtiments",
    ],
    outputs: [
      "Dashboard de consommation en temps réel par zone et usage",
      "Détection automatique des anomalies et gaspillages",
      "Consignes d'ajustement automatique HVAC et éclairage",
      "Prévision de consommation et coûts pour les 7 prochains jours",
      "Rapport mensuel d'empreinte carbone avec évolution",
    ],
    risks: [
      "Ajustement HVAC trop agressif impactant le confort des occupants",
      "Données de compteurs défaillantes menant à des optimisations erronées",
      "Non-prise en compte de contraintes process industriel dans les ajustements",
    ],
    roiIndicatif:
      "-10 à 25% des coûts énergétiques, -10 à 20% d'empreinte carbone. Conformité facilitée avec le décret tertiaire. ROI sous 12-18 mois selon le volume de consommation.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "TimescaleDB", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "InfluxDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Compteurs  │────▶│  Collecteur  │────▶│  Agent LLM  │
│  intelligts │     │  énergie     │     │  (Analyse)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌──────▼──────┐
│  Automates  │◀────│  Optimiseur  │◀────│  TimescaleDB│
│  HVAC/GTB   │     │  consignes   │     │  (historique)│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour la collecte de données énergétiques, l'analyse de séries temporelles et la connexion aux automates de gestion technique du bâtiment (GTB). Configurez les accès aux compteurs et APIs tarifaires.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic psycopg2-binary pandas numpy requests python-dotenv schedule`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("TIMESCALEDB_URL")
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
GTB_API_URL = os.getenv("GTB_API_URL")
RTE_API_TOKEN = os.getenv("RTE_API_TOKEN")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des données de consommation et contexte",
        content:
          "Connectez-vous aux compteurs intelligents, récupérez les tarifs énergétiques en temps réel et les prévisions météo pour alimenter le moteur d'optimisation.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
import pandas as pd
from datetime import datetime

def get_energy_consumption(hours: int = 24) -> pd.DataFrame:
    conn = psycopg2.connect(DB_URL)
    return pd.read_sql(f"""
        SELECT meter_id, zone, energy_type, value_kwh, timestamp
        FROM energy_readings
        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
        ORDER BY timestamp
    """, conn)

def get_spot_prices() -> dict:
    resp = requests.get("https://digital.iservices.rte-france.com/open_api/wholesale_market/v2/france/spot",
        headers={"Authorization": f"Bearer {RTE_API_TOKEN}"})
    data = resp.json()
    return {
        "current_price_mwh": data["spot_prices"][-1]["value"],
        "next_hours": [{"hour": p["period"], "price": p["value"]}
                       for p in data["spot_prices"][-24:]]
    }

def get_weather_forecast(lat: float, lon: float) -> dict:
    resp = requests.get("https://api.openweathermap.org/data/2.5/forecast",
        params={"lat": lat, "lon": lon, "appid": WEATHER_API_KEY, "units": "metric"})
    forecasts = resp.json().get("list", [])
    return [{
        "datetime": f["dt_txt"],
        "temp": f["main"]["temp"],
        "humidity": f["main"]["humidity"],
        "clouds": f["clouds"]["all"]
    } for f in forecasts[:16]]  # 48h

def get_building_occupancy() -> dict:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT zone, current_occupancy, max_capacity
        FROM building_occupancy WHERE updated_at >= NOW() - INTERVAL '1 hour'
    """)
    return {row[0]: {"occupancy": row[1], "capacity": row[2]} for row in cur.fetchall()}`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Analyse et recommandations d'optimisation",
        content:
          "Utilisez le LLM pour analyser les patterns de consommation, détecter les anomalies et générer des consignes d'optimisation adaptées au contexte (météo, occupation, tarifs).",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def analyze_and_optimize(consumption: pd.DataFrame, prices: dict, weather: list, occupancy: dict) -> dict:
    consumption_summary = consumption.groupby(["zone", "energy_type"]).agg(
        {"value_kwh": ["sum", "mean", "max"]}
    ).to_string()

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en efficacité énergétique industrielle et tertiaire.
Analyse les données de consommation et recommande des optimisations.

Consommation (24h par zone):
{consumption_summary}

Tarifs énergie: {json.dumps(prices)}
Prévisions météo (48h): {json.dumps(weather)}
Occupation des zones: {json.dumps(occupancy)}

Analyse et retourne un JSON avec:
1. anomalies: gaspillages détectés avec zone, type et estimation kWh perdus
2. hvac_adjustments: consignes HVAC par zone (température cible, ventilation)
3. lighting_adjustments: ajustements éclairage par zone
4. load_shifting: recommandations de décalage de charge vers heures creuses
5. estimated_savings_kwh: économie estimée sur 24h
6. estimated_savings_eur: économie estimée en euros
7. carbon_reduction_kg: réduction CO2 estimée
8. comfort_impact: impact sur le confort (aucun/faible/modéré)"""
        }]
    )
    return json.loads(response.content[0].text)

def apply_hvac_adjustments(adjustments: list):
    for adj in adjustments:
        requests.post(f"{GTB_API_URL}/zones/{adj['zone']}/hvac", json={
            "target_temperature": adj["target_temperature"],
            "ventilation_mode": adj["ventilation_mode"],
            "source": "agent-optimisation-energetique"
        })`,
            filename: "energy_optimizer.py",
          },
        ],
      },
      {
        title: "Boucle d'optimisation continue et reporting",
        content:
          "Mettez en place la boucle d'optimisation qui tourne en continu, ajuste les consignes et produit les rapports de performance énergétique et d'empreinte carbone.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
import schedule
import threading

app = FastAPI()

CARBON_FACTOR_KWH = 0.052  # kg CO2/kWh (mix FR moyen)

def optimization_loop():
    consumption = get_energy_consumption(hours=24)
    prices = get_spot_prices()
    weather = get_weather_forecast(48.8566, 2.3522)
    occupancy = get_building_occupancy()

    analysis = analyze_and_optimize(consumption, prices, weather, occupancy)
    store_analysis(analysis)

    if analysis.get("comfort_impact") in ["aucun", "faible"]:
        apply_hvac_adjustments(analysis.get("hvac_adjustments", []))

    if analysis.get("anomalies"):
        send_anomaly_alert(analysis["anomalies"])

    return analysis

schedule.every(30).minutes.do(optimization_loop)
threading.Thread(target=lambda: [schedule.run_pending() or __import__('time').sleep(60) for _ in iter(int, 1)], daemon=True).start()

@app.get("/api/energy-dashboard")
async def energy_dashboard():
    consumption = get_energy_consumption(hours=24)
    return {
        "total_kwh_today": consumption["value_kwh"].sum(),
        "by_zone": consumption.groupby("zone")["value_kwh"].sum().to_dict(),
        "current_spot_price": get_spot_prices()["current_price_mwh"],
        "carbon_footprint_kg": consumption["value_kwh"].sum() * CARBON_FACTOR_KWH,
        "savings_today": get_today_savings(),
        "active_optimizations": get_active_adjustments()
    }

@app.get("/api/energy-report/{period}")
async def energy_report(period: str = "monthly"):
    return generate_energy_report(period)`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle traitée. Les données sont des mesures de consommation énergétique par zone, sans identification individuelle. Les données d'occupation sont agrégées par zone sans suivi individuel.",
      auditLog: "Chaque cycle d'optimisation tracé : consommations relevées, tarifs appliqués, consignes envoyées, économies estimées vs réelles, anomalies détectées, ajustements HVAC effectués, empreinte carbone calculée.",
      humanInTheLoop: "Les ajustements impactant le confort (niveau modéré) nécessitent une validation du facility manager. Les modifications de process industriel sont toujours validées par le responsable production. Les seuils de confort sont configurés par zone.",
      monitoring: "Consommation kWh avant/après optimisation, coût énergétique mensuel, empreinte carbone (scope 1+2), taux de confort (plaintes occupants), précision des prévisions de consommation, conformité décret tertiaire.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 30 min) → HTTP Request (compteurs énergie) → HTTP Request (RTE tarifs spot) → HTTP Request (météo) → HTTP Request LLM (analyse + optimisation) → HTTP Request (GTB ajustements HVAC) → Google Sheets (log) → Slack si anomalie.",
      nodes: ["Cron Trigger (30 min)", "HTTP Request (compteurs)", "HTTP Request (RTE tarifs)", "HTTP Request (météo)", "HTTP Request (LLM optimisation)", "HTTP Request (GTB HVAC)", "Google Sheets (log)", "Slack (alerte anomalie)"],
      triggerType: "Cron (toutes les 30 minutes)",
    },
    estimatedTime: "14-20h",
    difficulty: "Expert",
    sectors: ["Industrie", "Distribution"],
    metiers: ["Energy Manager", "Facility Management", "RSE"],
    functions: ["Operations"],
    metaTitle: "Agent IA d'Optimisation Énergétique — Guide Opérations & RSE",
    metaDescription:
      "Réduisez vos coûts énergétiques et votre empreinte carbone avec un agent IA. Analyse temps réel, ajustement HVAC automatique et reporting carbone.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-planification-logistique",
    title: "Agent de Planification Logistique Terrain",
    subtitle: "Optimisez les tournées et plannings de vos équipes terrain en temps réel",
    problem:
      "La planification des tournées et des équipes terrain (techniciens, livreurs, commerciaux) est un casse-tête quotidien. Les aléas (trafic, météo, absences, urgences) rendent les plannings obsolètes dès le matin. Les équipes perdent du temps en trajets inutiles et les clients subissent des retards.",
    value:
      "Un agent IA recalcule en continu les tournées et plannings des équipes terrain en fonction des aléas en temps réel. Il optimise les trajets, réaffecte les interventions en cas d'absence et prévient les clients automatiquement en cas de changement d'horaire.",
    inputs: [
      "Liste des interventions planifiées avec adresses et durées",
      "Disponibilité et compétences des techniciens/livreurs",
      "Données trafic en temps réel et prévisions météo",
      "Historique des interventions et temps de trajet réels",
    ],
    outputs: [
      "Planning optimisé par technicien avec itinéraire détaillé",
      "Replanification automatique en cas d'aléa (temps réel)",
      "Notifications clients avec créneau horaire précis",
      "Estimation des temps de trajet et heures d'arrivée",
      "Rapport quotidien de performance logistique (km, interventions, ponctualité)",
    ],
    risks: [
      "Replanification trop fréquente perturbant les équipes terrain",
      "Données trafic imprécises menant à des estimations erronées",
      "Non-prise en compte de contraintes terrain spécifiques (accès, matériel)",
    ],
    roiIndicatif:
      "-15 à 25% des coûts de transport, ROI sous 6-12 mois. Amélioration de 20-30% de la ponctualité des interventions. +15% d'interventions réalisées par jour par technicien.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Interven-  │────▶│  Optimiseur  │────▶│  Agent LLM  │
│  tions      │     │  tournées    │     │  (Replanif.) │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
┌─────────────┐     ┌──────▼───────┐     ┌──────▼──────┐
│  Trafic /   │────▶│  PostgreSQL  │     │  Notif.     │
│  Météo      │     │  (plannings) │     │  client/tech│
└─────────────┘     └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances pour le calcul d'itinéraires, la gestion des plannings et la connexion au LLM. Configurez vos accès aux APIs de géolocalisation et de trafic.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic psycopg2-binary requests python-dotenv schedule geopy`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import os
from dotenv import load_dotenv
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
SMS_API_KEY = os.getenv("TWILIO_AUTH_TOKEN")`,
            filename: "config.py",
          },
        ],
      },
      {
        title: "Collecte des données et calcul des distances",
        content:
          "Récupérez les interventions planifiées, les disponibilités des techniciens et les données de trafic en temps réel. Calculez la matrice de distances et de temps de trajet entre chaque point.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import psycopg2
import pandas as pd
from datetime import datetime

def get_daily_interventions(date: str) -> list:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, client_name, address, lat, lon,
               estimated_duration_min, required_skills,
               priority, time_window_start, time_window_end
        FROM interventions
        WHERE scheduled_date = %s AND status = 'planned'
        ORDER BY priority DESC
    """, (date,))
    return [{"id": r[0], "client": r[1], "address": r[2],
             "lat": r[3], "lon": r[4], "duration_min": r[5],
             "skills": r[6], "priority": r[7],
             "window_start": str(r[8]), "window_end": str(r[9])}
            for r in cur.fetchall()]

def get_available_technicians(date: str) -> list:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, name, skills, home_lat, home_lon,
               max_km_per_day, start_time, end_time
        FROM technicians
        WHERE id NOT IN (SELECT tech_id FROM absences WHERE date = %s)
    """, (date,))
    return [{"id": r[0], "name": r[1], "skills": r[2],
             "home_lat": r[3], "home_lon": r[4],
             "max_km": r[5], "start": str(r[6]), "end": str(r[7])}
            for r in cur.fetchall()]

def get_travel_time(origin: tuple, destination: tuple) -> dict:
    resp = requests.get("https://maps.googleapis.com/maps/api/distancematrix/json",
        params={
            "origins": f"{origin[0]},{origin[1]}",
            "destinations": f"{destination[0]},{destination[1]}",
            "departure_time": "now",
            "key": GOOGLE_MAPS_API_KEY
        })
    element = resp.json()["rows"][0]["elements"][0]
    return {
        "distance_km": element["distance"]["value"] / 1000,
        "duration_min": element["duration_in_traffic"]["value"] / 60
    }`,
            filename: "data_collector.py",
          },
        ],
      },
      {
        title: "Optimisation des tournées avec le LLM",
        content:
          "Utilisez l'agent LLM pour optimiser l'affectation des interventions aux techniciens et l'ordre des tournées en tenant compte de toutes les contraintes (compétences, fenêtres horaires, trafic, météo).",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def optimize_routes(interventions: list, technicians: list, weather: dict) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en optimisation logistique et planification de tournées.
Optimise l'affectation et l'ordre des interventions pour minimiser les km et maximiser la ponctualité.

Interventions à planifier: {json.dumps(interventions)}
Techniciens disponibles: {json.dumps(technicians)}
Conditions météo: {json.dumps(weather)}

Contraintes:
- Respecter les compétences requises par intervention
- Respecter les fenêtres horaires clients
- Respecter le kilométrage max par technicien
- Respecter les horaires de travail des techniciens
- Prioriser les interventions de haute priorité
- Ajouter 15% de marge sur les temps de trajet si pluie/neige

Retourne un JSON:
1. routes: pour chaque technicien, liste ordonnée des interventions avec heure d'arrivée estimée
2. unassigned: interventions non affectables (avec raison)
3. total_km: km total pour toutes les tournées
4. total_interventions: nombre d'interventions planifiées
5. estimated_completion_rate: taux de réalisation estimé
6. optimization_notes: remarques et suggestions"""
        }]
    )
    return json.loads(response.content[0].text)

def replan_on_disruption(current_plan: dict, disruption: dict) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Replanifie les tournées suite à un aléa.

Plan actuel: {json.dumps(current_plan)}
Aléa survenu: {json.dumps(disruption)}

Minimise les changements tout en maintenant la ponctualité.
Retourne le plan mis à jour au même format, avec un champ changes_summary."""
        }]
    )
    return json.loads(response.content[0].text)`,
            filename: "route_optimizer.py",
          },
        ],
      },
      {
        title: "API, notifications et suivi en temps réel",
        content:
          "Exposez le service de planification via une API REST, envoyez les notifications aux clients et aux techniciens, et mettez en place le suivi en temps réel des tournées.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, Request
import schedule
import threading
import requests as http_requests

app = FastAPI()

def notify_client(client_phone: str, tech_name: str, eta: str):
    http_requests.post("https://api.twilio.com/2010-04-01/Accounts/ACXXX/Messages.json",
        auth=("ACXXX", SMS_API_KEY),
        data={
            "From": "+33XXXXXXXXX",
            "To": client_phone,
            "Body": f"Bonjour, votre technicien {tech_name} arrivera vers {eta}. "
                    f"Vous recevrez une notification 30 min avant son arrivée."
        })

def morning_planning():
    today = datetime.now().strftime("%Y-%m-%d")
    interventions = get_daily_interventions(today)
    technicians = get_available_technicians(today)
    weather = get_weather_conditions()
    plan = optimize_routes(interventions, technicians, weather)
    store_plan(today, plan)

    for route in plan.get("routes", []):
        for intervention in route.get("interventions", []):
            notify_client(
                intervention["client_phone"],
                route["technician_name"],
                intervention["eta"]
            )
    return plan

schedule.every().day.at("06:30").do(morning_planning)
threading.Thread(target=lambda: [schedule.run_pending() or __import__('time').sleep(60) for _ in iter(int, 1)], daemon=True).start()

@app.post("/api/disruption")
async def report_disruption(request: Request):
    disruption = await request.json()
    current_plan = get_today_plan()
    new_plan = replan_on_disruption(current_plan, disruption)
    store_plan(datetime.now().strftime("%Y-%m-%d"), new_plan)
    notify_affected_clients(current_plan, new_plan)
    return {"status": "replanned", "changes": new_plan.get("changes_summary")}

@app.get("/api/routes/today")
async def get_today_routes():
    plan = get_today_plan()
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "routes": plan.get("routes", []),
        "total_km": plan.get("total_km"),
        "completion_rate": get_realtime_completion_rate()
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données clients (noms, adresses, téléphones) sont nécessaires pour les interventions. Stockage sécurisé en base interne. Les données de géolocalisation des techniciens sont utilisées uniquement pendant les heures de travail. Conformité RGPD avec consentement client.",
      auditLog: "Chaque planification tracée : interventions affectées, tournées générées, km estimés vs réels, replanifications effectuées, raison des changements, notifications envoyées, taux de ponctualité par technicien.",
      humanInTheLoop: "Le responsable logistique valide le planning matinal avant envoi aux techniciens. Les replanifications mineures sont automatiques, les majeures (>30% de changements) nécessitent une validation humaine. Les techniciens peuvent signaler des contraintes terrain.",
      monitoring: "Km parcourus vs optimaux, taux de ponctualité, nombre d'interventions/jour/technicien, taux de replanification, satisfaction client (enquête post-intervention), coût de transport par intervention.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (6h30 chaque matin) → HTTP Request (interventions du jour) → HTTP Request (disponibilités techniciens) → HTTP Request (trafic Google Maps) → HTTP Request LLM (optimisation tournées) → Webhook (envoi planning techniciens) → SMS notifications clients.",
      nodes: ["Cron Trigger (6h30)", "HTTP Request (interventions)", "HTTP Request (techniciens)", "HTTP Request (Google Maps)", "HTTP Request (LLM optimisation)", "Webhook (planning)", "Twilio SMS (notifications)"],
      triggerType: "Cron (quotidien à 6h30)",
    },
    estimatedTime: "4-8h",
    difficulty: "Facile",
    sectors: ["Distribution", "Services"],
    metiers: ["Logistique", "Planification", "Direction Opérations"],
    functions: ["Supply Chain"],
    metaTitle: "Agent IA de Planification Logistique Terrain — Guide Supply Chain",
    metaDescription:
      "Optimisez les tournées de vos équipes terrain avec un agent IA. Replanification en temps réel, réduction des km et amélioration de la ponctualité.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-conformite-fiscale",
    title: "Agent de Conformité Fiscale et Optimisation TVA",
    subtitle: "Automatisez la catégorisation TVA, la surveillance réglementaire et les déclarations fiscales grâce à l'IA",
    problem:
      "La complexité croissante des règles TVA (OSS/IOSS, facturation électronique obligatoire en France 2026) expose les entreprises à des erreurs de catégorisation coûteuses. Un simple écart de taux ou une mauvaise affectation de régime fiscal peut déclencher un redressement fiscal majeur, avec pénalités et intérêts de retard.",
    value:
      "Un agent IA catégorise automatiquement chaque transaction selon le bon régime TVA, surveille en continu les évolutions réglementaires (OSS, IOSS, e-invoicing), pré-remplit les déclarations TVA et génère des alertes en cas d'anomalie détectée. Le risque de redressement est drastiquement réduit.",
    inputs: [
      "Flux de transactions (ERP, e-commerce, POS)",
      "Référentiel réglementaire TVA (taux, régimes, seuils)",
      "Paramètres entreprise (régime fiscal, pays, SIRET)",
      "Historique des déclarations TVA précédentes",
    ],
    outputs: [
      "Transactions catégorisées avec taux TVA appliqué",
      "Déclaration TVA pré-remplie (CA3, OSS, IOSS)",
      "Rapport d'anomalies et écarts détectés",
      "Alertes réglementaires (changements de taux, nouvelles obligations)",
      "Score de conformité global par période",
    ],
    risks: [
      "Erreur de catégorisation sur des régimes TVA complexes (triangulaires, autoliquidation)",
      "Retard dans la prise en compte d'un changement réglementaire",
      "Dépendance excessive à l'automatisation sans vérification humaine des déclarations",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de préparation des déclarations TVA. Diminution de 90% des erreurs de catégorisation. Économie moyenne de 15K€/an en pénalités évitées pour une PME.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Transactions│────▶│  Agent LLM   │────▶│ Déclaration │
│  (ERP/POS)  │     │ (Catégoris.) │     │  TVA prête  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │ Référentiel  │
                    │  TVA / Veille│
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires et configurez l'accès à l'API Anthropic. Préparez votre référentiel de taux TVA et régimes fiscaux par pays.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary python-dotenv fastapi uvicorn`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/fiscal
VAT_REFERENCE_PATH=./data/vat_rates.json`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Référentiel TVA et modèles de données",
        content:
          "Définissez les modèles de données pour les transactions, les régimes TVA et les résultats de catégorisation. Le référentiel doit couvrir tous les pays et régimes applicables (OSS, IOSS, autoliquidation).",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from enum import Enum
from datetime import date

class VATRegime(str, Enum):
    STANDARD = "standard"
    REDUCED = "réduit"
    SUPER_REDUCED = "super_réduit"
    EXEMPT = "exonéré"
    OSS = "oss"
    IOSS = "ioss"
    REVERSE_CHARGE = "autoliquidation"

class TransactionCategory(BaseModel):
    transaction_id: str
    vat_regime: VATRegime
    vat_rate: float = Field(ge=0, le=30)
    country_code: str
    reasoning: str
    confidence: float = Field(ge=0, le=1)
    alerts: list[str] = []

class VATDeclaration(BaseModel):
    period: str
    total_ht: float
    total_tva: float
    lines: list[dict]
    compliance_score: float = Field(ge=0, le=100)
    anomalies: list[str] = []`,
            filename: "models.py",
          },
        ],
      },
      {
        title: "Agent de catégorisation TVA",
        content:
          "Construisez l'agent qui analyse chaque transaction, détermine le régime TVA applicable et détecte les anomalies. L'agent utilise le référentiel réglementaire comme contexte pour ses décisions.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json

client = anthropic.Anthropic()

def load_vat_reference():
    with open("./data/vat_rates.json") as f:
        return json.load(f)

def categorize_transaction(transaction: dict, reference: dict) -> TransactionCategory:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en fiscalité TVA européenne.
Analyse cette transaction et détermine le régime TVA applicable.

Transaction: {json.dumps(transaction, ensure_ascii=False)}
Référentiel TVA: {json.dumps(reference, ensure_ascii=False)}

Retourne un JSON avec: transaction_id, vat_regime, vat_rate,
country_code, reasoning, confidence, alerts (liste d'anomalies)."""
        }]
    )
    result = json.loads(message.content[0].text)
    return TransactionCategory(**result)

def batch_categorize(transactions: list[dict]) -> list[TransactionCategory]:
    reference = load_vat_reference()
    return [categorize_transaction(tx, reference) for tx in transactions]`,
            filename: "agent_fiscal.py",
          },
        ],
      },
      {
        title: "API de déclaration et monitoring",
        content:
          "Exposez l'agent via une API REST. L'endpoint principal catégorise un lot de transactions et génère un brouillon de déclaration TVA. Un dashboard permet de suivre le score de conformité en temps réel.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class BatchRequest(BaseModel):
    transactions: list[dict]
    period: str
    company_id: str

@app.post("/api/vat/categorize")
async def categorize(req: BatchRequest):
    results = batch_categorize(req.transactions)
    anomalies = [r for r in results if r.confidence < 0.8 or r.alerts]
    total_ht = sum(tx.get("amount_ht", 0) for tx in req.transactions)
    total_tva = sum(
        tx.get("amount_ht", 0) * (r.vat_rate / 100)
        for tx, r in zip(req.transactions, results)
    )
    return {
        "period": req.period,
        "total_transactions": len(results),
        "total_ht": total_ht,
        "total_tva": round(total_tva, 2),
        "anomalies": [a.model_dump() for a in anomalies],
        "compliance_score": round(
            100 * (1 - len(anomalies) / max(len(results), 1)), 1
        )
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données de facturation (noms, adresses, SIRET) sont stockées en base interne chiffrée. Seules les données agrégées et anonymisées sont envoyées au LLM pour la catégorisation. Conformité RGPD et secret fiscal respectés.",
      auditLog: "Chaque catégorisation tracée : transaction ID, régime TVA appliqué, taux, score de confiance, horodatage, version du référentiel utilisé. Piste d'audit complète pour contrôle fiscal.",
      humanInTheLoop: "Les transactions avec un score de confiance < 0.8 ou présentant des anomalies sont soumises au comptable pour validation. Les déclarations TVA finales nécessitent une approbation du directeur financier avant soumission.",
      monitoring: "Dashboard : volume de transactions catégorisées/jour, score de conformité moyen, nombre d'anomalies détectées, alertes réglementaires actives, écart TVA collectée vs déclarée.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien) → HTTP Request (nouvelles transactions ERP) → HTTP Request LLM (catégorisation TVA) → IF anomalie détectée → Email alerte comptable + Mise à jour PostgreSQL → Cron mensuel → Génération déclaration TVA.",
      nodes: ["Cron Trigger (quotidien)", "HTTP Request (ERP transactions)", "HTTP Request (LLM catégorisation)", "IF (anomalie)", "Email (alerte comptable)", "PostgreSQL (mise à jour)", "Cron Trigger (mensuel déclaration)"],
      triggerType: "Cron (quotidien + mensuel)",
    },
    estimatedTime: "10-16h",
    difficulty: "Expert",
    sectors: ["E-commerce", "Retail", "Services"],
    metiers: ["Comptabilité", "Direction Financière", "Fiscalité"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Conformité Fiscale et Optimisation TVA — Guide Expert",
    metaDescription:
      "Automatisez la catégorisation TVA et la conformité fiscale avec un agent IA. OSS, IOSS, facturation électronique : tutoriel complet et ROI détaillé.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-soc-cybersecurite",
    title: "Agent SOC Autonome de Cybersécurité",
    subtitle: "Détectez, corrélez et répondez aux menaces cyber automatiquement grâce à un agent IA SOC",
    problem:
      "Les SOC (Security Operations Centers) sont submergés par des milliers d'alertes par jour, dont 80% sont des faux positifs. Les analystes peinent à traiter le volume, les menaces réelles sont noyées dans le bruit. La directive NIS2 impose désormais des délais de détection et de notification stricts.",
    value:
      "Un agent IA corrèle les événements de sécurité multi-sources (SIEM, EDR, firewall), élimine les faux positifs, conduit une investigation automatisée sur les alertes suspectes, applique des remédiations prédéfinies et escalade les incidents critiques avec un dossier pré-constitué complet.",
    inputs: [
      "Flux d'alertes SIEM (Splunk, Elastic, Sentinel)",
      "Logs EDR et firewall",
      "Base de Threat Intelligence (IoC, TTPs MITRE ATT&CK)",
      "Playbooks de réponse à incidents",
    ],
    outputs: [
      "Alertes corrélées et dédupliquées avec score de sévérité",
      "Rapport d'investigation automatisé (timeline, IoC, impact)",
      "Actions de remédiation exécutées (blocage IP, isolation endpoint)",
      "Dossier d'escalade pré-constitué pour l'analyste L2/L3",
      "Métriques SOC : MTTD, MTTR, taux de faux positifs",
    ],
    risks: [
      "Faux négatif : menace réelle classée comme bénigne par l'agent",
      "Remédiation automatique trop agressive causant un déni de service interne",
      "Exfiltration de données sensibles via les prompts envoyés au LLM",
    ],
    roiIndicatif:
      "Réduction de 85% du volume d'alertes à traiter manuellement. MTTD (Mean Time To Detect) divisé par 4. Économie de 2 à 3 ETP analystes SOC L1.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Elasticsearch", category: "Database" },
      { name: "AWS", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Wazuh", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Docker self-hosted", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ SIEM/EDR    │────▶│  Agent LLM   │────▶│ Remédiation │
│  Alertes    │     │ (Corrélation)│     │ / Escalade  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   Threat     │
                    │ Intelligence │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez les accès aux sources de données de sécurité. L'agent nécessite un accès en lecture au SIEM et en écriture pour les actions de remédiation.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain elasticsearch python-dotenv fastapi uvicorn requests`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
ELASTICSEARCH_URL=https://siem.internal:9200
ELASTICSEARCH_API_KEY=...
MITRE_ATTACK_DB=./data/mitre_attack.json
PLAYBOOKS_PATH=./playbooks/`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Corrélation d'événements et triage",
        content:
          "Construisez le module de corrélation qui agrège les alertes multi-sources, élimine les doublons et les faux positifs évidents, puis soumet les alertes suspectes à l'agent LLM pour investigation approfondie.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch

client = anthropic.Anthropic()
es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))

def fetch_recent_alerts(minutes: int = 15) -> list[dict]:
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": f"now-{minutes}m",
                    "lte": "now"
                }
            }
        },
        "size": 500,
        "sort": [{"@timestamp": "desc"}]
    }
    result = es.search(index="siem-alerts-*", body=query)
    return [hit["_source"] for hit in result["hits"]["hits"]]

def correlate_and_triage(alerts: list[dict]) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un analyste SOC expert.
Analyse ces {len(alerts)} alertes de sécurité.

Alertes: {json.dumps(alerts[:50], ensure_ascii=False, default=str)}

Pour chaque groupe d'alertes corrélées, retourne un JSON avec:
- alert_group_id, severity (critical/high/medium/low/false_positive)
- correlated_events (liste des IDs), attack_technique (MITRE ATT&CK)
- summary, recommended_action, requires_escalation (bool)"""
        }]
    )
    return json.loads(message.content[0].text)`,
            filename: "soc_correlator.py",
          },
        ],
      },
      {
        title: "Investigation automatisée et remédiation",
        content:
          "L'agent conduit une investigation approfondie sur les alertes à haute sévérité : enrichissement IoC, analyse de la kill chain, et exécution des playbooks de remédiation prédéfinis.",
        codeSnippets: [
          {
            language: "python",
            code: `def investigate_alert_group(alert_group: dict) -> dict:
    # Enrichissement via Threat Intelligence
    iocs = extract_iocs(alert_group)
    ti_results = enrich_iocs(iocs)

    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Investigation approfondie requise.

Groupe d'alertes: {json.dumps(alert_group, ensure_ascii=False)}
Enrichissement Threat Intel: {json.dumps(ti_results, ensure_ascii=False)}

Produis un rapport d'investigation complet:
- timeline (chronologie des événements)
- iocs_confirmed (IoC confirmés malveillants)
- attack_chain (étapes MITRE ATT&CK identifiées)
- impact_assessment (systèmes affectés, données à risque)
- remediation_actions (actions immédiates recommandées)
- escalation_brief (résumé pour analyste L3)"""
        }]
    )
    report = json.loads(message.content[0].text)

    # Exécution des remédiations automatiques si playbook existe
    if report.get("remediation_actions"):
        for action in report["remediation_actions"]:
            if action.get("auto_executable"):
                execute_playbook(action["playbook_id"], action["params"])

    return report`,
            filename: "soc_investigator.py",
          },
        ],
      },
      {
        title: "API SOC et dashboard",
        content:
          "Exposez l'agent via une API REST intégrée à votre stack SOC. Le pipeline tourne en continu, traitant les alertes par lots toutes les 15 minutes et exposant les métriques clés.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/api/soc/status")
async def soc_status():
    alerts = fetch_recent_alerts(minutes=60)
    triaged = correlate_and_triage(alerts)
    critical = [g for g in triaged.get("groups", [])
                if g["severity"] == "critical"]
    return {
        "total_alerts_1h": len(alerts),
        "groups_identified": len(triaged.get("groups", [])),
        "critical_incidents": len(critical),
        "false_positive_rate": triaged.get("false_positive_rate", 0),
        "mttd_minutes": triaged.get("avg_detection_time", 0)
    }

@app.post("/api/soc/investigate")
async def investigate(alert_group_id: str):
    alert_group = get_alert_group(alert_group_id)
    report = investigate_alert_group(alert_group)
    return report`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle n'est envoyée au LLM : seuls les métadonnées d'alertes (IPs, hashes, timestamps) sont transmises. Les logs bruts restent dans le SIEM interne. Chiffrement TLS pour tous les flux. Conformité NIS2 et ISO 27001.",
      auditLog: "Chaque corrélation, investigation et remédiation tracée : alertes traitées, sévérité assignée, actions exécutées, temps de détection, temps de réponse, analyste notifié, playbook déclenché.",
      humanInTheLoop: "Les remédiations critiques (isolation réseau, blocage utilisateur) nécessitent une approbation humaine. Les incidents de sévérité critique sont immédiatement escaladés vers l'analyste L3 avec dossier complet. Seuil configurable par type d'action.",
      monitoring: "MTTD (Mean Time To Detect), MTTR (Mean Time To Respond), taux de faux positifs, volume d'alertes traitées/heure, nombre de remédiations automatiques, taux d'escalade, couverture MITRE ATT&CK.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 15 min) → HTTP Request (alertes SIEM) → HTTP Request LLM (corrélation et triage) → IF sévérité critique → HTTP Request (investigation) → HTTP Request (remédiation) → Slack/PagerDuty (escalade).",
      nodes: ["Cron Trigger (15 min)", "HTTP Request (SIEM alertes)", "HTTP Request (LLM corrélation)", "IF (sévérité critique)", "HTTP Request (investigation)", "HTTP Request (remédiation)", "PagerDuty (escalade)"],
      triggerType: "Cron (toutes les 15 minutes)",
    },
    estimatedTime: "16-24h",
    difficulty: "Expert",
    sectors: ["Banque", "Santé", "Telecom"],
    metiers: ["RSSI", "SOC Analyst", "Direction IT"],
    functions: ["IT"],
    metaTitle: "Agent SOC Autonome de Cybersécurité — Guide Expert IA",
    metaDescription:
      "Déployez un agent IA SOC autonome pour détecter, corréler et répondre aux cybermenaces. Corrélation SIEM, investigation automatisée et conformité NIS2.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-qa-logicielle",
    title: "Agent de QA Logicielle Autonome",
    subtitle: "Générez, exécutez et analysez vos tests logiciels automatiquement grâce à l'IA",
    problem:
      "Les cycles de développement accélérés font des tests le goulet d'étranglement de la delivery. Le code généré par IA nécessite encore plus de vérification. Les équipes QA n'arrivent pas à suivre le rythme des releases, les régressions passent en production.",
    value:
      "Un agent IA génère automatiquement des plans de test à partir du code et des spécifications, exécute les tests dans le pipeline CI/CD, détecte les régressions et produit des rapports enrichis avec analyse d'impact. La couverture de test augmente sans effort manuel.",
    inputs: [
      "Code source et diff des pull requests",
      "Spécifications fonctionnelles (tickets Jira, docs)",
      "Historique des tests et bugs précédents",
      "Configuration CI/CD (GitHub Actions, GitLab CI)",
    ],
    outputs: [
      "Plan de test généré (cas de test, scénarios edge cases)",
      "Scripts de test exécutables (pytest, Playwright, Jest)",
      "Rapport d'exécution avec couverture et régressions détectées",
      "Analyse d'impact des changements sur les modules existants",
      "Score de qualité du code et recommandations",
    ],
    risks: [
      "Tests générés superficiels manquant des edge cases critiques",
      "Faux sentiment de sécurité lié à une couverture de test élevée mais peu pertinente",
      "Coût API élevé sur des codebases volumineuses si chaque PR déclenche une analyse complète",
    ],
    roiIndicatif:
      "Augmentation de 60% de la couverture de test. Réduction de 45% des régressions en production. Gain de 2 jours/sprint pour l'équipe QA.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "GitHub Actions", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + CodeLlama", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "GitLab CI (self-hosted)", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Pull Req.  │────▶│  Agent LLM   │────▶│  CI/CD      │
│  (code diff)│     │ (Génér.tests)│     │ (exécution) │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Historique  │
                    │ tests & bugs │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez l'accès au dépôt Git et à l'API Anthropic. L'agent s'intègre comme étape dans votre pipeline CI/CD existant.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain gitpython pytest python-dotenv fastapi`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
GITHUB_TOKEN=ghp_...
REPO_PATH=./my-project
TEST_OUTPUT_DIR=./generated_tests`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Analyse de diff et génération de tests",
        content:
          "L'agent analyse le diff d'une pull request, identifie les fonctions modifiées et génère des cas de test couvrant les chemins nominaux, les edge cases et les régressions potentielles.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
from git import Repo

client = anthropic.Anthropic()

def get_pr_diff(repo_path: str, base: str, head: str) -> str:
    repo = Repo(repo_path)
    diff = repo.git.diff(f"{base}...{head}")
    return diff

def generate_tests(diff: str, spec: str = "") -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un ingénieur QA expert.
Analyse ce diff et génère des tests complets.

Diff:
{diff[:8000]}

Spécifications: {spec if spec else "Non fournies"}

Retourne un JSON avec:
- test_plan: liste de cas de test (description, type: unit/integration/e2e)
- test_code: code pytest exécutable
- edge_cases: scénarios limites identifiés
- regression_risks: risques de régression sur les modules existants
- coverage_estimate: estimation de la couverture ajoutée"""
        }]
    )
    return json.loads(message.content[0].text)`,
            filename: "agent_qa.py",
          },
        ],
      },
      {
        title: "Exécution CI/CD et rapport",
        content:
          "Intégrez l'agent dans votre pipeline CI/CD. À chaque pull request, l'agent génère les tests, les exécute via pytest et produit un rapport enrichi avec analyse d'impact.",
        codeSnippets: [
          {
            language: "python",
            code: `import subprocess
import json

def run_generated_tests(test_code: str, output_dir: str) -> dict:
    # Écrire les tests générés
    test_file = f"{output_dir}/test_generated.py"
    with open(test_file, "w") as f:
        f.write(test_code)

    # Exécuter avec pytest
    result = subprocess.run(
        ["pytest", test_file, "--json-report", "--json-report-file=report.json", "-v"],
        capture_output=True, text=True
    )

    with open("report.json") as f:
        report = json.load(f)

    return {
        "passed": report["summary"]["passed"],
        "failed": report["summary"]["failed"],
        "errors": report["summary"].get("error", 0),
        "duration": report["duration"],
        "details": report.get("tests", []),
        "stdout": result.stdout[-2000:]
    }

def generate_quality_report(test_results: dict, diff: str) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Analyse les résultats de test et le diff.
Résultats: {json.dumps(test_results)}
Diff: {diff[:4000]}

Produis un rapport qualité: quality_score (0-100),
regressions_detected, recommendations, safe_to_merge (bool)."""
        }]
    )
    return json.loads(message.content[0].text)`,
            filename: "ci_runner.py",
          },
        ],
      },
      {
        title: "Intégration GitHub Actions",
        content:
          "Configurez un workflow GitHub Actions qui déclenche l'agent QA automatiquement sur chaque pull request. Le rapport est posté en commentaire sur la PR.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
import requests

app = FastAPI()

class PRWebhook(BaseModel):
    action: str
    pr_number: int
    base_branch: str
    head_branch: str
    repo: str

@app.post("/api/qa/webhook")
async def handle_pr(webhook: PRWebhook):
    if webhook.action not in ["opened", "synchronize"]:
        return {"status": "skipped"}

    diff = get_pr_diff(webhook.repo, webhook.base_branch, webhook.head_branch)
    test_plan = generate_tests(diff)
    results = run_generated_tests(test_plan["test_code"], "./generated_tests")
    report = generate_quality_report(results, diff)

    # Poster le rapport en commentaire sur la PR
    requests.post(
        f"https://api.github.com/repos/{webhook.repo}/issues/{webhook.pr_number}/comments",
        headers={"Authorization": f"token {GITHUB_TOKEN}"},
        json={"body": format_report_markdown(report)}
    )
    return {"status": "completed", "quality_score": report["quality_score"]}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Le code source est analysé en mémoire sans persistance externe. Les diffs envoyés au LLM sont tronqués pour exclure les fichiers sensibles (.env, credentials). Liste d'exclusion configurable. Aucune donnée client dans les tests générés.",
      auditLog: "Chaque exécution tracée : PR analysée, tests générés, résultats d'exécution, score qualité, recommandations, décision merge/block, temps d'analyse, coût API.",
      humanInTheLoop: "L'agent ne merge jamais automatiquement. Le rapport qualité est informatif. Les tests avec un score < 70 bloquent la PR et nécessitent une revue manuelle par le Tech Lead.",
      monitoring: "Couverture de test par module, taux de régressions détectées vs passées en prod, temps moyen de génération de tests, coût API par PR, score qualité moyen par équipe.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (GitHub PR event) → HTTP Request (récupération diff) → HTTP Request LLM (génération tests) → Execute Command (pytest) → HTTP Request LLM (rapport qualité) → HTTP Request (commentaire GitHub PR).",
      nodes: ["Webhook (GitHub PR)", "HTTP Request (diff)", "HTTP Request (LLM génération tests)", "Execute Command (pytest)", "HTTP Request (LLM rapport)", "HTTP Request (GitHub commentaire)"],
      triggerType: "Webhook (événement Pull Request)",
    },
    estimatedTime: "6-10h",
    difficulty: "Moyen",
    sectors: ["B2B SaaS", "E-commerce", "Banque"],
    metiers: ["QA Engineer", "Tech Lead", "CTO"],
    functions: ["IT"],
    metaTitle: "Agent IA de QA Logicielle Autonome — Guide Complet",
    metaDescription:
      "Automatisez vos tests logiciels avec un agent IA. Génération de tests, exécution CI/CD et détection de régressions. Tutoriel pas-à-pas.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-negociation-achats",
    title: "Agent de Négociation Achats Indirects",
    subtitle: "Analysez vos contrats, benchmarkez les prix et négociez automatiquement vos achats indirects",
    problem:
      "Les achats indirects représentent 15 à 30% des dépenses d'une entreprise mais sont rarement renégociés faute de temps et de données comparatives. Des millions d'euros de savings sont laissés sur la table chaque année : contrats reconduits tacitement, prix jamais challengés, fournisseurs alternatifs non évalués.",
    value:
      "Un agent IA analyse vos contrats en cours, effectue un benchmark tarifaire automatique, identifie les opportunités de savings et mène des négociations autonomes par email avec les fournisseurs. Il présente des recommandations avec options à valider par le décideur.",
    inputs: [
      "Contrats fournisseurs en cours (PDF, ERP)",
      "Historique des dépenses par catégorie d'achats",
      "Données de benchmark tarifaire (bases sectorielles, web)",
      "Politique achats et seuils de validation internes",
    ],
    outputs: [
      "Cartographie des dépenses avec potentiel de savings par catégorie",
      "Benchmark tarifaire comparatif (prix actuels vs marché)",
      "Emails de négociation générés et envoyés aux fournisseurs",
      "Recommandations de renégociation avec options chiffrées",
      "Suivi des négociations en cours et résultats obtenus",
    ],
    risks: [
      "Benchmark biaisé par des données de marché incomplètes ou obsolètes",
      "Ton de négociation inapproprié pouvant détériorer la relation fournisseur",
      "Engagement contractuel non autorisé si les garde-fous de validation sont contournés",
    ],
    roiIndicatif:
      "Savings moyen de 8 à 15% sur les achats indirects renégociés. ROI typique de 5x à 10x le coût de l'outil dès la première année. Gain de 3 jours/mois pour l'équipe achats.",
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
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Contrats   │────▶│  Agent LLM   │────▶│  Emails de  │
│  & Dépenses │     │ (Benchmark)  │     │ négociation │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Benchmark   │
                    │  tarifaire   │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez les accès. L'agent nécessite un accès aux contrats (PDF) et à l'historique des dépenses (ERP ou export CSV).",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary python-dotenv fastapi pymupdf pandas`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/achats
SMTP_HOST=smtp.company.com
SMTP_USER=achats@company.com
SMTP_PASSWORD=...`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Extraction et analyse de contrats",
        content:
          "L'agent extrait les informations clés des contrats fournisseurs (montants, échéances, clauses de reconduction, conditions tarifaires) et les structure dans une base de données pour analyse.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
import fitz  # PyMuPDF

client = anthropic.Anthropic()

def extract_contract_data(pdf_path: str) -> dict:
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])

    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Analyse ce contrat fournisseur et extrais les informations clés.

Contrat:
{text[:6000]}

Retourne un JSON avec:
- supplier_name, contract_id, start_date, end_date
- auto_renewal (bool), notice_period_days
- total_annual_value, payment_terms
- key_line_items: liste de (description, unit_price, quantity, annual_total)
- negotiation_levers: points de négociation identifiés
- renewal_deadline: date limite pour renégocier"""
        }]
    )
    return json.loads(message.content[0].text)`,
            filename: "contract_extractor.py",
          },
        ],
      },
      {
        title: "Benchmark et stratégie de négociation",
        content:
          "L'agent compare vos prix actuels aux données de marché, identifie les écarts et génère une stratégie de négociation adaptée à chaque fournisseur avec des arguments chiffrés.",
        codeSnippets: [
          {
            language: "python",
            code: `def generate_negotiation_strategy(contract: dict, benchmark: dict) -> dict:
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert en achats indirects.
Analyse ce contrat et ce benchmark pour définir une stratégie de négociation.

Contrat actuel: {json.dumps(contract, ensure_ascii=False)}
Benchmark marché: {json.dumps(benchmark, ensure_ascii=False)}

Retourne un JSON avec:
- savings_potential_pct: économie estimée en %
- savings_potential_eur: économie estimée en EUR/an
- negotiation_strategy: approche recommandée
- arguments: liste d'arguments de négociation chiffrés
- email_draft: brouillon d'email de négociation professionnel
- options: 3 scénarios (conservateur, modéré, ambitieux)
  avec pour chacun: target_saving, probability, risk_level
- alternative_suppliers: fournisseurs alternatifs à mentionner"""
        }]
    )
    return json.loads(message.content[0].text)`,
            filename: "negotiation_engine.py",
          },
        ],
      },
      {
        title: "API et suivi des négociations",
        content:
          "Exposez l'agent via une API. Le décideur valide la stratégie et les emails avant envoi. L'agent suit les réponses fournisseurs et adapte sa stratégie en fonction.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
import smtplib
from email.mime.text import MIMEText

app = FastAPI()

class NegotiationRequest(BaseModel):
    contract_id: str
    strategy_option: str  # conservateur, modéré, ambitieux
    approved_by: str

@app.post("/api/achats/negotiate")
async def start_negotiation(req: NegotiationRequest):
    contract = get_contract(req.contract_id)
    benchmark = get_benchmark(contract["category"])
    strategy = generate_negotiation_strategy(contract, benchmark)

    selected = next(
        o for o in strategy["options"]
        if o["level"] == req.strategy_option
    )
    return {
        "contract_id": req.contract_id,
        "strategy": strategy["negotiation_strategy"],
        "email_draft": strategy["email_draft"],
        "target_saving": selected["target_saving"],
        "status": "pending_approval",
        "approved_by": req.approved_by
    }

@app.post("/api/achats/send-email")
async def send_negotiation_email(contract_id: str, approved: bool):
    if not approved:
        return {"status": "cancelled"}
    negotiation = get_negotiation(contract_id)
    send_email(
        to=negotiation["supplier_email"],
        subject=f"Revue contrat {contract_id}",
        body=negotiation["email_draft"]
    )
    return {"status": "email_sent"}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contrats et données fournisseurs sont stockés en base interne chiffrée. Les montants et conditions contractuelles envoyés au LLM sont agrégés sans mention du nom de l'entreprise. Conformité RGPD pour les contacts fournisseurs.",
      auditLog: "Chaque analyse tracée : contrat analysé, benchmark effectué, stratégie générée, option choisie, email approuvé/envoyé, réponse fournisseur, saving obtenu, approbateur identifié.",
      humanInTheLoop: "L'agent ne peut jamais envoyer un email ou accepter une offre sans validation explicite du responsable achats. Chaque stratégie est présentée avec 3 options. Seuils de validation hiérarchiques selon les montants en jeu.",
      monitoring: "Savings obtenus vs estimés par catégorie, nombre de contrats renégociés, taux de réponse fournisseurs, délai moyen de négociation, pipeline de contrats à échéance, ROI global du programme achats.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (hebdomadaire) → HTTP Request (contrats à échéance < 90 jours) → HTTP Request LLM (analyse et benchmark) → HTTP Request LLM (stratégie négociation) → Email (notification responsable achats) → Wait approval → SMTP (envoi email fournisseur).",
      nodes: ["Cron Trigger (hebdomadaire)", "HTTP Request (contrats ERP)", "HTTP Request (LLM analyse)", "HTTP Request (LLM stratégie)", "Email (notification interne)", "Wait (approbation)", "SMTP (email fournisseur)"],
      triggerType: "Cron (hebdomadaire)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Services", "Banque", "Industrie"],
    metiers: ["Direction Achats", "Operations", "Direction Financière"],
    functions: ["Operations"],
    metaTitle: "Agent IA de Négociation Achats Indirects — Guide Complet",
    metaDescription:
      "Optimisez vos achats indirects avec un agent IA. Benchmark tarifaire, négociation automatisée et suivi des savings. Tutoriel pas-à-pas.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-analytics-conversationnel",
    title: "Agent d'Analytics Conversationnel",
    subtitle: "Interrogez vos données en langage naturel et obtenez des réponses instantanées avec graphiques",
    problem:
      "Accéder à une donnée nécessite de maîtriser SQL ou d'attendre que l'équipe data traite la demande (backlog de plusieurs semaines). Le patrimoine data de l'entreprise est sous-exploité : seuls les profils techniques y accèdent, les décideurs restent dépendants de rapports statiques.",
    value:
      "Un agent IA permet à n'importe quel collaborateur de poser des questions en langage naturel. L'agent traduit la question en requête SQL, l'exécute sur la base de données, et retourne une réponse synthétisée avec graphiques. Démocratisation complète de l'accès aux données.",
    inputs: [
      "Question en langage naturel de l'utilisateur",
      "Schéma de la base de données (tables, colonnes, relations)",
      "Dictionnaire métier (glossaire termes business → colonnes SQL)",
      "Historique des requêtes précédentes (cache et optimisation)",
    ],
    outputs: [
      "Requête SQL générée et validée",
      "Résultat structuré (tableau de données)",
      "Réponse en langage naturel synthétisant les résultats",
      "Graphique adapté au type de données (bar, line, pie chart)",
      "Suggestions de questions complémentaires pertinentes",
    ],
    risks: [
      "Requête SQL incorrecte retournant des données erronées prises pour argent comptant",
      "Accès involontaire à des données sensibles ou confidentielles (salaires, données personnelles)",
      "Requêtes lourdes impactant les performances de la base de production",
    ],
    roiIndicatif:
      "Réduction de 80% du backlog de demandes data. Temps d'accès à une donnée : de 3 jours à 30 secondes. Augmentation de 3x du nombre de décisions data-driven.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "DuckDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Question   │────▶│  Agent LLM   │────▶│  Réponse +  │
│  (langage)  │     │ (SQL + Synth)│     │  graphique  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │   Base de    │
                    │   données    │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez l'accès à votre base de données en lecture seule. L'agent nécessite le schéma de la base et un dictionnaire métier pour mapper les termes business aux colonnes SQL.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary python-dotenv fastapi plotly pandas`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://readonly_user:pass@localhost:5432/analytics
SCHEMA_PATH=./data/schema.json
GLOSSARY_PATH=./data/glossary.json`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Extraction du schéma et dictionnaire métier",
        content:
          "Chargez le schéma de la base de données et le dictionnaire métier. Le schéma permet à l'agent de connaître les tables et colonnes disponibles, le glossaire traduit les termes métier en noms techniques.",
        codeSnippets: [
          {
            language: "python",
            code: `import json
import psycopg2

def extract_schema(db_url: str) -> dict:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("""
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    schema = {}
    for table, column, dtype, nullable in cur.fetchall():
        if table not in schema:
            schema[table] = []
        schema[table].append({
            "column": column,
            "type": dtype,
            "nullable": nullable == "YES"
        })
    cur.close()
    conn.close()
    return schema

def load_glossary(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
    # Exemple: {"chiffre d'affaires": "orders.total_amount",
    #           "nombre de clients": "COUNT(DISTINCT customers.id)"}`,
            filename: "schema_loader.py",
          },
        ],
      },
      {
        title: "Agent Text-to-SQL et synthèse",
        content:
          "L'agent reçoit une question en langage naturel, génère la requête SQL correspondante, l'exécute en lecture seule, et produit une réponse synthétisée en français avec un graphique adapté.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
import psycopg2
import pandas as pd

client = anthropic.Anthropic()

def ask_data(question: str, schema: dict, glossary: dict) -> dict:
    # Étape 1 : Générer la requête SQL
    message = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""Tu es un expert SQL.
Traduis cette question en requête SQL PostgreSQL.

Question: {question}
Schéma: {json.dumps(schema, ensure_ascii=False)}
Glossaire: {json.dumps(glossary, ensure_ascii=False)}

Règles:
- SELECT uniquement (pas de INSERT, UPDATE, DELETE)
- LIMIT 1000 par défaut
- Retourne un JSON: sql, explanation, chart_type (bar/line/pie/table)"""
        }]
    )
    result = json.loads(message.content[0].text)

    # Étape 2 : Exécuter la requête
    conn = psycopg2.connect(DATABASE_URL)
    df = pd.read_sql_query(result["sql"], conn)
    conn.close()

    # Étape 3 : Synthétiser la réponse
    synthesis = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Synthétise ces résultats pour un décideur non-technique.
Question: {question}
Données: {df.head(20).to_json(orient="records", force_ascii=False)}

Retourne un JSON: answer (texte synthétique en français),
key_insights (3 points clés), follow_up_questions (3 suggestions)"""
        }]
    )
    synthesis_result = json.loads(synthesis.content[0].text)

    return {
        "sql": result["sql"],
        "chart_type": result["chart_type"],
        "data": df.to_dict(orient="records"),
        **synthesis_result
    }`,
            filename: "agent_analytics.py",
          },
        ],
      },
      {
        title: "API et interface conversationnelle",
        content:
          "Exposez l'agent via une API REST. Chaque question est traitée et retourne les données, la synthèse et le graphique. Un historique des conversations permet d'affiner les requêtes.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
import plotly.express as px
import plotly.io as pio

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    user_id: str
    conversation_id: str | None = None

@app.post("/api/analytics/ask")
async def ask(req: QuestionRequest):
    schema = extract_schema(DATABASE_URL)
    glossary = load_glossary(GLOSSARY_PATH)
    result = ask_data(req.question, schema, glossary)

    # Générer le graphique
    chart_html = None
    if result["chart_type"] != "table" and result["data"]:
        df = pd.DataFrame(result["data"])
        if result["chart_type"] == "bar":
            fig = px.bar(df, x=df.columns[0], y=df.columns[1])
        elif result["chart_type"] == "line":
            fig = px.line(df, x=df.columns[0], y=df.columns[1])
        elif result["chart_type"] == "pie":
            fig = px.pie(df, names=df.columns[0], values=df.columns[1])
        chart_html = pio.to_html(fig, full_html=False)

    return {
        "answer": result["answer"],
        "key_insights": result["key_insights"],
        "sql": result["sql"],
        "data": result["data"][:100],
        "chart_html": chart_html,
        "follow_up_questions": result["follow_up_questions"]
    }`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "L'agent utilise un utilisateur base de données en lecture seule avec accès restreint aux tables autorisées. Les colonnes sensibles (salaires, données personnelles) sont exclues du schéma exposé à l'agent. Les requêtes et résultats sont loggés sans les données brutes.",
      auditLog: "Chaque requête tracée : question posée, SQL généré, nombre de résultats, utilisateur, horodatage, temps d'exécution, coût API. Détection des tentatives d'injection SQL ou d'accès non autorisé.",
      humanInTheLoop: "Les requêtes touchant des tables sensibles (finance, RH) nécessitent une approbation du data owner. L'utilisateur voit toujours la requête SQL générée et peut la modifier avant exécution. Mode sandbox pour les nouveaux utilisateurs.",
      monitoring: "Nombre de questions/jour par utilisateur, taux de requêtes réussies vs erreurs SQL, temps de réponse moyen, tables les plus interrogées, coût API quotidien, satisfaction utilisateur (pouce haut/bas).",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (question utilisateur Slack/Teams) → HTTP Request LLM (génération SQL) → PostgreSQL (exécution requête) → HTTP Request LLM (synthèse) → HTTP Request (génération graphique) → Slack/Teams (réponse avec graphique).",
      nodes: ["Webhook (Slack/Teams)", "HTTP Request (LLM SQL)", "PostgreSQL (exécution)", "HTTP Request (LLM synthèse)", "HTTP Request (graphique)", "Slack (réponse)"],
      triggerType: "Webhook (message Slack ou Teams)",
    },
    estimatedTime: "4-6h",
    difficulty: "Facile",
    sectors: ["E-commerce", "Retail", "Services"],
    metiers: ["Direction Générale", "Data Analyst", "Direction Opérations"],
    functions: ["Operations"],
    metaTitle: "Agent IA d'Analytics Conversationnel — Guide Complet",
    metaDescription:
      "Interrogez vos données en langage naturel avec un agent IA. Questions → SQL → réponses avec graphiques. Démocratisation des données, tutoriel pas-à-pas.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-audit-interne",
    title: "Agent d'Audit Interne Automatisé",
    subtitle: "Automatisez vos audits de conformité et de processus internes grâce à l'IA",
    problem:
      "Les audits internes sont chronophages, mobilisent des ressources qualifiées pendant des semaines, et ne couvrent souvent qu'un échantillon limité des processus. Les non-conformités sont détectées tardivement, augmentant les risques réglementaires et financiers.",
    value:
      "Un agent IA analyse en continu les documents, processus et données internes pour identifier les écarts de conformité, les anomalies et les risques. Il génère automatiquement des rapports d'audit structurés avec recommandations priorisées, permettant une couverture exhaustive et une détection proactive.",
    inputs: [
      "Documents de procédures internes (PDF, Word)",
      "Référentiels réglementaires (ISO 27001, RGPD, SOX)",
      "Logs d'activité et traces d'audit existantes",
      "Données financières et opérationnelles",
      "Historique des audits précédents",
    ],
    outputs: [
      "Rapport d'audit structuré (conformités, non-conformités, observations)",
      "Score de conformité par processus (0-100%)",
      "Liste de non-conformités avec niveau de criticité",
      "Recommandations correctives priorisées",
      "Tableau de bord de suivi des plans d'action",
    ],
    risks: [
      "Faux positifs générant une surcharge de travail pour les équipes",
      "Interprétation incorrecte de textes réglementaires complexes",
      "Risque de biais dans l'évaluation de la conformité",
      "Dépendance excessive à l'automatisation pour des jugements nécessitant l'expertise humaine",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de réalisation d'un audit. Couverture des processus passant de 20% à 95%. Détection des non-conformités 3x plus rapide.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│  Agent LLM   │────▶│  Rapport    │
│  & Données  │     │  (Analyse)   │     │  d'audit    │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Vector DB   │
                    │ (Référentiels│
                    │  & Normes)   │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires et configurez votre environnement. Vous aurez besoin d'un accès API Anthropic et d'une base vectorielle pour stocker les référentiels réglementaires.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain chromadb python-dotenv pdfplumber`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
CHROMA_PERSIST_DIR=./chroma_audit_db
AUDIT_OUTPUT_DIR=./rapports_audit`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Indexation des référentiels réglementaires",
        content:
          "Indexez vos référentiels de conformité (ISO, RGPD, procédures internes) dans une base vectorielle. L'agent utilisera ces référentiels comme base de comparaison lors de l'analyse.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Charger les référentiels réglementaires
loader = DirectoryLoader("./referentiels", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# Découpage en chunks pour indexation
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# Créer la base vectorielle
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    chunks, embeddings, persist_directory="./chroma_audit_db"
)
vectorstore.persist()
print(f"{len(chunks)} chunks indexés depuis {len(docs)} pages.")`,
            filename: "index_referentiels.py",
          },
        ],
      },
      {
        title: "Agent d'analyse de conformité",
        content:
          "Construisez l'agent qui compare les documents et processus internes aux référentiels réglementaires. Il produit une analyse structurée avec score de conformité, non-conformités détectées et recommandations.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from pydantic import BaseModel, Field
from typing import List
import json

class NonConformite(BaseModel):
    description: str = Field(description="Description de la non-conformité")
    reference: str = Field(description="Article ou norme de référence")
    criticite: str = Field(description="Critique, Majeure, Mineure")
    recommandation: str = Field(description="Action corrective recommandée")

class AuditResult(BaseModel):
    processus: str = Field(description="Nom du processus audité")
    score_conformite: int = Field(ge=0, le=100)
    conformites: List[str] = Field(description="Points conformes identifiés")
    non_conformites: List[NonConformite] = Field(description="Non-conformités détectées")
    observations: List[str] = Field(description="Observations et pistes d'amélioration")

client = anthropic.Anthropic()

def audit_processus(document: str, referentiel_context: str) -> AuditResult:
    prompt = f"""Tu es un auditeur interne expert. Analyse le document suivant
en le comparant aux référentiels réglementaires fournis.

DOCUMENT À AUDITER :
{document}

RÉFÉRENTIELS APPLICABLES :
{referentiel_context}

Produis un rapport d'audit structuré au format JSON avec :
- processus : nom du processus
- score_conformite : score de 0 à 100
- conformites : liste des points conformes
- non_conformites : liste avec description, reference, criticite (Critique/Majeure/Mineure), recommandation
- observations : pistes d'amélioration"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    result_json = json.loads(response.content[0].text)
    return AuditResult(**result_json)`,
            filename: "agent_audit.py",
          },
        ],
      },
      {
        title: "Génération du rapport et API",
        content:
          "Exposez l'agent via une API REST qui accepte un document à auditer, interroge les référentiels, et retourne un rapport complet. Le rapport peut être exporté en PDF pour diffusion aux parties prenantes.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, UploadFile
from agent_audit import audit_processus, AuditResult
import pdfplumber

app = FastAPI()

@app.post("/api/audit")
async def run_audit(file: UploadFile, referentiel: str = "ISO27001"):
    # Extraction du texte du document
    with pdfplumber.open(file.file) as pdf:
        document_text = "\\n".join([p.extract_text() for p in pdf.pages])

    # Recherche des référentiels pertinents
    ref_docs = vectorstore.similarity_search(document_text, k=5)
    ref_context = "\\n".join([d.page_content for d in ref_docs])

    # Analyse de conformité
    result = audit_processus(document_text, ref_context)

    return {
        "processus": result.processus,
        "score": result.score_conformite,
        "conformites": result.conformites,
        "non_conformites": [nc.model_dump() for nc in result.non_conformites],
        "observations": result.observations
    }`,
            filename: "api_audit.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les documents d'audit peuvent contenir des données sensibles (noms d'employés, données financières). Anonymisation automatique via Presidio avant envoi au LLM. Les rapports générés sont stockés chiffrés avec accès restreint par rôle.",
      auditLog: "Chaque analyse est tracée : document audité (hash), référentiels utilisés, score de conformité, nombre de non-conformités, horodatage, utilisateur ayant lancé l'audit, coût API. Piste d'audit complète pour les régulateurs.",
      humanInTheLoop: "Les non-conformités critiques détectées par l'agent sont systématiquement validées par un auditeur humain avant publication du rapport. Le rapport final requiert une approbation du responsable conformité.",
      monitoring: "Dashboard de suivi : nombre d'audits réalisés/mois, score moyen de conformité par département, tendance des non-conformités, temps de résolution des plans d'action, alertes en cas de score < 60%.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Trigger planifié (cron mensuel) → Node Google Drive (récupération documents) → Node HTTP Request (extraction texte) → Node HTTP Request (API LLM audit) → Node IF (score < seuil) → Node Email (alerte non-conformité) → Node Google Sheets (suivi).",
      nodes: ["Schedule Trigger", "Google Drive (documents)", "HTTP Request (extraction)", "HTTP Request (LLM audit)", "IF (score < seuil)", "Email (alertes)", "Google Sheets (suivi)"],
      triggerType: "Schedule (cron mensuel ou à la demande)",
    },
    estimatedTime: "4-6h",
    difficulty: "Moyen",
    sectors: ["Finance", "Industrie", "Services", "Santé"],
    metiers: ["Audit Interne", "Conformité", "Direction Qualité"],
    functions: ["Compliance"],
    metaTitle: "Agent IA d'Audit Interne Automatisé — Guide Complet",
    metaDescription:
      "Automatisez vos audits internes avec un agent IA. Analyse de conformité, détection de non-conformités et rapports structurés. Tutoriel pas-à-pas avec stack complète.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-pricing-optimisation",
    title: "Agent d'Optimisation Tarifaire",
    subtitle: "Ajustez dynamiquement vos prix en fonction du marché, de la concurrence et de la demande",
    problem:
      "La tarification des services est souvent basée sur l'intuition ou des grilles figées. Les entreprises de services perdent du revenu en sous-évaluant certaines prestations ou perdent des contrats en surévaluant d'autres. L'analyse manuelle de la concurrence et de la demande est trop lente pour réagir aux évolutions du marché.",
    value:
      "Un agent IA analyse en temps réel les données de marché, la concurrence, l'historique des ventes et la demande pour recommander des ajustements tarifaires optimaux. Il simule l'impact de chaque changement de prix sur le chiffre d'affaires et les marges.",
    inputs: [
      "Historique des ventes et tarifs pratiqués",
      "Données concurrentielles (prix publics, positionnement)",
      "Données de demande (saisonnalité, tendances sectorielles)",
      "Coûts de revient et marges cibles",
      "Segments clients et élasticité prix observée",
    ],
    outputs: [
      "Recommandation tarifaire par service/prestation",
      "Simulation d'impact sur le CA et la marge",
      "Positionnement concurrentiel (matrice prix/valeur)",
      "Alertes de prix anormaux (trop haut ou trop bas)",
      "Rapport hebdomadaire de veille tarifaire",
    ],
    risks: [
      "Recommandations de prix trop agressives aliénant les clients existants",
      "Données concurrentielles incomplètes ou obsolètes",
      "Non-prise en compte de facteurs qualitatifs (relation client, stratégie long terme)",
      "Risque juridique sur les pratiques de prix dynamiques dans certains secteurs",
    ],
    roiIndicatif:
      "Augmentation moyenne de 12% des marges. Réduction de 40% du temps d'analyse tarifaire. Amélioration de 18% du taux de conversion des propositions commerciales.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Données    │────▶│  Agent LLM   │────▶│ Recomman-   │
│  marché &   │     │  (Analyse &  │     │ dations     │
│  ventes     │     │  Simulation) │     │ tarifaires  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Base SQL    │
                    │ (Historique  │
                    │  prix/ventes)│
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez votre base de données avec l'historique des prix et des ventes. Un minimum de 6 mois de données est recommandé pour des recommandations fiables.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain psycopg2-binary pandas numpy python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/pricing_db`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Modèle de données et collecte",
        content:
          "Structurez vos données de prix, ventes et concurrence. Le modèle doit permettre l'analyse historique et la comparaison avec le marché.",
        codeSnippets: [
          {
            language: "python",
            code: `import pandas as pd
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from typing import List, Optional
import os

engine = create_engine(os.getenv("DATABASE_URL"))

class PricingData(BaseModel):
    service: str = Field(description="Nom du service ou prestation")
    prix_actuel: float = Field(description="Prix actuel en euros")
    cout_revient: float = Field(description="Coût de revient")
    volume_ventes_mensuel: int = Field(description="Volume de ventes mensuel")
    prix_concurrent_min: Optional[float] = None
    prix_concurrent_max: Optional[float] = None
    prix_concurrent_median: Optional[float] = None

def charger_donnees_pricing() -> List[PricingData]:
    query = text("""
        SELECT s.nom as service, s.prix_actuel, s.cout_revient,
               COUNT(v.id) as volume_ventes_mensuel,
               MIN(c.prix) as prix_concurrent_min,
               MAX(c.prix) as prix_concurrent_max,
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY c.prix) as prix_concurrent_median
        FROM services s
        LEFT JOIN ventes v ON v.service_id = s.id AND v.date >= NOW() - INTERVAL '30 days'
        LEFT JOIN concurrents_prix c ON c.service_ref = s.categorie
        GROUP BY s.id, s.nom, s.prix_actuel, s.cout_revient
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return [PricingData(**row) for row in df.to_dict(orient="records")]`,
            filename: "data_pricing.py",
          },
        ],
      },
      {
        title: "Agent d'optimisation tarifaire",
        content:
          "L'agent analyse les données de pricing, compare avec la concurrence, et produit des recommandations tarifaires argumentées avec simulation d'impact financier.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json

class RecommandationPrix(BaseModel):
    service: str
    prix_actuel: float
    prix_recommande: float
    variation_pct: float = Field(description="Variation en pourcentage")
    impact_ca_estime: float = Field(description="Impact estimé sur le CA mensuel en euros")
    impact_marge_estime: float = Field(description="Impact estimé sur la marge mensuelle en euros")
    justification: str = Field(description="Argumentaire de la recommandation")
    risque: str = Field(description="Risques associés à cette modification")
    priorite: str = Field(description="Haute, Moyenne, Basse")

class AnalyseTarifaire(BaseModel):
    recommandations: List[RecommandationPrix]
    synthese: str = Field(description="Synthèse globale de l'analyse")
    impact_ca_total: float
    impact_marge_total: float

client = OpenAI()

def analyser_pricing(donnees: list, contexte_marche: str = "") -> AnalyseTarifaire:
    donnees_json = json.dumps([d.model_dump() for d in donnees], ensure_ascii=False)

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.2,
        messages=[
            {"role": "system", "content": """Tu es un expert en stratégie tarifaire pour des entreprises de services B2B.
Analyse les données de pricing fournies et produis des recommandations d'optimisation.

Règles :
- Ne jamais recommander un prix inférieur au coût de revient + 15% de marge minimale
- Tenir compte du positionnement concurrentiel
- Prioriser les services à fort volume pour maximiser l'impact
- Justifier chaque recommandation avec des données chiffrées
- Produire le résultat au format JSON conforme au schéma demandé"""},
            {"role": "user", "content": f"Données pricing :\\n{donnees_json}\\n\\nContexte marché :\\n{contexte_marche}"}
        ],
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    return AnalyseTarifaire(**result)`,
            filename: "agent_pricing.py",
          },
        ],
      },
      {
        title: "API et tableau de bord",
        content:
          "Exposez l'agent via une API REST et créez un endpoint de simulation pour tester différents scénarios tarifaires avant mise en production.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from data_pricing import charger_donnees_pricing
from agent_pricing import analyser_pricing, AnalyseTarifaire

app = FastAPI()

@app.get("/api/pricing/analyse")
async def get_analyse() -> dict:
    donnees = charger_donnees_pricing()
    analyse = analyser_pricing(donnees)
    return analyse.model_dump()

@app.post("/api/pricing/simulation")
async def simuler_prix(service: str, nouveau_prix: float) -> dict:
    donnees = charger_donnees_pricing()
    service_data = next((d for d in donnees if d.service == service), None)
    if not service_data:
        return {"error": "Service non trouvé"}

    variation = ((nouveau_prix - service_data.prix_actuel) / service_data.prix_actuel) * 100
    # Estimation simplifiée de l'élasticité prix
    elasticite = -1.2  # coefficient d'élasticité moyen services B2B
    impact_volume = service_data.volume_ventes_mensuel * (1 + (variation / 100) * elasticite)
    impact_ca = impact_volume * nouveau_prix - service_data.volume_ventes_mensuel * service_data.prix_actuel

    return {
        "service": service,
        "prix_actuel": service_data.prix_actuel,
        "prix_simule": nouveau_prix,
        "variation_pct": round(variation, 1),
        "volume_estime": round(impact_volume),
        "impact_ca_mensuel": round(impact_ca, 2)
    }`,
            filename: "api_pricing.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données tarifaires sont confidentielles. L'agent ne reçoit que des données agrégées sans information client nominative. Les prix concurrentiels proviennent de sources publiques uniquement. Les recommandations sont stockées chiffrées avec accès limité à la direction commerciale.",
      auditLog: "Chaque recommandation tarifaire est tracée : services analysés, recommandations produites, décision prise (acceptée/rejetée/modifiée), impact réel mesuré à 30 jours, utilisateur ayant validé. Historique complet pour analyse de la performance du modèle.",
      humanInTheLoop: "Toute modification de prix supérieure à 10% requiert une validation du directeur commercial. Les recommandations sont présentées comme des suggestions avec justification — la décision finale reste humaine. Comité de revue tarifaire mensuel.",
      monitoring: "Tableau de bord : écart entre prix recommandé et prix appliqué, impact réel vs impact estimé, taux d'acceptation des recommandations, évolution des marges par service, alertes si un concurrent modifie significativement ses prix.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (hebdomadaire) → Node PostgreSQL (extraction données ventes) → Node HTTP Request (scraping prix concurrents) → Node HTTP Request (API LLM analyse) → Node IF (variation > seuil) → Node Slack (notification direction commerciale) → Node Google Sheets (historique recommandations).",
      nodes: ["Schedule Trigger", "PostgreSQL (données ventes)", "HTTP Request (veille concurrentielle)", "HTTP Request (LLM analyse)", "IF (variation > seuil)", "Slack (notification)", "Google Sheets (historique)"],
      triggerType: "Schedule (cron hebdomadaire)",
    },
    estimatedTime: "4-6h",
    difficulty: "Moyen",
    sectors: ["Services", "Conseil", "SaaS", "Industrie"],
    metiers: ["Direction Commerciale", "Revenue Manager", "Direction Générale"],
    functions: ["Commercial"],
    metaTitle: "Agent IA d'Optimisation Tarifaire — Guide Complet",
    metaDescription:
      "Optimisez dynamiquement vos prix avec un agent IA. Analyse concurrentielle, simulation d'impact et recommandations tarifaires. Tutoriel pas-à-pas pour entreprises de services.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-moderation-contenu",
    title: "Agent de Modération de Contenu",
    subtitle: "Modérez automatiquement les contenus générés par les utilisateurs avec l'IA",
    problem:
      "Les plateformes e-commerce et médias reçoivent des milliers d'avis, commentaires et contenus utilisateurs quotidiennement. La modération manuelle est coûteuse, lente et inconsistante. Des contenus inappropriés, frauduleux ou diffamatoires passent entre les mailles du filet et nuisent à la réputation de la marque.",
    value:
      "Un agent IA analyse chaque contenu utilisateur en temps réel, détecte les contenus inappropriés (haine, spam, faux avis, diffamation), classifie par type de violation, et prend une action automatique (publier, masquer, escalader). La modération est instantanée, cohérente et traçable.",
    inputs: [
      "Contenu textuel (avis, commentaires, messages)",
      "Métadonnées utilisateur (historique, réputation)",
      "Règles de modération et charte communautaire",
      "Contexte du contenu (produit, article, page concernée)",
      "Historique des modérations précédentes",
    ],
    outputs: [
      "Décision de modération (publier, masquer, escalader, supprimer)",
      "Type de violation détectée (spam, haine, faux avis, hors-sujet, etc.)",
      "Score de confiance de la décision",
      "Explication de la décision pour l'utilisateur",
      "Statistiques de modération (dashboard temps réel)",
    ],
    risks: [
      "Censure excessive de contenus légitimes (faux positifs)",
      "Non-détection de contenus subtilement toxiques ou ironiques",
      "Biais culturels ou linguistiques dans la modération",
      "Non-conformité avec les obligations légales de modération (DSA européen)",
    ],
    roiIndicatif:
      "Réduction de 85% du temps de modération manuelle. Temps de réaction passant de 4h à < 10 secondes. Diminution de 60% des contenus inappropriés publiés.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Redis", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Contenu    │────▶│  Agent LLM   │────▶│  Action     │
│  utilisateur│     │ (Modération) │     │ (publier/   │
│  (UGC)      │     │              │     │  masquer)   │
└─────────────┘     └──────┬───────┘     └──────┬──────┘
                           │                     │
                    ┌──────▼───────┐     ┌──────▼──────┐
                    │  Redis       │     │  Dashboard  │
                    │  (cache +    │     │  modération │
                    │  file queue) │     │             │
                    └──────────────┘     └─────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez l'accès API. Redis est utilisé comme file d'attente pour gérer les pics de volume de contenus à modérer.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain redis python-dotenv fastapi uvicorn`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
REDIS_URL=redis://localhost:6379/0
MODERATION_THRESHOLD=0.8
AUTO_PUBLISH_THRESHOLD=0.95`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Définition des règles de modération",
        content:
          "Formalisez votre charte de modération dans un format structuré. L'agent s'appuiera sur ces règles pour prendre ses décisions de manière cohérente et explicable.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ViolationType(str, Enum):
    SPAM = "spam"
    HATE_SPEECH = "discours_haineux"
    FAKE_REVIEW = "faux_avis"
    DEFAMATION = "diffamation"
    OFF_TOPIC = "hors_sujet"
    INAPPROPRIATE = "contenu_inapproprie"
    PERSONAL_INFO = "donnees_personnelles"
    NONE = "aucune_violation"

class ModerationAction(str, Enum):
    PUBLISH = "publier"
    HIDE = "masquer"
    ESCALATE = "escalader"
    DELETE = "supprimer"

class ModerationResult(BaseModel):
    action: ModerationAction
    violation_type: ViolationType
    confidence: float = Field(ge=0, le=1)
    explanation_interne: str = Field(description="Explication pour les modérateurs")
    explanation_utilisateur: Optional[str] = Field(description="Message à afficher à l'utilisateur si contenu rejeté")

CHARTE_MODERATION = """
RÈGLES DE MODÉRATION :
1. PUBLIER : Contenu respectueux, constructif, en lien avec le sujet
2. MASQUER : Contenu potentiellement problématique nécessitant vérification
3. ESCALADER : Contenu ambigu ou cas limite nécessitant jugement humain
4. SUPPRIMER : Violation claire (spam, haine, données personnelles exposées)

CRITÈRES DE DÉTECTION :
- Spam : liens commerciaux, contenu promotionnel déguisé, texte répétitif
- Faux avis : langage excessivement positif/négatif sans détails concrets
- Haine : insultes, discrimination, menaces
- Données personnelles : numéros de téléphone, adresses, emails dans le contenu
"""`,
            filename: "moderation_rules.py",
          },
        ],
      },
      {
        title: "Agent de modération",
        content:
          "Implémentez l'agent qui analyse chaque contenu, applique les règles de modération et retourne une décision structurée avec justification. L'agent gère le contexte (produit, historique utilisateur) pour une modération plus fine.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from moderation_rules import ModerationResult, CHARTE_MODERATION, ViolationType, ModerationAction
import json

client = anthropic.Anthropic()

def moderer_contenu(
    contenu: str,
    contexte: str = "",
    historique_utilisateur: str = ""
) -> ModerationResult:
    prompt = f"""Tu es un agent de modération de contenu professionnel.
Analyse le contenu suivant selon la charte de modération.

{CHARTE_MODERATION}

CONTENU À MODÉRER :
\"{contenu}\"

CONTEXTE (produit/page) :
{contexte}

HISTORIQUE UTILISATEUR :
{historique_utilisateur}

Réponds au format JSON avec :
- action : publier, masquer, escalader, supprimer
- violation_type : spam, discours_haineux, faux_avis, diffamation, hors_sujet, contenu_inapproprie, donnees_personnelles, aucune_violation
- confidence : score de confiance entre 0 et 1
- explanation_interne : justification détaillée pour les modérateurs
- explanation_utilisateur : message pour l'utilisateur si contenu rejeté (null si publié)"""

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    result = json.loads(response.content[0].text)
    return ModerationResult(**result)

def moderer_batch(contenus: list) -> list:
    """Modération en batch pour les gros volumes"""
    return [moderer_contenu(c["text"], c.get("context", "")) for c in contenus]`,
            filename: "agent_moderation.py",
          },
        ],
      },
      {
        title: "API temps réel avec file d'attente",
        content:
          "Déployez l'API de modération avec une file d'attente Redis pour absorber les pics de trafic. Les contenus sont modérés de manière asynchrone avec notification du résultat.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
import redis
import json
from agent_moderation import moderer_contenu

app = FastAPI()
r = redis.from_url("redis://localhost:6379/0")

class ContentRequest(BaseModel):
    content_id: str
    text: str
    context: str = ""
    user_id: str = ""

@app.post("/api/moderate")
async def moderate(req: ContentRequest):
    # Modération synchrone pour les contenus individuels
    result = moderer_contenu(req.text, req.context)

    # Stockage du résultat et mise à jour des stats
    r.hset(f"moderation:{req.content_id}", mapping={
        "action": result.action.value,
        "violation": result.violation_type.value,
        "confidence": str(result.confidence),
        "explanation": result.explanation_interne
    })

    # Incrémenter les compteurs pour le dashboard
    r.incr(f"stats:moderation:{result.action.value}")
    r.incr("stats:moderation:total")

    return {
        "content_id": req.content_id,
        "action": result.action.value,
        "violation_type": result.violation_type.value,
        "confidence": result.confidence,
        "explanation_utilisateur": result.explanation_utilisateur
    }

@app.get("/api/moderate/stats")
async def get_stats():
    total = int(r.get("stats:moderation:total") or 0)
    return {
        "total": total,
        "published": int(r.get("stats:moderation:publier") or 0),
        "hidden": int(r.get("stats:moderation:masquer") or 0),
        "escalated": int(r.get("stats:moderation:escalader") or 0),
        "deleted": int(r.get("stats:moderation:supprimer") or 0)
    }`,
            filename: "api_moderation.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contenus utilisateurs sont traités sans stockage long terme dans le LLM. Les données personnelles détectées dans les contenus (emails, téléphones) sont automatiquement masquées avant traitement. Conformité DSA et RGPD assurée avec droit de recours pour les utilisateurs.",
      auditLog: "Chaque décision de modération est tracée : contenu ID, action prise, type de violation, score de confiance, modèle utilisé, horodatage. Les contestations utilisateurs sont liées à la décision initiale. Rétention des logs 2 ans pour conformité légale.",
      humanInTheLoop: "Les contenus avec un score de confiance < 0.8 sont systématiquement escaladés vers un modérateur humain. Les utilisateurs peuvent contester une décision de modération, déclenchant une revue humaine. Les modérateurs peuvent corriger les décisions de l'agent pour améliorer le modèle.",
      monitoring: "Dashboard temps réel : volume de contenus modérés/heure, répartition des décisions, taux d'escalade, temps de traitement moyen, taux de contestation, précision mesurée (vs revue humaine), alertes si le taux d'escalade dépasse 20%.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouveau contenu UGC) → Node HTTP Request (API LLM modération) → Node Switch (action: publier/masquer/escalader) → Branch publier: Node HTTP Request (API plateforme: publier) → Branch masquer: Node HTTP Request (API: masquer) + Node Email (notification utilisateur) → Branch escalader: Node Slack (alerte modérateur humain).",
      nodes: ["Webhook (nouveau contenu)", "HTTP Request (LLM modération)", "Switch (action)", "HTTP Request (publier)", "HTTP Request (masquer)", "Email (notification)", "Slack (escalade modérateur)"],
      triggerType: "Webhook (nouveau contenu UGC)",
    },
    estimatedTime: "3-5h",
    difficulty: "Facile",
    sectors: ["E-commerce", "Media", "Marketplace", "Réseaux sociaux"],
    metiers: ["Community Manager", "Trust & Safety", "Direction Digitale"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Modération de Contenu — Guide Complet",
    metaDescription:
      "Modérez automatiquement les avis, commentaires et contenus utilisateurs avec un agent IA. Détection de spam, haine et faux avis. Tutoriel pas-à-pas pour e-commerce et médias.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-enrichissement-donnees",
    title: "Agent d'Enrichissement de Données CRM",
    subtitle: "Enrichissez automatiquement vos contacts CRM avec des données externes qualifiées",
    problem:
      "Les bases CRM contiennent souvent des fiches contacts incomplètes : poste manquant, entreprise non renseignée, taille et secteur inconnus. Les commerciaux perdent du temps à rechercher manuellement ces informations, et la segmentation marketing est approximative faute de données fiables.",
    value:
      "Un agent IA enrichit automatiquement chaque fiche contact avec des données publiques : profil LinkedIn, informations entreprise (taille, CA, secteur), technologie utilisée, actualités récentes. Les fiches CRM deviennent complètes et exploitables pour la segmentation et la personnalisation.",
    inputs: [
      "Fiche contact CRM (nom, email, entreprise partielle)",
      "Domaine email professionnel",
      "Données LinkedIn publiques",
      "Bases de données entreprises (Societe.com, Pappers)",
      "Sources d'actualités sectorielles",
    ],
    outputs: [
      "Fiche contact enrichie (poste, département, ancienneté)",
      "Fiche entreprise complète (taille, CA, secteur, localisation)",
      "Stack technologique détectée (pour les SaaS/IT)",
      "Score de fiabilité des données (0-100%)",
      "Signaux d'affaires (levée de fonds, recrutement, déménagement)",
    ],
    risks: [
      "Données obsolètes ou incorrectes dégradant la qualité CRM",
      "Non-conformité RGPD sur la collecte de données personnelles",
      "Scraping de sources non autorisées (violation des CGU)",
      "Confusion entre homonymes (mauvais rattachement de profil)",
    ],
    roiIndicatif:
      "Taux de complétion des fiches CRM passant de 30% à 85%. Réduction de 90% du temps d'enrichissement manuel. Amélioration de 25% du taux de réponse aux campagnes grâce à une meilleure segmentation.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Contact    │────▶│  Agent LLM   │────▶│  CRM        │
│  CRM        │     │ (Enrichisse- │     │  (fiche     │
│  (partiel)  │     │  ment)       │     │  enrichie)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │ LinkedIn  │ │ Pappers  │ │ Google   │
       │ (profil)  │ │ (société)│ │ (actus)  │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez les accès aux APIs de données. L'API Pappers (données entreprises françaises) offre un plan gratuit suffisant pour un MVP. Pour LinkedIn, utilisez les données publiques via l'API officielle ou des services tiers conformes.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain requests psycopg2-binary python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
PAPPERS_API_KEY=...
DATABASE_URL=postgresql://user:pass@localhost:5432/crm_db
ENRICHMENT_BATCH_SIZE=50`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte de données externes",
        content:
          "Créez des connecteurs pour récupérer les données depuis les sources externes. Chaque connecteur retourne des données structurées et un score de fiabilité.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
import os
from pydantic import BaseModel, Field
from typing import Optional

class CompanyInfo(BaseModel):
    nom: str
    siren: Optional[str] = None
    forme_juridique: Optional[str] = None
    effectif: Optional[str] = None
    chiffre_affaires: Optional[float] = None
    secteur_activite: Optional[str] = None
    code_naf: Optional[str] = None
    adresse: Optional[str] = None
    date_creation: Optional[str] = None
    dirigeant: Optional[str] = None
    fiabilite: float = Field(ge=0, le=1, description="Score de fiabilité")

def enrichir_entreprise_pappers(nom_entreprise: str) -> Optional[CompanyInfo]:
    """Recherche d'informations entreprise via l'API Pappers"""
    api_key = os.getenv("PAPPERS_API_KEY")
    response = requests.get(
        "https://api.pappers.fr/v2/recherche",
        params={"q": nom_entreprise, "api_token": api_key, "par_page": 3}
    )
    if response.status_code != 200 or not response.json().get("resultats"):
        return None

    entreprise = response.json()["resultats"][0]
    return CompanyInfo(
        nom=entreprise.get("nom_entreprise", nom_entreprise),
        siren=entreprise.get("siren"),
        forme_juridique=entreprise.get("forme_juridique"),
        effectif=entreprise.get("effectif"),
        chiffre_affaires=entreprise.get("chiffre_affaires"),
        secteur_activite=entreprise.get("libelle_code_naf"),
        code_naf=entreprise.get("code_naf"),
        adresse=entreprise.get("siege", {}).get("adresse_ligne_1"),
        date_creation=entreprise.get("date_creation"),
        dirigeant=entreprise.get("representants", [{}])[0].get("nom_complet") if entreprise.get("representants") else None,
        fiabilite=0.9 if entreprise.get("siren") else 0.5
    )

def extraire_domaine_email(email: str) -> str:
    """Extrait le domaine professionnel d'une adresse email"""
    domaines_generiques = ["gmail.com", "yahoo.fr", "hotmail.com", "outlook.com", "free.fr"]
    domaine = email.split("@")[1] if "@" in email else ""
    return domaine if domaine not in domaines_generiques else ""`,
            filename: "data_connectors.py",
          },
        ],
      },
      {
        title: "Agent d'enrichissement intelligent",
        content:
          "L'agent orchestre les différentes sources de données, résout les ambiguïtés (homonymes, entreprises similaires), et produit une fiche enrichie consolidée avec score de fiabilité par champ.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from data_connectors import enrichir_entreprise_pappers, extraire_domaine_email, CompanyInfo
from pydantic import BaseModel, Field
from typing import Optional, List
import json

class ContactEnrichi(BaseModel):
    nom: str
    email: str
    poste_estime: Optional[str] = None
    departement_estime: Optional[str] = None
    entreprise: Optional[CompanyInfo] = None
    signaux_affaires: List[str] = Field(default_factory=list)
    score_completude: float = Field(ge=0, le=1)
    sources_utilisees: List[str] = Field(default_factory=list)

client = OpenAI()

def enrichir_contact(nom: str, email: str, entreprise_nom: str = "") -> ContactEnrichi:
    # Étape 1 : Détecter le domaine
    domaine = extraire_domaine_email(email)
    entreprise_recherche = entreprise_nom or domaine.split(".")[0] if domaine else ""

    # Étape 2 : Enrichissement entreprise via Pappers
    company_info = None
    sources = []
    if entreprise_recherche:
        company_info = enrichir_entreprise_pappers(entreprise_recherche)
        if company_info:
            sources.append("Pappers")

    # Étape 3 : Estimation du poste via LLM (basé sur le contexte)
    context_parts = [f"Nom: {nom}", f"Email: {email}"]
    if company_info:
        context_parts.append(f"Entreprise: {company_info.nom} ({company_info.secteur_activite})")
        context_parts.append(f"Effectif: {company_info.effectif}")

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=[
            {"role": "system", "content": """Tu es un expert en intelligence commerciale B2B.
À partir des informations partielles fournies, estime le poste probable et le département du contact.
Réponds en JSON : {"poste_estime": "...", "departement_estime": "...", "signaux_affaires": ["..."], "confidence": 0.X}
Si tu ne peux pas estimer, mets null."""},
            {"role": "user", "content": "\\n".join(context_parts)}
        ],
        response_format={"type": "json_object"}
    )
    llm_result = json.loads(response.choices[0].message.content)
    sources.append("LLM (estimation)")

    # Calcul du score de complétude
    champs_remplis = sum([
        bool(llm_result.get("poste_estime")),
        bool(llm_result.get("departement_estime")),
        bool(company_info),
        bool(company_info and company_info.chiffre_affaires),
        bool(company_info and company_info.effectif),
    ])
    score = champs_remplis / 5

    return ContactEnrichi(
        nom=nom,
        email=email,
        poste_estime=llm_result.get("poste_estime"),
        departement_estime=llm_result.get("departement_estime"),
        entreprise=company_info,
        signaux_affaires=llm_result.get("signaux_affaires", []),
        score_completude=score,
        sources_utilisees=sources
    )`,
            filename: "agent_enrichissement.py",
          },
        ],
      },
      {
        title: "API et enrichissement batch",
        content:
          "Exposez l'agent via une API REST avec support du traitement en batch pour enrichir les contacts CRM existants. L'endpoint batch traite les contacts par lots pour optimiser les appels API.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from agent_enrichissement import enrichir_contact

app = FastAPI()

class ContactInput(BaseModel):
    nom: str
    email: str
    entreprise: str = ""

class BatchRequest(BaseModel):
    contacts: List[ContactInput]

@app.post("/api/enrich")
async def enrich_single(contact: ContactInput):
    result = enrichir_contact(contact.nom, contact.email, contact.entreprise)
    return result.model_dump()

@app.post("/api/enrich/batch")
async def enrich_batch(request: BatchRequest):
    results = []
    for contact in request.contacts:
        try:
            result = enrichir_contact(contact.nom, contact.email, contact.entreprise)
            results.append({"status": "success", "data": result.model_dump()})
        except Exception as e:
            results.append({"status": "error", "contact": contact.nom, "error": str(e)})

    enriched = sum(1 for r in results if r["status"] == "success")
    return {
        "total": len(request.contacts),
        "enriched": enriched,
        "errors": len(request.contacts) - enriched,
        "results": results
    }`,
            filename: "api_enrichissement.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Enrichissement exclusivement basé sur des données publiques et des APIs conformes RGPD. Consentement requis pour le traitement des données personnelles. Les contacts peuvent exercer leur droit d'accès et de suppression. Aucune donnée personnelle n'est envoyée au LLM — seules les estimations contextuelles sont demandées.",
      auditLog: "Chaque enrichissement tracé : contact ID (hashé), sources consultées, données ajoutées, score de fiabilité, horodatage, coût API. Registre des traitements conforme à l'article 30 du RGPD. Traçabilité complète de la provenance de chaque donnée.",
      humanInTheLoop: "Les enrichissements avec un score de fiabilité < 0.6 sont signalés pour validation humaine. Les commerciaux peuvent corriger les données enrichies, améliorant la précision future. Revue trimestrielle de la qualité des enrichissements par le responsable CRM.",
      monitoring: "Dashboard : nombre de contacts enrichis/jour, taux de complétion moyen, score de fiabilité moyen, répartition par source, coût par enrichissement, alertes si le taux d'erreur API dépasse 5%, évolution de la qualité CRM dans le temps.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Trigger CRM (nouveau contact ou import batch) → Node HTTP Request (API Pappers enrichissement entreprise) → Node HTTP Request (LLM estimation poste) → Node IF (score fiabilité > seuil) → Node CRM (mise à jour fiche) → Node Slack (notification commercial si contact high-value).",
      nodes: ["CRM Trigger (nouveau contact)", "HTTP Request (Pappers)", "HTTP Request (LLM estimation)", "IF (fiabilité > seuil)", "CRM Update (fiche enrichie)", "Slack (notification)"],
      triggerType: "CRM Trigger (nouveau contact ou batch planifié)",
    },
    estimatedTime: "3-5h",
    difficulty: "Facile",
    sectors: ["SaaS", "Services", "Conseil", "B2B"],
    metiers: ["Sales Ops", "Marketing Ops", "Direction Commerciale"],
    functions: ["Commercial"],
    metaTitle: "Agent IA d'Enrichissement de Données CRM — Guide Complet",
    metaDescription:
      "Enrichissez automatiquement vos contacts CRM avec un agent IA. Données entreprise, estimation de poste et signaux d'affaires. Tutoriel pas-à-pas conforme RGPD.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-prediction-churn",
    title: "Agent de Prédiction du Churn Client",
    subtitle: "Identifiez les clients à risque de départ et déclenchez des actions de rétention proactives",
    problem:
      "Les entreprises détectent le churn trop tard, quand le client a déjà décidé de partir. Les signaux faibles (baisse d'usage, tickets non résolus, retards de paiement) sont dispersés dans différents systèmes et rarement analysés de manière consolidée. Le coût d'acquisition d'un nouveau client étant 5 à 7 fois supérieur à la rétention, chaque départ évitable représente une perte significative.",
    value:
      "Un agent IA analyse en continu les données d'usage, de support, de facturation et d'engagement pour calculer un score de risque de churn par client. Il identifie les signaux faibles, prédit les départs à 30/60/90 jours, et recommande des actions de rétention personnalisées pour chaque compte à risque.",
    inputs: [
      "Données d'usage produit (connexions, fonctionnalités utilisées, fréquence)",
      "Historique des tickets support (volume, satisfaction, temps de résolution)",
      "Données de facturation (retards, litiges, downgrades)",
      "Données d'engagement (ouverture emails, participation événements, NPS)",
      "Historique des churns passés (pour entraînement du modèle)",
    ],
    outputs: [
      "Score de risque de churn par client (0-100)",
      "Probabilité de churn à 30, 60, 90 jours",
      "Top 3 des facteurs de risque par client",
      "Recommandation d'action de rétention personnalisée",
      "Tableau de bord des comptes à risque avec priorisation",
    ],
    risks: [
      "Faux positifs générant des actions de rétention inutiles et coûteuses",
      "Faux négatifs manquant des départs évitables",
      "Effet Hawthorne : l'attention portée au client peut modifier son comportement",
      "Biais du modèle favorisant certains segments clients au détriment d'autres",
    ],
    roiIndicatif:
      "Réduction du taux de churn de 15% à 20%. Augmentation de la LTV moyenne de 25%. ROI de 300% sur les actions de rétention ciblées. Détection des comptes à risque 45 jours plus tôt en moyenne.",
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
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Données    │────▶│  Agent LLM   │────▶│  Actions    │
│  client     │     │  (Scoring &  │     │  rétention  │
│  (multi-src)│     │  Prédiction) │     │  (CRM/Slack)│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │  Usage    │ │ Support  │ │ Billing  │
       │  Analytics│ │ Tickets  │ │  Data    │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez l'accès à vos différentes sources de données client. Un minimum de 12 mois d'historique est recommandé pour un modèle de prédiction fiable.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary pandas scikit-learn python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/analytics_db
CHURN_RISK_THRESHOLD=70
ALERT_CHANNEL_WEBHOOK=https://hooks.slack.com/services/...`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte et agrégation des signaux",
        content:
          "Consolidez les données de plusieurs sources (usage, support, facturation) en un profil client unifié. Chaque signal est normalisé et horodaté pour permettre l'analyse de tendance.",
        codeSnippets: [
          {
            language: "python",
            code: `import pandas as pd
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import os

engine = create_engine(os.getenv("DATABASE_URL"))

class SignalClient(BaseModel):
    client_id: str
    nom_client: str
    mrr: float = Field(description="Revenu mensuel récurrent en euros")
    anciennete_mois: int
    # Signaux d'usage
    connexions_30j: int = 0
    variation_usage_pct: float = Field(default=0, description="Variation d'usage vs mois précédent")
    fonctionnalites_actives: int = 0
    dernier_login_jours: int = 0
    # Signaux support
    tickets_ouverts_30j: int = 0
    tickets_non_resolus: int = 0
    nps_dernier: Optional[int] = None
    satisfaction_moyenne: Optional[float] = None
    # Signaux facturation
    retards_paiement_90j: int = 0
    litiges_ouverts: int = 0
    downgrade_recent: bool = False
    # Signaux engagement
    emails_ouverts_pct_30j: float = 0
    participation_events: int = 0

def collecter_signaux_client(client_id: str) -> SignalClient:
    query = text("""
        WITH usage_data AS (
            SELECT client_id,
                   COUNT(*) as connexions_30j,
                   COUNT(DISTINCT feature_name) as fonctionnalites_actives,
                   EXTRACT(DAY FROM NOW() - MAX(login_date)) as dernier_login_jours
            FROM user_events
            WHERE client_id = :cid AND event_date >= NOW() - INTERVAL '30 days'
            GROUP BY client_id
        ),
        support_data AS (
            SELECT client_id,
                   COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days') as tickets_30j,
                   COUNT(*) FILTER (WHERE status = 'open') as tickets_non_resolus,
                   AVG(satisfaction_score) as satisfaction_moyenne
            FROM support_tickets WHERE client_id = :cid
            GROUP BY client_id
        ),
        billing_data AS (
            SELECT client_id, mrr,
                   COUNT(*) FILTER (WHERE payment_status = 'late' AND due_date >= NOW() - INTERVAL '90 days') as retards_90j,
                   COUNT(*) FILTER (WHERE type = 'dispute' AND status = 'open') as litiges
            FROM billing WHERE client_id = :cid
            GROUP BY client_id, mrr
        )
        SELECT c.id as client_id, c.nom as nom_client, c.created_at,
               COALESCE(u.connexions_30j, 0) as connexions_30j,
               COALESCE(u.fonctionnalites_actives, 0) as fonctionnalites_actives,
               COALESCE(u.dernier_login_jours, 999) as dernier_login_jours,
               COALESCE(s.tickets_30j, 0) as tickets_ouverts_30j,
               COALESCE(s.tickets_non_resolus, 0) as tickets_non_resolus,
               s.satisfaction_moyenne,
               COALESCE(b.mrr, 0) as mrr,
               COALESCE(b.retards_90j, 0) as retards_paiement_90j,
               COALESCE(b.litiges, 0) as litiges_ouverts
        FROM clients c
        LEFT JOIN usage_data u ON u.client_id = c.id
        LEFT JOIN support_data s ON s.client_id = c.id
        LEFT JOIN billing_data b ON b.client_id = c.id
        WHERE c.id = :cid
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"cid": client_id}).fetchone()

    anciennete = (datetime.now() - row.created_at).days // 30
    return SignalClient(
        client_id=row.client_id,
        nom_client=row.nom_client,
        mrr=row.mrr,
        anciennete_mois=anciennete,
        connexions_30j=row.connexions_30j,
        fonctionnalites_actives=row.fonctionnalites_actives,
        dernier_login_jours=int(row.dernier_login_jours),
        tickets_ouverts_30j=row.tickets_ouverts_30j,
        tickets_non_resolus=row.tickets_non_resolus,
        satisfaction_moyenne=row.satisfaction_moyenne,
        retards_paiement_90j=row.retards_paiement_90j,
        litiges_ouverts=row.litiges_ouverts
    )`,
            filename: "data_signals.py",
          },
        ],
      },
      {
        title: "Agent de scoring et recommandation",
        content:
          "L'agent analyse les signaux consolidés, calcule un score de risque, identifie les facteurs principaux et recommande des actions de rétention spécifiques à chaque situation client.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from data_signals import SignalClient
from pydantic import BaseModel, Field
from typing import List
import json

class ChurnPrediction(BaseModel):
    client_id: str
    score_risque: int = Field(ge=0, le=100, description="Score de risque de churn")
    probabilite_30j: float = Field(ge=0, le=1)
    probabilite_60j: float = Field(ge=0, le=1)
    probabilite_90j: float = Field(ge=0, le=1)
    facteurs_risque: List[str] = Field(description="Top facteurs de risque identifiés")
    action_recommandee: str = Field(description="Action de rétention recommandée")
    urgence: str = Field(description="Critique, Haute, Moyenne, Basse")
    argumentaire: str = Field(description="Argumentaire pour le CSM")

client = anthropic.Anthropic()

def predire_churn(signaux: SignalClient) -> ChurnPrediction:
    signaux_json = signaux.model_dump_json()

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": f"""Tu es un expert en Customer Success et rétention client B2B SaaS.
Analyse les signaux suivants pour ce client et produis une prédiction de churn.

SIGNAUX CLIENT :
{signaux_json}

RÈGLES D'ANALYSE :
- Connexions en baisse forte (> -30%) = signal critique
- Tickets non résolus > 3 = signal fort
- Retard de paiement = signal fort
- Dernier login > 14 jours = signal d'alerte
- NPS < 7 = insatisfaction
- Downgrade récent = intention de départ probable

Produis un JSON avec :
- score_risque (0-100)
- probabilite_30j, probabilite_60j, probabilite_90j (entre 0 et 1)
- facteurs_risque (top 3 facteurs)
- action_recommandee (action précise et personnalisée)
- urgence (Critique si score > 80, Haute si > 60, Moyenne si > 40, Basse sinon)
- argumentaire (script pour le CSM : ce qu'il doit dire/faire)"""}
        ]
    )
    result = json.loads(response.content[0].text)
    result["client_id"] = signaux.client_id
    return ChurnPrediction(**result)

def analyser_portefeuille(clients_ids: list) -> list:
    """Analyse un portefeuille complet de clients"""
    from data_signals import collecter_signaux_client
    predictions = []
    for cid in clients_ids:
        signaux = collecter_signaux_client(cid)
        prediction = predire_churn(signaux)
        predictions.append(prediction)
    # Tri par score de risque décroissant
    predictions.sort(key=lambda p: p.score_risque, reverse=True)
    return predictions`,
            filename: "agent_churn.py",
          },
        ],
      },
      {
        title: "API et alertes automatiques",
        content:
          "Déployez l'API de prédiction avec des alertes Slack automatiques pour les comptes à risque. Le système s'exécute quotidiennement et notifie les CSM des comptes nécessitant une intervention urgente.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from agent_churn import predire_churn, analyser_portefeuille
from data_signals import collecter_signaux_client
import requests
import os

app = FastAPI()
SLACK_WEBHOOK = os.getenv("ALERT_CHANNEL_WEBHOOK")
RISK_THRESHOLD = int(os.getenv("CHURN_RISK_THRESHOLD", 70))

@app.get("/api/churn/client/{client_id}")
async def get_churn_risk(client_id: str):
    signaux = collecter_signaux_client(client_id)
    prediction = predire_churn(signaux)
    return prediction.model_dump()

@app.post("/api/churn/scan")
async def scan_portfolio():
    """Scan complet du portefeuille client"""
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        ids = [r[0] for r in conn.execute(text("SELECT id FROM clients WHERE status = 'active'")).fetchall()]

    predictions = analyser_portefeuille(ids)
    high_risk = [p for p in predictions if p.score_risque >= RISK_THRESHOLD]

    # Alertes Slack pour les comptes critiques
    if high_risk and SLACK_WEBHOOK:
        blocks = []
        for p in high_risk[:10]:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{p.client_id}* — Score: {p.score_risque}/100 ({p.urgence})\\n"
                            f"Facteurs: {', '.join(p.facteurs_risque)}\\n"
                            f"Action: {p.action_recommandee}"
                }
            })
        requests.post(SLACK_WEBHOOK, json={
            "text": f"🚨 {len(high_risk)} comptes à risque de churn détectés",
            "blocks": blocks
        })

    return {
        "total_clients": len(ids),
        "high_risk": len(high_risk),
        "predictions": [p.model_dump() for p in predictions[:20]]
    }`,
            filename: "api_churn.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données client sont traitées de manière agrégée — seuls les signaux comportementaux anonymisés sont envoyés au LLM, jamais les données personnelles (nom, email, téléphone). Le scoring est stocké en base interne avec accès restreint aux équipes Customer Success et Direction.",
      auditLog: "Chaque prédiction est tracée : client ID (hashé), score de risque, facteurs identifiés, action recommandée, action effectivement prise par le CSM, résultat à 90 jours (churn effectif ou rétention). Permet le calcul de la précision du modèle et son amélioration continue.",
      humanInTheLoop: "Le score de churn est un outil d'aide à la décision — le CSM décide de l'action finale. Les comptes stratégiques (MRR > seuil) nécessitent une validation du Head of CS avant action. Revue hebdomadaire des comptes à risque en comité CS.",
      monitoring: "Dashboard Customer Success : distribution des scores de risque, évolution du taux de churn réel vs prédit, précision du modèle (matrice de confusion), MRR at risk, taux de rétention des comptes alertés, coût API quotidien, alertes si le taux de churn réel dépasse la prédiction de plus de 5 points.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (quotidien 8h) → Node PostgreSQL (extraction signaux clients actifs) → Node HTTP Request (API LLM scoring churn) → Node IF (score > seuil) → Branch critique: Node Slack (alerte CSM) + Node CRM (créer tâche rétention) → Branch normal: Node Google Sheets (suivi).",
      nodes: ["Schedule Trigger (quotidien)", "PostgreSQL (signaux clients)", "HTTP Request (LLM scoring)", "IF (score > seuil)", "Slack (alerte CSM)", "CRM (tâche rétention)", "Google Sheets (suivi)"],
      triggerType: "Schedule (cron quotidien à 8h00)",
    },
    estimatedTime: "5-8h",
    difficulty: "Moyen",
    sectors: ["SaaS", "Telecom", "Assurance", "Services B2B"],
    metiers: ["Customer Success", "Direction Commerciale", "Direction Générale"],
    functions: ["Commercial"],
    metaTitle: "Agent IA de Prédiction du Churn Client — Guide Complet",
    metaDescription:
      "Prédisez le churn client avec un agent IA. Scoring de risque, détection de signaux faibles et actions de rétention personnalisées. Tutoriel pas-à-pas pour SaaS et services B2B.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
];
