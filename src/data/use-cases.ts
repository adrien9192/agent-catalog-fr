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
      "Implémentez un agent IA de triage automatique pour votre support client. Classification, routage intelligent, stack technique et tutoriel pas-à-pas avec ROI détaillé.",
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
      "Automatisez la qualification de vos leads avec un agent IA. Scoring BANT/MEDDIC, enrichissement firmographique et intégration CRM. Tutoriel complet avec code.",
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
      "Automatisez le tri des CVs avec un agent IA. Extraction de compétences, scoring objectif, conformité RGPD et intégration ATS. Guide complet avec tutoriel.",
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
      "Automatisez votre veille concurrentielle avec un agent IA. Monitoring continu des prix, produits et actualités. Alertes et synthèses hebdomadaires automatiques.",
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
      "Automatisez vos rapports financiers avec un agent IA. Collecte de données ERP, détection d'anomalies et commentaires automatiques. Stack et tutoriel inclus.",
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
      "Automatisez le diagnostic et la résolution des incidents IT avec un agent IA. Corrélation de logs, runbooks automatiques et escalade intelligente. Guide DevOps.",
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
      "Générez du contenu marketing de qualité avec un agent IA. Articles SEO, posts réseaux sociaux et newsletters automatisés. Tutoriel complet avec stack technique.",
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
      "Créez un assistant IA d'onboarding pour vos nouveaux collaborateurs. Chatbot RAG sur Slack, parcours personnalisé et checklist automatique. Guide RH complet.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-detection-fraude",
    title: "Agent de Détection de Fraude Multi-secteur",
    subtitle: "Détectez les transactions frauduleuses en temps réel sur vos plateformes e-commerce, SaaS et paiements en ligne",
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
    metaTitle: "Agent IA de Détection de Fraude Multi-secteur — Guide Expert",
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
      "Automatisez vos processus achats avec un agent IA. Comparaison de devis, validation workflow, suivi fournisseurs et optimisation des coûts. Guide Supply Chain.",
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
  {
    slug: "agent-analyse-sentiments",
    title: "Agent d'Analyse de Sentiments Client Multi-canal",
    subtitle: "Collectez et analysez les feedbacks clients sur tous vos canaux (email, chat, avis, réseaux sociaux) pour optimiser l'expérience client",
    problem:
      "Les équipes marketing et CX sont incapables de traiter manuellement le volume croissant de feedbacks clients provenant de multiples canaux (emails, chat en direct, réseaux sociaux, avis en ligne). Les sentiments négatifs passent inaperçus pendant des jours, les tendances émergentes sont détectées trop tard, et les rapports manuels sont biaisés par l'échantillonnage humain. Résultat : des crises réputationnelles évitables et des opportunités d'amélioration manquées.",
    value:
      "Un agent IA collecte et analyse en temps réel les feedbacks de tous les canaux, détecte le sentiment (positif, négatif, neutre, mixte), identifie les thèmes récurrents et les signaux faibles, et génère des alertes immédiates pour les situations critiques. Les équipes disposent d'un tableau de bord unifié avec des tendances et recommandations actionnables.",
    inputs: [
      "Emails et tickets support client",
      "Conversations de chat en direct (Intercom, Zendesk Chat)",
      "Publications et commentaires réseaux sociaux (Twitter/X, LinkedIn, Instagram)",
      "Avis en ligne (Google Reviews, Trustpilot, G2)",
      "Enquêtes NPS et CSAT",
    ],
    outputs: [
      "Score de sentiment par message (-1 à +1) avec label (positif/négatif/neutre/mixte)",
      "Thèmes et sujets récurrents par canal et par période",
      "Alertes temps réel pour les pics de sentiment négatif",
      "Rapport hebdomadaire de tendances avec recommandations",
      "Tableau de bord unifié multi-canal avec évolution temporelle",
    ],
    risks: [
      "Mauvaise interprétation du sarcasme, de l'ironie ou de l'humour culturel",
      "Biais linguistique sur les expressions régionales ou le langage informel",
      "Surcharge d'alertes faux-positifs provoquant une fatigue d'alerte chez les équipes",
      "Non-conformité RGPD si des données personnelles sont transmises au LLM",
    ],
    roiIndicatif:
      "Détection des crises réputationnelles 48h plus tôt en moyenne. Réduction de 40% du temps d'analyse manuelle des feedbacks. Amélioration de 15% du score NPS grâce aux actions correctives rapides.",
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
│  Sources    │────▶│  Agent LLM   │────▶│  Dashboard  │
│  multi-canal│     │  (Analyse    │     │  & Alertes  │
│  (API/Webhook)    │  Sentiment)  │     │  (Grafana)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │  Email /  │ │  Social  │ │  Chat /  │
       │  Tickets  │ │  Media   │ │  Avis    │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires et configurez les accès aux différentes APIs de collecte. Vous aurez besoin d'un compte Anthropic et des tokens d'accès aux réseaux sociaux que vous souhaitez monitorer.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary tweepy python-dotenv fastapi uvicorn`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/sentiments_db
TWITTER_BEARER_TOKEN=...
SLACK_WEBHOOK_ALERTS=https://hooks.slack.com/services/...
INTERCOM_ACCESS_TOKEN=...`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte multi-canal",
        content:
          "Mettez en place les connecteurs pour collecter les feedbacks depuis chaque canal. Chaque message est normalisé dans un format unifié avant analyse. Les connecteurs fonctionnent en mode webhook (temps réel) ou polling (batch périodique).",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
import tweepy
import os

class FeedbackMessage(BaseModel):
    id: str
    source: Literal["email", "chat", "twitter", "linkedin", "review", "nps"]
    contenu: str
    auteur: Optional[str] = None
    date: datetime
    metadata: dict = Field(default_factory=dict)

class TwitterCollector:
    def __init__(self):
        self.client = tweepy.Client(
            bearer_token=os.getenv("TWITTER_BEARER_TOKEN")
        )

    def collecter_mentions(self, query: str, max_results: int = 100) -> list[FeedbackMessage]:
        tweets = self.client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "lang"]
        )
        messages = []
        for tweet in tweets.data or []:
            if tweet.lang == "fr":
                messages.append(FeedbackMessage(
                    id=str(tweet.id),
                    source="twitter",
                    contenu=tweet.text,
                    auteur=str(tweet.author_id),
                    date=tweet.created_at,
                    metadata={"lang": tweet.lang}
                ))
        return messages

class EmailCollector:
    def collecter_depuis_webhook(self, payload: dict) -> FeedbackMessage:
        return FeedbackMessage(
            id=payload["message_id"],
            source="email",
            contenu=payload["body_text"],
            auteur=payload.get("from_email"),
            date=datetime.fromisoformat(payload["received_at"]),
            metadata={"subject": payload.get("subject", "")}
        )`,
            filename: "collectors.py",
          },
        ],
      },
      {
        title: "Agent d'analyse de sentiment",
        content:
          "L'agent analyse chaque message, détecte le sentiment, identifie les thèmes abordés et extrait les insights actionnables. Il utilise un prompt structuré pour produire une analyse cohérente et comparable entre les canaux.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from collectors import FeedbackMessage
from pydantic import BaseModel, Field
from typing import List
import json

class AnalyseSentiment(BaseModel):
    message_id: str
    score_sentiment: float = Field(ge=-1, le=1, description="Score de -1 (très négatif) à +1 (très positif)")
    label: str = Field(description="positif, négatif, neutre ou mixte")
    themes: List[str] = Field(description="Thèmes identifiés dans le message")
    emotions: List[str] = Field(description="Émotions détectées (frustration, satisfaction, colère, etc.)")
    urgence: bool = Field(description="True si action immédiate requise")
    resume: str = Field(description="Résumé en une phrase du feedback")
    action_suggeree: str = Field(description="Action recommandée pour l'équipe CX")

client = anthropic.Anthropic()

def analyser_sentiment(message: FeedbackMessage) -> AnalyseSentiment:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": f"""Tu es un expert en analyse de sentiment client.
Analyse le message suivant provenant du canal "{message.source}".

MESSAGE :
{message.contenu}

MÉTADONNÉES :
- Source : {message.source}
- Date : {message.date.isoformat()}
- Auteur : {message.auteur or "Anonyme"}

Produis un JSON avec :
- score_sentiment : float de -1 (très négatif) à +1 (très positif)
- label : "positif", "négatif", "neutre" ou "mixte"
- themes : liste des thèmes abordés (prix, qualité, support, livraison, etc.)
- emotions : liste des émotions détectées
- urgence : true si le message nécessite une action immédiate (menace de départ, plainte grave, etc.)
- resume : résumé en une phrase
- action_suggeree : action concrète recommandée

Attention au sarcasme et à l'ironie. Analyse le contexte global, pas seulement les mots-clés."""}
        ]
    )
    result = json.loads(response.content[0].text)
    result["message_id"] = message.id
    return AnalyseSentiment(**result)

def analyser_batch(messages: List[FeedbackMessage]) -> List[AnalyseSentiment]:
    analyses = []
    for msg in messages:
        analyse = analyser_sentiment(msg)
        analyses.append(analyse)
    return analyses`,
            filename: "agent_sentiment.py",
          },
        ],
      },
      {
        title: "API et alertes temps réel",
        content:
          "Exposez l'agent via une API REST et configurez les alertes automatiques pour les sentiments négatifs urgents. Le système génère également des rapports de tendances agrégés.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from agent_sentiment import analyser_sentiment, analyser_batch, AnalyseSentiment
from collectors import FeedbackMessage, TwitterCollector
from datetime import datetime, timedelta
import requests
import os

app = FastAPI()
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_ALERTS")

@app.post("/api/sentiment/analyse")
async def analyse_single(message: FeedbackMessage):
    result = analyser_sentiment(message)
    # Alerte si urgent
    if result.urgence and SLACK_WEBHOOK:
        requests.post(SLACK_WEBHOOK, json={
            "text": f"⚠️ Sentiment négatif urgent détecté\\n"
                    f"Source: {message.source} | Score: {result.score_sentiment}\\n"
                    f"Résumé: {result.resume}\\n"
                    f"Action: {result.action_suggeree}"
        })
    return result.model_dump()

@app.get("/api/sentiment/tendances")
async def get_tendances(jours: int = 7):
    """Agrège les analyses des N derniers jours par thème et canal"""
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT source, label, themes, score_sentiment, date_analyse
            FROM analyses_sentiment
            WHERE date_analyse >= NOW() - INTERVAL ':jours days'
            ORDER BY date_analyse DESC
        """), {"jours": jours}).fetchall()

    tendances = {
        "periode": f"Derniers {jours} jours",
        "total_messages": len(rows),
        "score_moyen": sum(r.score_sentiment for r in rows) / max(len(rows), 1),
        "repartition": {},
        "themes_frequents": {}
    }
    for row in rows:
        tendances["repartition"][row.label] = tendances["repartition"].get(row.label, 0) + 1
    return tendances`,
            filename: "api_sentiment.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les messages sont anonymisés avant envoi au LLM — noms, emails, numéros de téléphone et identifiants client sont masqués via des expressions régulières et Microsoft Presidio. Seul le contenu textuel nettoyé est transmis à l'API. Les données brutes restent en base interne avec accès restreint.",
      auditLog: "Chaque analyse est tracée : message ID (hashé), source, score de sentiment, thèmes détectés, actions recommandées, horodatage. Les modifications manuelles du label par un analyste humain sont enregistrées pour améliorer le modèle. Rétention des logs : 24 mois.",
      humanInTheLoop: "Les messages classés comme urgents déclenchent une notification immédiate au responsable CX qui valide l'action recommandée avant exécution. Les analyses avec un score de confiance faible (sentiment mixte ou ambigu) sont renvoyées pour revue humaine. Revue hebdomadaire des faux positifs et faux négatifs.",
      monitoring: "Dashboard Grafana : volume de messages analysés par canal, distribution des sentiments (temps réel), évolution du score moyen par semaine, top thèmes négatifs, temps de réponse de l'API, coût API par jour, alertes si le taux de sentiment négatif dépasse 30% sur une fenêtre de 4h.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (toutes les 15 min) → Node HTTP Request (collecte Twitter API) + Node Webhook (emails entrants) + Node HTTP Request (Intercom conversations) → Node Merge (unification) → Node HTTP Request (API LLM analyse sentiment) → Node IF (urgence = true) → Branch urgente: Node Slack (alerte immédiate) + Node Airtable (log) → Branch normale: Node PostgreSQL (stockage).",
      nodes: ["Schedule Trigger (15 min)", "HTTP Request (Twitter API)", "Webhook (emails)", "HTTP Request (Intercom)", "Merge (unification)", "HTTP Request (LLM analyse)", "IF (urgence)", "Slack (alerte)", "PostgreSQL (stockage)", "Airtable (log)"],
      triggerType: "Schedule (cron toutes les 15 minutes) + Webhook (emails entrants)",
    },
    estimatedTime: "4-6h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "Services", "SaaS", "Hôtellerie", "Retail"],
    metiers: ["Marketing", "Customer Experience", "Communication"],
    functions: ["Marketing"],
    metaTitle: "Agent IA d'Analyse de Sentiments Multi-Canal — Guide Complet",
    metaDescription:
      "Déployez un agent IA d'analyse de sentiments multi-canal. Détectez en temps réel le sentiment client sur emails, chat, réseaux sociaux. Tutoriel pas-à-pas avec stack complète.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-generation-rapports-esg",
    title: "Agent de Génération de Rapports ESG",
    subtitle: "Automatisez la collecte de données et la génération de rapports ESG/RSE conformes aux réglementations",
    problem:
      "La génération de rapports ESG (Environnement, Social, Gouvernance) est un processus annuel fastidieux qui mobilise des dizaines de collaborateurs pendant des mois. Les données sont dispersées dans de multiples départements (RH, opérations, achats, finance), souvent dans des formats hétérogènes (tableurs, ERP, emails). Les réglementations évoluent rapidement (CSRD, taxonomie européenne) et les erreurs de reporting exposent l'entreprise à des sanctions réglementaires et un risque réputationnel majeur.",
    value:
      "Un agent IA orchestre la collecte automatique des données ESG depuis les différents systèmes d'information, vérifie leur cohérence, calcule les indicateurs clés (émissions carbone, diversité, gouvernance), et génère des rapports conformes aux standards GRI, CSRD et taxonomie européenne. Le temps de production du rapport annuel passe de 3 mois à 2 semaines.",
    inputs: [
      "Données RH (effectifs, diversité, formation, accidents du travail)",
      "Données environnementales (consommation énergie, eau, déchets, émissions)",
      "Données fournisseurs (audits sociaux, certifications, provenance)",
      "Données financières (investissements verts, part CA durable)",
      "Référentiels réglementaires (GRI, CSRD, taxonomie UE)",
    ],
    outputs: [
      "Rapport ESG complet conforme CSRD avec indicateurs GRI",
      "Tableau de bord des KPI ESG avec évolution N-1/N-2",
      "Matrice de double matérialité auto-générée",
      "Plan d'action avec recommandations d'amélioration priorisées",
      "Alertes de non-conformité réglementaire",
    ],
    risks: [
      "Données source incorrectes ou incomplètes conduisant à un reporting erroné",
      "Hallucinations du LLM sur des chiffres réglementaires ou des seuils",
      "Évolution réglementaire non prise en compte entre deux mises à jour du prompt",
      "Greenwashing involontaire si l'agent surinterprète positivement les données",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de production du rapport ESG annuel. Économie de 50 000 à 150 000 EUR en coûts de conseil externe. Diminution de 90% des erreurs de consolidation de données inter-départements.",
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
│  Sources    │────▶│  Agent LLM   │────▶│  Rapport    │
│  données    │     │  (Collecte & │     │  ESG/CSRD   │
│  (multi-dept)│    │  Génération) │     │  (PDF/Web)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │    RH     │ │  Ops /   │ │ Finance  │
       │  (SIRH)   │ │  Env.    │ │  (ERP)   │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez les accès aux systèmes sources. Préparez le référentiel des indicateurs GRI et CSRD qui servira de grille de collecte pour l'agent.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain psycopg2-binary pandas openpyxl python-dotenv fastapi jinja2 weasyprint`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/esg_db
SIRH_API_URL=https://sirh.entreprise.com/api
ERP_API_URL=https://erp.entreprise.com/api
YEAR_REPORTING=2024`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte et normalisation des données",
        content:
          "L'agent collecte les données depuis les différents systèmes sources (SIRH, ERP, tableurs) et les normalise dans un format unifié. Chaque indicateur est mappé au référentiel GRI/CSRD correspondant.",
        codeSnippets: [
          {
            language: "python",
            code: `import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import requests
import os

class IndicateurESG(BaseModel):
    code_gri: str = Field(description="Code indicateur GRI (ex: GRI 305-1)")
    categorie: str = Field(description="E, S ou G")
    nom: str
    valeur: float
    unite: str
    annee: int
    source: str
    methode_calcul: str
    valeur_n_moins_1: Optional[float] = None
    evolution_pct: Optional[float] = None

class DonneesESG(BaseModel):
    entreprise: str
    annee_reporting: int
    indicateurs: List[IndicateurESG]
    completude_pct: float = Field(description="Pourcentage d'indicateurs renseignés")

def collecter_donnees_rh() -> List[IndicateurESG]:
    """Collecte les indicateurs sociaux depuis le SIRH"""
    sirh_url = os.getenv("SIRH_API_URL")
    annee = int(os.getenv("YEAR_REPORTING"))

    effectifs = requests.get(f"{sirh_url}/effectifs?year={annee}").json()
    formation = requests.get(f"{sirh_url}/formation?year={annee}").json()
    securite = requests.get(f"{sirh_url}/securite?year={annee}").json()

    indicateurs = [
        IndicateurESG(
            code_gri="GRI 2-7",
            categorie="S",
            nom="Effectif total",
            valeur=effectifs["total"],
            unite="ETP",
            annee=annee,
            source="SIRH",
            methode_calcul="Comptage ETP au 31/12"
        ),
        IndicateurESG(
            code_gri="GRI 405-1",
            categorie="S",
            nom="Part de femmes dans le management",
            valeur=effectifs["pct_femmes_management"],
            unite="%",
            annee=annee,
            source="SIRH",
            methode_calcul="Femmes managers / Total managers x 100"
        ),
        IndicateurESG(
            code_gri="GRI 404-1",
            categorie="S",
            nom="Heures de formation par salarié",
            valeur=formation["heures_par_salarie"],
            unite="heures/ETP",
            annee=annee,
            source="SIRH",
            methode_calcul="Total heures formation / Effectif moyen"
        ),
        IndicateurESG(
            code_gri="GRI 403-9",
            categorie="S",
            nom="Taux de fréquence des accidents",
            valeur=securite["taux_frequence"],
            unite="pour 1M heures",
            annee=annee,
            source="SIRH",
            methode_calcul="(Nb accidents AT / Heures travaillées) x 1 000 000"
        ),
    ]
    return indicateurs

def collecter_donnees_environnement() -> List[IndicateurESG]:
    """Collecte les indicateurs environnementaux"""
    annee = int(os.getenv("YEAR_REPORTING"))
    # Lecture depuis un fichier Excel consolidé par les opérations
    df = pd.read_excel("data/donnees_environnement.xlsx", sheet_name=str(annee))

    indicateurs = [
        IndicateurESG(
            code_gri="GRI 305-1",
            categorie="E",
            nom="Émissions GES Scope 1",
            valeur=df.loc[df["indicateur"] == "scope1", "valeur"].values[0],
            unite="tCO2eq",
            annee=annee,
            source="Bilan Carbone",
            methode_calcul="Méthode Bilan Carbone ADEME - émissions directes"
        ),
        IndicateurESG(
            code_gri="GRI 305-2",
            categorie="E",
            nom="Émissions GES Scope 2",
            valeur=df.loc[df["indicateur"] == "scope2", "valeur"].values[0],
            unite="tCO2eq",
            annee=annee,
            source="Bilan Carbone",
            methode_calcul="Méthode location-based - électricité et chaleur"
        ),
        IndicateurESG(
            code_gri="GRI 302-1",
            categorie="E",
            nom="Consommation énergétique totale",
            valeur=df.loc[df["indicateur"] == "energie_totale", "valeur"].values[0],
            unite="MWh",
            annee=annee,
            source="Factures énergie",
            methode_calcul="Somme des consommations électricité + gaz + carburants"
        ),
    ]
    return indicateurs`,
            filename: "collecte_esg.py",
          },
        ],
      },
      {
        title: "Agent de génération de rapport",
        content:
          "L'agent LLM prend en entrée les indicateurs consolidés et génère le contenu rédactionnel du rapport ESG. Il structure le rapport selon le standard CSRD, rédige les commentaires d'analyse, et identifie les points d'amélioration.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from collecte_esg import DonneesESG, IndicateurESG
from pydantic import BaseModel, Field
from typing import List
import json

class SectionRapport(BaseModel):
    titre: str
    contenu_markdown: str
    indicateurs_cles: List[dict]
    points_forts: List[str]
    axes_amelioration: List[str]

class RapportESG(BaseModel):
    titre: str
    annee: int
    sections: List[SectionRapport]
    synthese_executive: str
    score_conformite_csrd: float = Field(ge=0, le=100)
    recommandations_prioritaires: List[str]

client = OpenAI()

def generer_rapport(donnees: DonneesESG) -> RapportESG:
    indicateurs_json = json.dumps(
        [ind.model_dump() for ind in donnees.indicateurs],
        ensure_ascii=False, indent=2
    )

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """Tu es un expert en reporting ESG/RSE.
Tu génères des rapports conformes à la directive CSRD et aux standards GRI.
Tu ne dois JAMAIS inventer de chiffres. Utilise uniquement les données fournies.
Si un indicateur est manquant, signale-le explicitement comme lacune."""},
            {"role": "user", "content": f"""Génère un rapport ESG structuré pour l'année {donnees.annee_reporting}.

ENTREPRISE : {donnees.entreprise}
COMPLÉTUDE DES DONNÉES : {donnees.completude_pct}%

INDICATEURS :
{indicateurs_json}

Structure le rapport en 3 sections (Environnement, Social, Gouvernance).
Pour chaque section :
- Rédige un commentaire analytique des indicateurs
- Compare avec N-1 si disponible
- Identifie les points forts et axes d'amélioration
- Évalue la conformité CSRD

Produis un JSON avec le format RapportESG."""}
        ]
    )
    result = json.loads(response.choices[0].message.content)
    return RapportESG(**result)`,
            filename: "agent_esg.py",
          },
        ],
      },
      {
        title: "API et génération PDF",
        content:
          "Déployez l'API de génération de rapports avec export PDF. Le rapport est généré à partir d'un template Jinja2 et converti en PDF via WeasyPrint pour un rendu professionnel prêt à publier.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from fastapi.responses import FileResponse
from agent_esg import generer_rapport, RapportESG
from collecte_esg import collecter_donnees_rh, collecter_donnees_environnement, DonneesESG
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import os

app = FastAPI()

@app.post("/api/esg/generer")
async def generer_rapport_esg():
    annee = int(os.getenv("YEAR_REPORTING"))

    # Collecte depuis toutes les sources
    indicateurs = []
    indicateurs.extend(collecter_donnees_rh())
    indicateurs.extend(collecter_donnees_environnement())

    donnees = DonneesESG(
        entreprise="Mon Entreprise SAS",
        annee_reporting=annee,
        indicateurs=indicateurs,
        completude_pct=round(len(indicateurs) / 30 * 100, 1)
    )

    rapport = generer_rapport(donnees)

    # Génération PDF
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("rapport_esg.html")
    html_content = template.render(rapport=rapport.model_dump())

    pdf_path = f"output/rapport_esg_{annee}.pdf"
    HTML(string=html_content).write_pdf(pdf_path)

    return {
        "rapport": rapport.model_dump(),
        "pdf_url": f"/api/esg/download/{annee}",
        "completude": donnees.completude_pct
    }

@app.get("/api/esg/download/{annee}")
async def download_rapport(annee: int):
    pdf_path = f"output/rapport_esg_{annee}.pdf"
    return FileResponse(pdf_path, media_type="application/pdf",
                       filename=f"rapport_esg_{annee}.pdf")`,
            filename: "api_esg.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données individuelles RH (noms, salaires, évaluations) sont agrégées avant envoi au LLM — seuls les indicateurs statistiques (moyennes, taux, totaux) sont transmis. Aucune donnée nominative ne quitte le périmètre interne. Les données fournisseurs sont anonymisées (code fournisseur uniquement).",
      auditLog: "Chaque génération de rapport est tracée : date, version, sources de données utilisées, complétude, indicateurs calculés, modifications manuelles apportées par le responsable RSE. Historique complet des versions du rapport avec diff entre versions. Rétention : durée légale de 10 ans.",
      humanInTheLoop: "Le rapport généré est systématiquement relu et validé par le responsable RSE avant publication. Les chiffres clés (émissions carbone, effectifs, investissements) nécessitent une double validation (opérationnel + direction). Comité de validation ESG avant publication finale.",
      monitoring: "Dashboard de suivi : taux de complétude des données par département, statut de collecte par source, comparaison N/N-1 des indicateurs, alertes si un indicateur dévie de plus de 20% vs N-1 (possible erreur de données), coût de génération par rapport, historique des scores de conformité CSRD.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (mensuel) → Node HTTP Request (API SIRH données RH) + Node Google Sheets (données environnement) + Node HTTP Request (API ERP données finance) → Node Merge (consolidation) → Node Code (calcul indicateurs GRI) → Node HTTP Request (API LLM génération rapport) → Node IF (complétude > 80%) → Branch complète: Node Google Drive (stockage PDF) + Node Email (envoi direction RSE) → Branch incomplète: Node Slack (alerte données manquantes).",
      nodes: ["Schedule Trigger (mensuel)", "HTTP Request (SIRH)", "Google Sheets (environnement)", "HTTP Request (ERP)", "Merge (consolidation)", "Code (calcul KPI)", "HTTP Request (LLM génération)", "IF (complétude)", "Google Drive (stockage)", "Email (envoi direction)", "Slack (alerte)"],
      triggerType: "Schedule (cron mensuel le 5 du mois)",
    },
    estimatedTime: "6-10h",
    difficulty: "Expert",
    sectors: ["Industrie", "Finance", "Énergie", "Grande distribution", "Services"],
    metiers: ["RSE / Développement Durable", "Direction Générale", "Conformité"],
    functions: ["RSE"],
    metaTitle: "Agent IA de Génération de Rapports ESG/CSRD — Guide Complet",
    metaDescription:
      "Automatisez la production de vos rapports ESG avec un agent IA. Collecte de données multi-départements, conformité CSRD/GRI, génération PDF. Tutoriel pas-à-pas pour entreprises réglementées.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-optimisation-campagnes-ads",
    title: "Agent d'Optimisation des Campagnes Publicitaires",
    subtitle: "Optimisez automatiquement vos dépenses publicitaires sur Google Ads, Meta et LinkedIn grâce à l'IA",
    problem:
      "Les équipes marketing gèrent des campagnes publicitaires sur de multiples plateformes (Google Ads, Meta Ads, LinkedIn Ads) avec des budgets croissants mais une optimisation manuelle chronophage et réactive. Les ajustements de budget, d'enchères et de ciblage sont faits trop tard, les données de performance sont consultées en silos par plateforme, et les corrélations cross-canal restent invisibles. Résultat : un ROAS sous-optimal et un gaspillage de budget estimé entre 20% et 40%.",
    value:
      "Un agent IA analyse en temps réel les performances de toutes vos campagnes publicitaires, identifie les créas et audiences les plus performantes, réalloue automatiquement les budgets entre plateformes et campagnes, et ajuste les enchères pour maximiser le ROAS. L'optimisation cross-canal permet de détecter des synergies invisibles à l'oeil humain.",
    inputs: [
      "Données de performance Google Ads (impressions, clics, conversions, CPA, ROAS)",
      "Données de performance Meta Ads (reach, engagement, conversions, CPM)",
      "Données de performance LinkedIn Ads (impressions, clics, leads, CPL)",
      "Budget total et contraintes d'allocation par plateforme",
      "Objectifs de campagne (conversions, notoriété, leads) et KPI cibles",
    ],
    outputs: [
      "Recommandations d'allocation budgétaire optimisée par plateforme et campagne",
      "Ajustements d'enchères automatiques par mot-clé / audience",
      "Rapport de performance cross-canal unifié avec attribution",
      "Alertes sur les campagnes sous-performantes ou les anomalies de dépense",
      "Prévisions de performance à 7/14/30 jours avec intervalles de confiance",
    ],
    risks: [
      "Réallocation trop agressive causant un arrêt de diffusion sur certaines campagnes",
      "Données de conversion retardées (attribution post-view) faussant l'optimisation en temps réel",
      "Sur-optimisation court-terme au détriment du branding et de la notoriété long-terme",
      "Dépendance aux API tierces avec risques de changements de format ou de quotas",
    ],
    roiIndicatif:
      "Amélioration de 25% à 40% du ROAS global. Réduction de 30% du CPA moyen. Économie de 15h par semaine de travail manuel d'optimisation. Détection des anomalies de dépense 6x plus rapide.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
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
│  Plateformes│────▶│  Agent LLM   │────▶│  Actions    │
│  Ads        │     │  (Analyse &  │     │  (Budget /  │
│  (API)      │     │  Optimisation│     │  Enchères)  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │  Google   │ │   Meta   │ │ LinkedIn │
       │  Ads API  │ │  Ads API │ │ Ads API  │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez les accès aux APIs publicitaires. Chaque plateforme nécessite des credentials OAuth2 spécifiques. Prévoyez un compte développeur sur chaque plateforme.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary google-ads facebook-business python-dotenv fastapi pandas`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/ads_optimizer_db
GOOGLE_ADS_DEVELOPER_TOKEN=...
GOOGLE_ADS_CLIENT_ID=...
GOOGLE_ADS_CLIENT_SECRET=...
GOOGLE_ADS_REFRESH_TOKEN=...
GOOGLE_ADS_CUSTOMER_ID=123-456-7890
META_APP_ID=...
META_APP_SECRET=...
META_ACCESS_TOKEN=...
META_AD_ACCOUNT_ID=act_123456789
LINKEDIN_ACCESS_TOKEN=...
LINKEDIN_AD_ACCOUNT_ID=...
SLACK_WEBHOOK_ADS=https://hooks.slack.com/services/...`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte des données de performance",
        content:
          "Connectez-vous aux APIs de chaque plateforme publicitaire et récupérez les métriques de performance dans un format unifié. La normalisation des données est essentielle pour permettre une comparaison cross-canal pertinente.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date
import os

class CampaignMetrics(BaseModel):
    plateforme: Literal["google_ads", "meta_ads", "linkedin_ads"]
    campaign_id: str
    campaign_name: str
    date_debut: date
    date_fin: date
    budget_quotidien: float
    depense_totale: float
    impressions: int
    clics: int
    conversions: float
    revenu: float
    ctr: float = Field(description="Click-through rate en %")
    cpc: float = Field(description="Coût par clic")
    cpa: float = Field(description="Coût par acquisition")
    roas: float = Field(description="Return on Ad Spend")
    cpm: float = Field(description="Coût pour mille impressions")

class PerformanceGlobale(BaseModel):
    periode: str
    budget_total: float
    depense_totale: float
    campagnes: List[CampaignMetrics]
    roas_global: float
    cpa_moyen: float

def collecter_google_ads(date_from: date, date_to: date) -> List[CampaignMetrics]:
    from google.ads.googleads.client import GoogleAdsClient
    client = GoogleAdsClient.load_from_env()
    service = client.get_service("GoogleAdsService")
    customer_id = os.getenv("GOOGLE_ADS_CUSTOMER_ID").replace("-", "")

    query = f"""
        SELECT campaign.id, campaign.name,
               metrics.impressions, metrics.clicks, metrics.conversions,
               metrics.conversions_value, metrics.cost_micros,
               campaign.campaign_budget
        FROM campaign
        WHERE segments.date BETWEEN '{date_from}' AND '{date_to}'
        AND campaign.status = 'ENABLED'
    """
    response = service.search(customer_id=customer_id, query=query)
    campagnes = []
    for row in response:
        cost = row.metrics.cost_micros / 1_000_000
        convs = row.metrics.conversions
        campagnes.append(CampaignMetrics(
            plateforme="google_ads",
            campaign_id=str(row.campaign.id),
            campaign_name=row.campaign.name,
            date_debut=date_from,
            date_fin=date_to,
            budget_quotidien=0,
            depense_totale=cost,
            impressions=row.metrics.impressions,
            clics=row.metrics.clicks,
            conversions=convs,
            revenu=row.metrics.conversions_value,
            ctr=round(row.metrics.clicks / max(row.metrics.impressions, 1) * 100, 2),
            cpc=round(cost / max(row.metrics.clicks, 1), 2),
            cpa=round(cost / max(convs, 0.01), 2),
            roas=round(row.metrics.conversions_value / max(cost, 0.01), 2),
            cpm=round(cost / max(row.metrics.impressions, 1) * 1000, 2)
        ))
    return campagnes

def collecter_meta_ads(date_from: date, date_to: date) -> List[CampaignMetrics]:
    from facebook_business.api import FacebookAdsApi
    from facebook_business.adobjects.adaccount import AdAccount
    FacebookAdsApi.init(os.getenv("META_APP_ID"), os.getenv("META_APP_SECRET"),
                        os.getenv("META_ACCESS_TOKEN"))
    account = AdAccount(os.getenv("META_AD_ACCOUNT_ID"))
    campaigns = account.get_campaigns(
        fields=["name", "status"],
        params={"effective_status": ["ACTIVE"]}
    )
    campagnes = []
    for camp in campaigns:
        insights = camp.get_insights(params={
            "time_range": {"since": str(date_from), "until": str(date_to)},
            "fields": ["impressions", "clicks", "spend", "actions", "action_values"]
        })
        for row in insights:
            spend = float(row.get("spend", 0))
            conversions = sum(a["value"] for a in row.get("actions", [])
                            if a["action_type"] == "offsite_conversion") if row.get("actions") else 0
            revenue = sum(float(a["value"]) for a in row.get("action_values", [])
                        if a["action_type"] == "offsite_conversion") if row.get("action_values") else 0
            campagnes.append(CampaignMetrics(
                plateforme="meta_ads",
                campaign_id=camp["id"],
                campaign_name=camp["name"],
                date_debut=date_from, date_fin=date_to,
                budget_quotidien=0, depense_totale=spend,
                impressions=int(row.get("impressions", 0)),
                clics=int(row.get("clicks", 0)),
                conversions=float(conversions), revenu=float(revenue),
                ctr=round(int(row.get("clicks", 0)) / max(int(row.get("impressions", 1)), 1) * 100, 2),
                cpc=round(spend / max(int(row.get("clicks", 1)), 1), 2),
                cpa=round(spend / max(float(conversions), 0.01), 2),
                roas=round(float(revenue) / max(spend, 0.01), 2),
                cpm=round(spend / max(int(row.get("impressions", 1)), 1) * 1000, 2)
            ))
    return campagnes`,
            filename: "collecte_ads.py",
          },
        ],
      },
      {
        title: "Agent d'optimisation",
        content:
          "L'agent analyse les performances cross-canal, identifie les opportunités d'optimisation et génère des recommandations actionnables de réallocation budgétaire et d'ajustement d'enchères.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from collecte_ads import PerformanceGlobale, CampaignMetrics
from pydantic import BaseModel, Field
from typing import List
import json

class RecommandationAds(BaseModel):
    campaign_id: str
    campaign_name: str
    plateforme: str
    action: str = Field(description="augmenter_budget, reduire_budget, pauser, ajuster_enchere, modifier_ciblage")
    variation_budget_pct: float = Field(description="Variation de budget recommandée en %")
    justification: str
    impact_estime_roas: float = Field(description="Impact estimé sur le ROAS")
    priorite: str = Field(description="haute, moyenne, basse")

class PlanOptimisation(BaseModel):
    recommandations: List[RecommandationAds]
    budget_reallocation: dict = Field(description="Nouvelle répartition du budget par plateforme")
    roas_prevu: float
    economies_estimees: float
    synthese: str

client = anthropic.Anthropic()

def optimiser_campagnes(performance: PerformanceGlobale) -> PlanOptimisation:
    perf_json = json.dumps(
        [c.model_dump() for c in performance.campagnes],
        ensure_ascii=False, indent=2, default=str
    )
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": f"""Tu es un expert en acquisition digitale et optimisation media.
Analyse les performances suivantes et produis un plan d'optimisation.

BUDGET TOTAL : {performance.budget_total} EUR
DÉPENSE TOTALE : {performance.depense_totale} EUR
ROAS GLOBAL : {performance.roas_global}
CPA MOYEN : {performance.cpa_moyen} EUR

CAMPAGNES :
{perf_json}

RÈGLES D'OPTIMISATION :
- Réallouer le budget des campagnes avec ROAS < 1.5 vers celles avec ROAS > 3
- Ne jamais couper plus de 30% du budget d'une campagne en une fois
- Prendre en compte la phase de la campagne (apprentissage Meta = min 50 conversions)
- Campagne avec CPA > 2x CPA moyen = candidate à la pause
- Campagne avec CTR < 0.5% sur Google Search = revoir les annonces
- Toujours garder un minimum de 20% du budget en test/expérimentation

Produis un JSON PlanOptimisation avec des recommandations actionnables."""}
        ]
    )
    result = json.loads(response.content[0].text)
    return PlanOptimisation(**result)`,
            filename: "agent_ads.py",
          },
        ],
      },
      {
        title: "API et exécution automatique",
        content:
          "Déployez l'API d'optimisation avec la possibilité d'appliquer automatiquement les recommandations validées. Le système inclut un mode simulation (dry-run) pour prévisualiser les changements avant application.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from agent_ads import optimiser_campagnes, PlanOptimisation
from collecte_ads import collecter_google_ads, collecter_meta_ads, PerformanceGlobale
from datetime import date, timedelta
import requests
import os

app = FastAPI()
SLACK_WEBHOOK = os.getenv("SLACK_WEBHOOK_ADS")

@app.post("/api/ads/optimiser")
async def lancer_optimisation(dry_run: bool = True):
    today = date.today()
    date_from = today - timedelta(days=7)

    # Collecte cross-canal
    campagnes = []
    campagnes.extend(collecter_google_ads(date_from, today))
    campagnes.extend(collecter_meta_ads(date_from, today))

    depense_totale = sum(c.depense_totale for c in campagnes)
    revenu_total = sum(c.revenu for c in campagnes)
    convs_total = sum(c.conversions for c in campagnes)

    performance = PerformanceGlobale(
        periode=f"{date_from} - {today}",
        budget_total=depense_totale * 1.2,
        depense_totale=depense_totale,
        campagnes=campagnes,
        roas_global=round(revenu_total / max(depense_totale, 0.01), 2),
        cpa_moyen=round(depense_totale / max(convs_total, 0.01), 2)
    )

    plan = optimiser_campagnes(performance)

    # Notification Slack
    if SLACK_WEBHOOK:
        reco_text = "\\n".join([
            f"• {r.campaign_name} ({r.plateforme}): {r.action} ({r.variation_budget_pct:+.0f}%) - {r.justification}"
            for r in plan.recommandations[:5]
        ])
        requests.post(SLACK_WEBHOOK, json={
            "text": f"📊 Plan d'optimisation Ads généré\\n"
                    f"ROAS actuel: {performance.roas_global} → Prévu: {plan.roas_prevu}\\n"
                    f"Économies estimées: {plan.economies_estimees:.0f} EUR\\n"
                    f"Mode: {'SIMULATION' if dry_run else 'APPLICATION'}\\n\\n"
                    f"Top recommandations :\\n{reco_text}"
        })

    if not dry_run:
        # Appliquer les changements via les APIs
        for reco in plan.recommandations:
            if reco.priorite == "haute":
                appliquer_recommandation(reco)

    return {
        "plan": plan.model_dump(),
        "mode": "dry_run" if dry_run else "applied",
        "campagnes_analysees": len(campagnes)
    }

def appliquer_recommandation(reco):
    """Applique une recommandation via l'API de la plateforme concernée"""
    # Implémentation spécifique par plateforme
    pass`,
            filename: "api_ads.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle utilisateur n'est transmise au LLM — seules les métriques agrégées de campagne (impressions, clics, conversions, dépense) sont envoyées. Les données d'audience et de ciblage restent dans les plateformes publicitaires. Les identifiants de campagne internes ne sont pas exposés.",
      auditLog: "Chaque cycle d'optimisation est tracé : date, performance avant/après, recommandations générées, recommandations appliquées, résultat observé à J+7. Historique complet des modifications de budget et d'enchères avec rollback possible. Rétention : 36 mois pour les données de performance.",
      humanInTheLoop: "Mode dry-run par défaut — les recommandations sont présentées pour validation humaine avant application. Les changements de budget supérieurs à 20% nécessitent une approbation du responsable acquisition. Revue hebdomadaire des performances post-optimisation avec le Head of Marketing.",
      monitoring: "Dashboard temps réel : ROAS par plateforme et par campagne, CPA et CPL avec tendances, budget consommé vs alloué, alertes si le CPA dépasse 150% de la cible, alertes si une campagne dépense sans convertir pendant 48h, coût API LLM par cycle d'optimisation.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (quotidien 7h) → Node HTTP Request (Google Ads API) + Node HTTP Request (Meta Ads API) + Node HTTP Request (LinkedIn Ads API) → Node Merge (consolidation cross-canal) → Node HTTP Request (API LLM optimisation) → Node IF (dry_run ou apply) → Branch simulation: Node Slack (rapport recommandations) → Branch application: Node HTTP Request (modifier budgets APIs) + Node Slack (confirmation) → Node Google Sheets (historique).",
      nodes: ["Schedule Trigger (quotidien 7h)", "HTTP Request (Google Ads)", "HTTP Request (Meta Ads)", "HTTP Request (LinkedIn Ads)", "Merge (consolidation)", "HTTP Request (LLM optimisation)", "IF (dry_run)", "Slack (recommandations)", "HTTP Request (modifier budgets)", "Google Sheets (historique)"],
      triggerType: "Schedule (cron quotidien à 7h00)",
    },
    estimatedTime: "6-8h",
    difficulty: "Expert",
    sectors: ["E-commerce", "SaaS", "Services", "Retail", "Startup"],
    metiers: ["Marketing Digital", "Growth", "Acquisition"],
    functions: ["Marketing"],
    metaTitle: "Agent IA d'Optimisation des Campagnes Publicitaires — Guide Complet",
    metaDescription:
      "Optimisez automatiquement vos campagnes Google Ads, Meta et LinkedIn avec un agent IA. Réallocation budgétaire, ajustement d'enchères et analyse cross-canal. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-gestion-connaissances-juridiques",
    title: "Agent de Gestion des Connaissances Juridiques",
    subtitle: "Un assistant RAG intelligent pour les directions juridiques d'entreprise",
    problem:
      "Les directions juridiques croulent sous un volume documentaire considérable : contrats, jurisprudences, réglementations, notes internes, avis juridiques. Retrouver une clause spécifique dans des milliers de contrats, vérifier la conformité d'une pratique avec la dernière réglementation, ou identifier un précédent jurisprudentiel prend des heures de recherche manuelle. Les connaissances sont souvent concentrées chez quelques experts seniors, créant un risque de perte de savoir critique.",
    value:
      "Un agent RAG (Retrieval-Augmented Generation) indexe l'intégralité du corpus juridique de l'entreprise et permet aux juristes d'interroger cette base en langage naturel. L'agent retrouve les documents pertinents, synthétise les informations, compare les clauses entre contrats, et produit des analyses juridiques sourcées avec références précises aux documents originaux.",
    inputs: [
      "Corpus de contrats (PDF, Word) avec métadonnées (type, date, parties, statut)",
      "Base jurisprudentielle interne et externe (décisions de justice pertinentes)",
      "Textes réglementaires et législatifs (codes, directives, décrets)",
      "Notes et avis juridiques internes",
      "Questions en langage naturel des juristes",
    ],
    outputs: [
      "Réponses sourcées avec références précises aux documents (page, paragraphe, clause)",
      "Synthèses comparatives de clauses entre plusieurs contrats",
      "Alertes de non-conformité réglementaire sur les contrats existants",
      "Fiches de synthèse jurisprudentielle sur un sujet donné",
      "Suggestions de clauses types basées sur les précédents internes",
    ],
    risks: [
      "Hallucination juridique : l'agent invente des références ou des interprétations",
      "Omission de documents pertinents dans la recherche vectorielle (recall insuffisant)",
      "Interprétation erronée de clauses ambiguës hors contexte",
      "Confidentialité : risque de fuite de contrats sensibles via le LLM",
    ],
    roiIndicatif:
      "Réduction de 60% du temps de recherche juridique. Économie de 80 000 EUR annuels en heures de juristes seniors sur les recherches documentaires. Détection de 30% de risques contractuels supplémentaires grâce à l'analyse systématique.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Pinecone", category: "Database" },
      { name: "Vercel", category: "Hosting" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mixtral", category: "LLM", isFree: true },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Question   │────▶│  Agent RAG   │────▶│  Réponse    │
│  juriste    │     │  (Retrieval  │     │  sourcée    │
│  (NL)       │     │  + Génération│     │  + Réfs     │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │ Contrats  │ │ Jurispru-│ │ Réglemen- │
       │ (Vector)  │ │ dence    │ │ tation   │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires pour le pipeline RAG. Vous aurez besoin d'un extracteur de texte PDF robuste (pymupdf ou unstructured) et d'une base vectorielle pour l'indexation des documents juridiques.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain pinecone-client pymupdf unstructured python-dotenv fastapi tiktoken`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX=juridique-kb
DOCS_PATH=./corpus_juridique
CHUNK_SIZE=1000
CHUNK_OVERLAP=200`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Indexation du corpus juridique",
        content:
          "Indexez l'ensemble du corpus juridique dans la base vectorielle. Le découpage (chunking) est critique pour les documents juridiques : il faut préserver les clauses complètes et leur contexte. Un chunking trop fin perd le contexte, trop large dilue la pertinence.",
        codeSnippets: [
          {
            language: "python",
            code: `from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, PyMuPDFLoader
from typing import List
import os

def charger_corpus(docs_path: str) -> List:
    """Charge tous les documents juridiques (PDF, DOCX)"""
    loader = DirectoryLoader(
        docs_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"{len(documents)} pages chargées depuis {docs_path}")
    return documents

def decouper_documents(documents: List) -> List:
    """Découpe les documents en chunks optimisés pour le juridique"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
        separators=[
            "\\nArticle ",     # Articles de contrats
            "\\nClause ",      # Clauses
            "\\nSection ",     # Sections
            "\\nChapitre ",    # Chapitres
            "\\n\\n",          # Paragraphes
            "\\n",             # Lignes
            ". ",              # Phrases
        ]
    )
    chunks = splitter.split_documents(documents)
    # Enrichir chaque chunk avec des métadonnées
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source_file"] = chunk.metadata.get("source", "inconnu")
        chunk.metadata["page"] = chunk.metadata.get("page", 0)
    print(f"{len(chunks)} chunks créés")
    return chunks

def indexer_corpus():
    """Pipeline complet d'indexation"""
    docs_path = os.getenv("DOCS_PATH", "./corpus_juridique")
    documents = charger_corpus(docs_path)
    chunks = decouper_documents(documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = Pinecone.from_documents(
        chunks,
        embeddings,
        index_name=os.getenv("PINECONE_INDEX"),
        batch_size=100
    )
    print(f"Indexation terminée : {len(chunks)} chunks dans Pinecone")
    return vectorstore

if __name__ == "__main__":
    indexer_corpus()`,
            filename: "indexation_juridique.py",
          },
        ],
      },
      {
        title: "Agent RAG juridique",
        content:
          "L'agent RAG combine la recherche vectorielle avec la génération augmentée. Il retrouve les passages les plus pertinents du corpus puis génère une réponse structurée avec des références précises aux documents sources. Un système de re-ranking améliore la pertinence des résultats.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os

class Reference(BaseModel):
    document: str
    page: int
    extrait: str = Field(description="Passage pertinent extrait du document")
    pertinence: float = Field(ge=0, le=1)

class ReponseJuridique(BaseModel):
    reponse: str = Field(description="Réponse détaillée à la question juridique")
    references: List[Reference] = Field(description="Documents sources avec extraits")
    niveau_confiance: str = Field(description="élevé, moyen, faible")
    points_attention: List[str] = Field(description="Points nécessitant une vérification humaine")
    suggestions_recherche: List[str] = Field(description="Recherches complémentaires suggérées")

# Initialisation
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Pinecone.from_existing_index(
    os.getenv("PINECONE_INDEX"), embeddings
)
client = OpenAI()

def rechercher_et_repondre(question: str, top_k: int = 10) -> ReponseJuridique:
    # Étape 1 : Recherche vectorielle
    docs = vectorstore.similarity_search_with_score(question, k=top_k)

    # Étape 2 : Préparer le contexte avec métadonnées
    contexte_parts = []
    for doc, score in docs:
        source = doc.metadata.get("source_file", "Document inconnu")
        page = doc.metadata.get("page", 0)
        contexte_parts.append(
            f"[Source: {source} | Page: {page} | Pertinence: {score:.3f}]\\n{doc.page_content}"
        )
    contexte = "\\n\\n---\\n\\n".join(contexte_parts)

    # Étape 3 : Génération de la réponse
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": """Tu es un assistant juridique expert pour une direction juridique d'entreprise.

RÈGLES STRICTES :
1. Ne réponds QUE sur la base des documents fournis dans le contexte
2. Si l'information n'est pas dans les documents, dis-le explicitement
3. CITE TOUJOURS tes sources avec le nom du document et le numéro de page
4. N'invente JAMAIS de références juridiques, d'articles de loi ou de jurisprudences
5. Signale tout point ambigu nécessitant une vérification humaine
6. Utilise un langage juridique précis mais accessible"""},
            {"role": "user", "content": f"""QUESTION : {question}

DOCUMENTS PERTINENTS :
{contexte}

Produis une réponse JSON structurée avec : reponse, references, niveau_confiance, points_attention, suggestions_recherche."""}
        ]
    )
    result = json.loads(response.choices[0].message.content)
    return ReponseJuridique(**result)`,
            filename: "agent_juridique.py",
          },
        ],
      },
      {
        title: "API et interface de recherche",
        content:
          "Exposez l'agent via une API REST avec des endpoints de recherche, d'analyse comparative de clauses, et de vérification de conformité. L'interface permet aux juristes d'interagir en langage naturel.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, Query
from agent_juridique import rechercher_et_repondre, ReponseJuridique
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

app = FastAPI(title="Assistant Juridique IA")

class QuestionRequest(BaseModel):
    question: str
    filtre_type_doc: Optional[str] = None  # contrat, jurisprudence, reglementation
    filtre_date_apres: Optional[str] = None

@app.post("/api/juridique/recherche")
async def recherche_juridique(req: QuestionRequest):
    result = rechercher_et_repondre(req.question)
    # Log de la requête pour audit
    log_requete(req.question, result)
    return result.model_dump()

@app.post("/api/juridique/comparer-clauses")
async def comparer_clauses(clause_type: str, contrat_ids: list[str]):
    """Compare une clause spécifique entre plusieurs contrats"""
    question = f"Compare la clause de {clause_type} entre les contrats suivants : {', '.join(contrat_ids)}. Identifie les différences et les risques."
    result = rechercher_et_repondre(question, top_k=20)
    return result.model_dump()

@app.post("/api/juridique/conformite")
async def verifier_conformite(contrat_id: str, reglementation: str):
    """Vérifie la conformité d'un contrat avec une réglementation"""
    question = f"Le contrat {contrat_id} est-il conforme à {reglementation} ? Identifie les clauses manquantes ou non conformes."
    result = rechercher_et_repondre(question, top_k=15)
    return result.model_dump()

def log_requete(question: str, result: ReponseJuridique):
    """Log pour audit et amélioration continue"""
    from sqlalchemy import create_engine, text
    import os
    engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///audit.db"))
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO audit_juridique (date, question, confiance, nb_refs, refs_docs)
            VALUES (:date, :question, :confiance, :nb_refs, :refs)
        """), {
            "date": datetime.now().isoformat(),
            "question": question,
            "confiance": result.niveau_confiance,
            "nb_refs": len(result.references),
            "refs": ",".join([r.document for r in result.references])
        })
        conn.commit()`,
            filename: "api_juridique.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contrats contiennent des informations confidentielles (parties, montants, conditions). En mode cloud, seuls les passages pertinents (chunks) sont envoyés au LLM, jamais les contrats complets. Pour les entreprises avec des exigences élevées, déploiement on-premise recommandé avec Ollama + Mixtral. Les noms de parties et montants peuvent être masqués avant envoi au LLM si nécessaire.",
      auditLog: "Chaque requête est tracée : horodatage, utilisateur (juriste), question posée, documents retrouvés, réponse générée, niveau de confiance, feedback du juriste (utile/pas utile). Les statistiques d'usage permettent d'identifier les lacunes du corpus. Rétention des logs : 5 ans (conformité réglementaire).",
      humanInTheLoop: "L'agent est un outil d'aide à la recherche — il ne remplace jamais l'avis d'un juriste qualifié. Les réponses avec un niveau de confiance 'faible' sont signalées en rouge. Les analyses de conformité nécessitent une validation par un juriste senior avant transmission au client interne. Feedback obligatoire sur la pertinence pour améliorer le modèle.",
      monitoring: "Dashboard usage : nombre de requêtes par jour/semaine, temps de réponse moyen, taux de satisfaction des juristes (feedback), documents les plus consultés, questions sans réponse satisfaisante (gap du corpus), coût API par requête, alertes si le temps de réponse dépasse 30 secondes ou si le taux de confiance moyen chute sous 60%.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (requête juriste) → Node HTTP Request (recherche vectorielle Pinecone) → Node Code (re-ranking et filtrage) → Node HTTP Request (API LLM génération réponse) → Node IF (confiance > seuil) → Branch confiance haute: Node Slack (réponse directe au juriste) → Branch confiance basse: Node Email (escalade juriste senior) → Node PostgreSQL (audit log).",
      nodes: ["Webhook (requête juriste)", "HTTP Request (Pinecone)", "Code (re-ranking)", "HTTP Request (LLM)", "IF (confiance)", "Slack (réponse)", "Email (escalade)", "PostgreSQL (audit)"],
      triggerType: "Webhook (requête depuis l'interface juriste)",
    },
    estimatedTime: "5-8h",
    difficulty: "Expert",
    sectors: ["Finance", "Industrie", "Services", "Assurance", "Immobilier"],
    metiers: ["Direction Juridique", "Compliance", "Secrétariat Général"],
    functions: ["Juridique"],
    metaTitle: "Agent RAG Juridique pour Direction Juridique — Guide Complet",
    metaDescription:
      "Déployez un assistant IA RAG pour votre direction juridique. Recherche documentaire intelligente, analyse de contrats, vérification de conformité. Tutoriel pas-à-pas avec stack RAG complète.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-scoring-credit",
    title: "Agent de Scoring Crédit Automatisé",
    subtitle: "Automatisez l'évaluation de solvabilité avec un agent IA intégrant de multiples sources de données",
    problem:
      "L'évaluation de solvabilité traditionnelle repose sur des modèles statistiques rigides et un nombre limité de variables (historique bancaire, revenus déclarés). Le processus d'instruction est lent (48-72h), ne prend pas en compte les données alternatives, et pénalise les profils atypiques (freelances, néo-entrepreneurs, jeunes actifs) qui n'ont pas d'historique bancaire classique. Les fintechs et banques challenger ont besoin d'un scoring plus rapide, plus inclusif et plus précis.",
    value:
      "Un agent IA orchestre la collecte de données multi-sources (open banking, données fiscales, données alternatives), applique des modèles ML de scoring, et génère une décision de crédit argumentée en moins de 5 minutes. Le modèle intègre des données alternatives (transactions, comportement de remboursement, données professionnelles) pour un scoring plus fin et plus inclusif.",
    inputs: [
      "Données open banking (transactions, soldes, crédits en cours) via API DSP2",
      "Données fiscales (revenus, charges, patrimoine déclaré)",
      "Données du bureau de crédit (Banque de France, fichiers d'incidents)",
      "Données alternatives (historique de paiement loyer, factures, abonnements)",
      "Informations du demandeur (profession, ancienneté, situation familiale)",
    ],
    outputs: [
      "Score de crédit (0-1000) avec niveau de risque (A à E)",
      "Probabilité de défaut à 12, 24 et 36 mois",
      "Décision argumentée (accepté, refusé, contre-proposition) avec justification",
      "Montant maximum recommandé et taux proposé",
      "Rapport de scoring détaillé avec contribution de chaque variable",
    ],
    risks: [
      "Biais discriminatoire dans le scoring (géographique, socio-démographique, ethnique indirect)",
      "Non-conformité réglementaire (droit à l'explication RGPD, réglementation bancaire)",
      "Modèle adversarial : tentatives de fraude par manipulation des données d'entrée",
      "Drift du modèle : dégradation des performances si les conditions économiques changent",
    ],
    roiIndicatif:
      "Réduction du temps d'instruction de 72h à 5 minutes. Diminution de 25% du taux de défaut grâce à un scoring plus précis. Augmentation de 15% du taux d'acceptation grâce à l'intégration de données alternatives. Économie de 40% sur les coûts d'instruction par dossier.",
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
│  Demande    │────▶│  Agent IA    │────▶│  Décision   │
│  crédit     │     │  (Scoring ML │     │  argumentée │
│  (formulaire)│    │  + LLM)      │     │  + rapport  │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │  Open     │ │ Bureau   │ │ Données  │
       │  Banking  │ │ Crédit   │ │ Alterna- │
       │  (DSP2)   │ │ (BdF)   │ │ tives    │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances nécessaires et configurez les accès aux APIs de données. L'accès aux APIs open banking (DSP2) et au bureau de crédit nécessite des agréments spécifiques.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary pandas scikit-learn xgboost python-dotenv fastapi shap`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/scoring_db
OPEN_BANKING_API_URL=https://api.openbanking-provider.com/v2
OPEN_BANKING_CLIENT_ID=...
OPEN_BANKING_CLIENT_SECRET=...
CREDIT_BUREAU_API_URL=https://api.credit-bureau.fr/v1
CREDIT_BUREAU_API_KEY=...
RISK_THRESHOLD_ACCEPT=650
RISK_THRESHOLD_REVIEW=450`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte et enrichissement des données",
        content:
          "Collectez les données depuis les différentes sources (open banking, bureau de crédit, données alternatives) et construisez le profil financier complet du demandeur. Chaque source est interrogée via son API dédiée.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date
import requests
import os

class ProfilFinancier(BaseModel):
    demandeur_id: str
    # Données déclaratives
    revenu_mensuel_net: float
    charges_mensuelles: float
    profession: str
    anciennete_emploi_mois: int
    situation_familiale: str
    personnes_a_charge: int
    # Données Open Banking
    solde_moyen_3m: float = 0
    revenus_reguliers_detectes: float = 0
    depenses_jeu_alcool_pct: float = 0
    nb_rejets_prelevement_6m: int = 0
    nb_credits_en_cours: int = 0
    mensualites_credits: float = 0
    epargne_detectee: float = 0
    variation_solde_tendance: float = 0
    # Données bureau de crédit
    score_bureau_credit: Optional[int] = None
    incidents_paiement: int = 0
    fichage_bdf: bool = False
    # Données alternatives
    regularite_loyer_12m: Optional[float] = None
    anciennete_adresse_mois: int = 0
    # Ratios calculés
    taux_endettement: float = 0
    reste_a_vivre: float = 0
    capacite_remboursement: float = 0

def collecter_open_banking(consent_token: str, demandeur_id: str) -> dict:
    """Collecte les données via API Open Banking (DSP2)"""
    api_url = os.getenv("OPEN_BANKING_API_URL")
    headers = {"Authorization": f"Bearer {consent_token}"}

    # Récupérer les comptes
    comptes = requests.get(f"{api_url}/accounts", headers=headers).json()

    # Récupérer les transactions (6 derniers mois)
    transactions = []
    for compte in comptes["accounts"]:
        txs = requests.get(
            f"{api_url}/accounts/{compte['id']}/transactions",
            headers=headers,
            params={"from": "2024-07-01", "to": "2025-01-01"}
        ).json()
        transactions.extend(txs["transactions"])

    # Analyser les transactions
    import pandas as pd
    df = pd.DataFrame(transactions)
    df["amount"] = df["amount"].astype(float)
    df["date"] = pd.to_datetime(df["bookingDate"])

    solde_moyen = df.groupby(df["date"].dt.to_period("M"))["amount"].sum().mean()
    revenus = df[df["amount"] > 0].groupby(df[df["amount"] > 0]["date"].dt.to_period("M"))["amount"].sum().mean()
    rejets = len(df[df["status"] == "rejected"])

    return {
        "solde_moyen_3m": round(solde_moyen, 2),
        "revenus_reguliers_detectes": round(revenus, 2),
        "nb_rejets_prelevement_6m": rejets,
        "nb_credits_en_cours": len(df[df["category"] == "loan_repayment"]["creditorName"].unique()),
        "mensualites_credits": abs(df[df["category"] == "loan_repayment"]["amount"].sum() / 6),
        "epargne_detectee": df[df["category"] == "savings"]["amount"].sum()
    }

def collecter_bureau_credit(identifiant_national: str) -> dict:
    """Interroge le bureau de crédit"""
    api_url = os.getenv("CREDIT_BUREAU_API_URL")
    api_key = os.getenv("CREDIT_BUREAU_API_KEY")
    response = requests.get(
        f"{api_url}/scoring/{identifiant_national}",
        headers={"X-API-Key": api_key}
    ).json()
    return {
        "score_bureau_credit": response.get("score"),
        "incidents_paiement": response.get("nb_incidents", 0),
        "fichage_bdf": response.get("ficp", False) or response.get("fcc", False)
    }

def construire_profil(demandeur_id: str, donnees_declaratives: dict,
                      consent_token: str, identifiant_national: str) -> ProfilFinancier:
    """Construit le profil financier complet"""
    ob_data = collecter_open_banking(consent_token, demandeur_id)
    cb_data = collecter_bureau_credit(identifiant_national)

    revenu = donnees_declaratives["revenu_mensuel_net"]
    charges = donnees_declaratives["charges_mensuelles"]
    mensualites = ob_data.get("mensualites_credits", 0)

    profil = ProfilFinancier(
        demandeur_id=demandeur_id,
        **donnees_declaratives,
        **ob_data,
        **cb_data,
        taux_endettement=round((mensualites + charges) / max(revenu, 1) * 100, 2),
        reste_a_vivre=round(revenu - charges - mensualites, 2),
        capacite_remboursement=round((revenu - charges - mensualites) * 0.33, 2)
    )
    return profil`,
            filename: "collecte_credit.py",
          },
        ],
      },
      {
        title: "Modèle de scoring et agent de décision",
        content:
          "Combinez un modèle ML (XGBoost) pour le scoring quantitatif avec un agent LLM pour la génération de la décision argumentée. Le LLM analyse le profil et le score ML pour produire une décision explicable et conforme aux exigences réglementaires.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import xgboost as xgb
import shap
import numpy as np
import json
from collecte_credit import ProfilFinancier
from pydantic import BaseModel, Field
from typing import List, Optional

class DecisionCredit(BaseModel):
    demandeur_id: str
    score_credit: int = Field(ge=0, le=1000, description="Score de 0 à 1000")
    niveau_risque: str = Field(description="A (excellent) à E (très risqué)")
    probabilite_defaut_12m: float = Field(ge=0, le=1)
    probabilite_defaut_24m: float = Field(ge=0, le=1)
    probabilite_defaut_36m: float = Field(ge=0, le=1)
    decision: str = Field(description="accepte, refuse, contre_proposition, revue_manuelle")
    montant_max_recommande: Optional[float] = None
    taux_propose: Optional[float] = None
    justification: str = Field(description="Explication détaillée de la décision")
    facteurs_positifs: List[str]
    facteurs_negatifs: List[str]
    conditions_speciales: List[str] = Field(default_factory=list)

# Charger le modèle XGBoost pré-entraîné
model = xgb.XGBClassifier()
model.load_model("models/scoring_credit_v2.json")
explainer = shap.TreeExplainer(model)

llm_client = anthropic.Anthropic()

def scorer_ml(profil: ProfilFinancier) -> tuple:
    """Calcule le score ML et les contributions SHAP"""
    features = np.array([[
        profil.revenu_mensuel_net,
        profil.charges_mensuelles,
        profil.taux_endettement,
        profil.reste_a_vivre,
        profil.anciennete_emploi_mois,
        profil.solde_moyen_3m,
        profil.nb_rejets_prelevement_6m,
        profil.nb_credits_en_cours,
        profil.incidents_paiement,
        int(profil.fichage_bdf),
        profil.epargne_detectee,
        profil.anciennete_adresse_mois,
        profil.regularite_loyer_12m or 0
    ]])
    proba_defaut = model.predict_proba(features)[0][1]
    score = int((1 - proba_defaut) * 1000)
    shap_values = explainer.shap_values(features)
    return score, proba_defaut, shap_values[0]

def decider_credit(profil: ProfilFinancier, montant_demande: float,
                   duree_mois: int) -> DecisionCredit:
    """Génère une décision de crédit complète"""
    score, proba_defaut, shap_vals = scorer_ml(profil)

    feature_names = ["revenu", "charges", "endettement", "reste_a_vivre",
                    "anciennete_emploi", "solde_moyen", "rejets_prelevement",
                    "credits_en_cours", "incidents", "fichage_bdf",
                    "epargne", "anciennete_adresse", "regularite_loyer"]
    contributions = dict(zip(feature_names, shap_vals.tolist()))

    # Niveau de risque
    if score >= 800: niveau = "A"
    elif score >= 650: niveau = "B"
    elif score >= 500: niveau = "C"
    elif score >= 350: niveau = "D"
    else: niveau = "E"

    profil_json = profil.model_dump_json()
    response = llm_client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[
            {"role": "user", "content": f"""Tu es un analyste crédit senior dans une banque française.
Génère une décision de crédit argumentée et conforme à la réglementation.

DEMANDE :
- Montant : {montant_demande} EUR
- Durée : {duree_mois} mois
- Mensualité estimée : {montant_demande / duree_mois:.2f} EUR/mois

PROFIL FINANCIER :
{profil_json}

SCORING ML :
- Score : {score}/1000
- Niveau de risque : {niveau}
- Probabilité de défaut 12 mois : {proba_defaut:.4f}
- Contributions des variables : {json.dumps(contributions, indent=2)}

RÈGLES RÉGLEMENTAIRES :
- Taux d'endettement max : 35% (HCSF)
- Fichage BdF = refus automatique
- Score < 350 = refus sauf exception motivée
- Score 350-450 = revue manuelle obligatoire
- Droit à l'explication RGPD : la décision doit être compréhensible par le demandeur

Produis un JSON DecisionCredit avec décision argumentée."""}
        ]
    )
    result = json.loads(response.content[0].text)
    result["demandeur_id"] = profil.demandeur_id
    result["score_credit"] = score
    return DecisionCredit(**result)`,
            filename: "agent_scoring.py",
          },
        ],
      },
      {
        title: "API et monitoring réglementaire",
        content:
          "Déployez l'API de scoring avec les endpoints d'évaluation, de suivi et de monitoring du modèle. Le système inclut un monitoring de drift pour détecter la dégradation des performances et des contrôles anti-biais.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, HTTPException
from agent_scoring import decider_credit, DecisionCredit
from collecte_credit import construire_profil, ProfilFinancier
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import os

app = FastAPI(title="API Scoring Crédit IA")

class DemandeCredit(BaseModel):
    demandeur_id: str
    montant: float
    duree_mois: int
    revenu_mensuel_net: float
    charges_mensuelles: float
    profession: str
    anciennete_emploi_mois: int
    situation_familiale: str
    personnes_a_charge: int
    consent_token: str  # Token de consentement Open Banking
    identifiant_national: str

@app.post("/api/scoring/evaluer")
async def evaluer_demande(demande: DemandeCredit):
    # Vérification fichage BdF préalable
    from collecte_credit import collecter_bureau_credit
    cb_data = collecter_bureau_credit(demande.identifiant_national)
    if cb_data.get("fichage_bdf"):
        return DecisionCredit(
            demandeur_id=demande.demandeur_id,
            score_credit=0,
            niveau_risque="E",
            probabilite_defaut_12m=1.0,
            probabilite_defaut_24m=1.0,
            probabilite_defaut_36m=1.0,
            decision="refuse",
            justification="Fichage Banque de France actif (FICP/FCC). Refus réglementaire automatique.",
            facteurs_positifs=[],
            facteurs_negatifs=["Fichage Banque de France actif"],
        ).model_dump()

    # Construction du profil complet
    profil = construire_profil(
        demandeur_id=demande.demandeur_id,
        donnees_declaratives={
            "revenu_mensuel_net": demande.revenu_mensuel_net,
            "charges_mensuelles": demande.charges_mensuelles,
            "profession": demande.profession,
            "anciennete_emploi_mois": demande.anciennete_emploi_mois,
            "situation_familiale": demande.situation_familiale,
            "personnes_a_charge": demande.personnes_a_charge,
        },
        consent_token=demande.consent_token,
        identifiant_national=demande.identifiant_national
    )

    # Scoring et décision
    decision = decider_credit(profil, demande.montant, demande.duree_mois)

    # Audit log
    log_decision(demande, decision)

    return decision.model_dump()

@app.get("/api/scoring/monitoring")
async def monitoring_modele():
    """Retourne les métriques de performance du modèle"""
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        stats = conn.execute(text("""
            SELECT
                COUNT(*) as total_demandes,
                AVG(score_credit) as score_moyen,
                COUNT(*) FILTER (WHERE decision = 'accepte') * 100.0 / COUNT(*) as taux_acceptation,
                COUNT(*) FILTER (WHERE decision = 'refuse') * 100.0 / COUNT(*) as taux_refus,
                COUNT(*) FILTER (WHERE decision = 'revue_manuelle') * 100.0 / COUNT(*) as taux_revue
            FROM decisions_credit
            WHERE date_decision >= NOW() - INTERVAL '30 days'
        """)).fetchone()

    return {
        "periode": "30 derniers jours",
        "total_demandes": stats.total_demandes,
        "score_moyen": round(stats.score_moyen, 0),
        "taux_acceptation": round(stats.taux_acceptation, 1),
        "taux_refus": round(stats.taux_refus, 1),
        "taux_revue_manuelle": round(stats.taux_revue, 1)
    }

def log_decision(demande, decision: DecisionCredit):
    """Log la décision pour audit réglementaire"""
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("DATABASE_URL"))
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO decisions_credit
            (date_decision, demandeur_id, montant, duree, score_credit,
             niveau_risque, decision, justification, proba_defaut_12m)
            VALUES (:date, :did, :montant, :duree, :score, :risque,
                    :decision, :justification, :proba)
        """), {
            "date": datetime.now().isoformat(),
            "did": demande.demandeur_id,
            "montant": demande.montant,
            "duree": demande.duree_mois,
            "score": decision.score_credit,
            "risque": decision.niveau_risque,
            "decision": decision.decision,
            "justification": decision.justification,
            "proba": decision.probabilite_defaut_12m
        })
        conn.commit()`,
            filename: "api_scoring.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données personnelles et financières sont traitées en environnement sécurisé (infrastructure certifiée PCI-DSS). Les identifiants sont pseudonymisés avant envoi au LLM — seuls les indicateurs financiers agrégés sont transmis (taux d'endettement, ratios, scores), jamais les noms, IBAN ou numéros de sécurité sociale. Chiffrement AES-256 au repos et TLS 1.3 en transit.",
      auditLog: "Conformité totale avec les exigences réglementaires bancaires : chaque décision de crédit est tracée avec horodatage, données d'entrée (hashées), score ML, contributions SHAP, décision LLM, justification, résultat final. Possibilité de rejouer une décision à l'identique pour audit. Rétention : durée légale de 5 ans après fin du contrat de crédit.",
      humanInTheLoop: "Les demandes avec un score entre 350 et 450 sont systématiquement renvoyées à un analyste crédit humain pour décision finale. Les refus génèrent automatiquement un courrier d'explication conforme RGPD. Les décisions d'acceptation au-delà de 50 000 EUR nécessitent une validation managériale. Comité de crédit hebdomadaire pour les cas limites.",
      monitoring: "Dashboard réglementaire : distribution des scores par segment, taux d'acceptation/refus, taux de défaut réel vs prédit (matrice de confusion), monitoring de biais (analyse par genre, âge, zone géographique), drift du modèle (PSI - Population Stability Index), alertes si le taux de défaut observé dépasse de plus de 2 points la prédiction, coût API par décision.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (demande de crédit) → Node HTTP Request (API Open Banking DSP2) + Node HTTP Request (Bureau de crédit) → Node Merge (consolidation profil) → Node Code (feature engineering) → Node HTTP Request (API scoring ML + LLM) → Node Switch (décision) → Branch accepté: Node HTTP Request (création offre) + Node Email (notification demandeur) → Branch refusé: Node Email (courrier motivation RGPD) → Branch revue: Node Slack (alerte analyste) → Node PostgreSQL (audit log).",
      nodes: ["Webhook (demande crédit)", "HTTP Request (Open Banking)", "HTTP Request (Bureau crédit)", "Merge (profil)", "Code (features)", "HTTP Request (scoring)", "Switch (décision)", "HTTP Request (offre)", "Email (notification)", "Email (motivation refus)", "Slack (analyste)", "PostgreSQL (audit)"],
      triggerType: "Webhook (soumission formulaire de demande de crédit)",
    },
    estimatedTime: "8-12h",
    difficulty: "Expert",
    sectors: ["Banque", "Fintech", "Assurance", "Crédit à la consommation"],
    metiers: ["Risques Crédit", "Data Science", "Direction des Risques"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Scoring Crédit Automatisé — Guide Complet",
    metaDescription:
      "Automatisez le scoring crédit avec un agent IA combinant ML et LLM. Open banking, données alternatives, décisions explicables et conformes RGPD. Tutoriel pas-à-pas pour banques et fintechs.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-gestion-avis-clients",
    title: "Agent de Gestion des Avis Clients",
    subtitle: "Répondez automatiquement aux avis sur Google, Trustpilot et l'App Store grâce à l'IA",
    problem:
      "Les entreprises reçoivent des centaines d'avis clients sur de multiples plateformes (Google Business, Trustpilot, App Store, G2). Répondre manuellement à chacun est chronophage, et l'absence de réponse nuit à la réputation en ligne. Les réponses tardives ou génériques dégradent la relation client et le référencement local (SEO).",
    value:
      "Un agent IA surveille en temps réel les avis publiés sur toutes les plateformes, analyse le sentiment et le contenu, puis génère une réponse personnalisée, empathique et alignée avec le ton de la marque. Les avis négatifs critiques sont escaladés vers l'équipe concernée avec un plan d'action suggéré.",
    inputs: [
      "Avis clients provenant de Google Business, Trustpilot, App Store, Play Store",
      "Historique des interactions client (CRM)",
      "Charte éditoriale et ton de la marque",
      "Base de connaissances produit/service",
      "Règles d'escalade selon la gravité",
    ],
    outputs: [
      "Réponse personnalisée à l'avis (adaptée au ton de la marque)",
      "Analyse de sentiment (positif, neutre, négatif, critique)",
      "Catégorisation thématique de l'avis (produit, service, livraison, SAV)",
      "Alerte d'escalade pour les avis critiques avec plan d'action",
      "Rapport hebdomadaire de tendances et insights",
    ],
    risks: [
      "Réponse inappropriée ou insensible à un avis émotionnel",
      "Ton robotique détectable par les clients",
      "Réponse erronée sur un problème technique produit",
      "Non-conformité avec les CGU des plateformes d'avis",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de réponse aux avis. Amélioration de 0.3 point de la note moyenne sur 6 mois. Augmentation de 20% du taux de réponse aux avis. Impact positif sur le SEO local (+15% de visibilité).",
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
│  Plateformes│────▶│  Agent LLM   │────▶│  Réponse    │
│  d'avis     │     │  (Analyse +  │     │  publiée    │
│  (API/Scrape)│    │  Génération) │     │  ou escalade│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  KB Marque   │
                    │  + CRM       │
                    └──────────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration des APIs",
        content:
          "Installez les dépendances et configurez les accès aux plateformes d'avis. Vous aurez besoin des APIs Google Business Profile, Trustpilot Business et App Store Connect.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary python-dotenv fastapi httpx google-auth`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/avis_db
GOOGLE_BUSINESS_ACCOUNT_ID=...
GOOGLE_BUSINESS_CREDENTIALS=./credentials.json
TRUSTPILOT_API_KEY=...
TRUSTPILOT_BUSINESS_UNIT_ID=...
APPSTORE_KEY_ID=...
APPSTORE_ISSUER_ID=...
APPSTORE_PRIVATE_KEY_PATH=./AuthKey.p8
SLACK_WEBHOOK_ESCALADE=https://hooks.slack.com/services/...`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte des avis multi-plateformes",
        content:
          "Créez un module de collecte unifié qui interroge les APIs de chaque plateforme et normalise les avis dans un format commun. Un scheduler exécute la collecte toutes les 15 minutes.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import httpx
import os

class AvisClient(BaseModel):
    plateforme: str
    auteur: str
    note: int = Field(ge=1, le=5)
    titre: Optional[str] = None
    contenu: str
    date_publication: datetime
    langue: str = "fr"
    avis_id: str
    url_avis: Optional[str] = None
    reponse_existante: Optional[str] = None

async def collecter_google_reviews(limit: int = 50) -> List[AvisClient]:
    """Collecte les avis Google Business Profile via API"""
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_BUSINESS_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/business.manage"]
    )
    account_id = os.getenv("GOOGLE_BUSINESS_ACCOUNT_ID")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://mybusiness.googleapis.com/v4/accounts/{account_id}/locations/-/reviews",
            headers={"Authorization": f"Bearer {credentials.token}"},
            params={"pageSize": limit}
        )
        data = resp.json()
    avis = []
    for r in data.get("reviews", []):
        avis.append(AvisClient(
            plateforme="google",
            auteur=r["reviewer"]["displayName"],
            note=int(r["starRating"].replace("STAR_", "").replace("_", "")),
            contenu=r.get("comment", ""),
            date_publication=r["createTime"],
            avis_id=r["reviewId"],
            reponse_existante=r.get("reviewReply", {}).get("comment")
        ))
    return avis

async def collecter_trustpilot_reviews(limit: int = 50) -> List[AvisClient]:
    """Collecte les avis Trustpilot via API"""
    api_key = os.getenv("TRUSTPILOT_API_KEY")
    bu_id = os.getenv("TRUSTPILOT_BUSINESS_UNIT_ID")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"https://api.trustpilot.com/v1/business-units/{bu_id}/reviews",
            headers={"apikey": api_key},
            params={"perPage": limit, "orderBy": "createdat.desc"}
        )
        data = resp.json()
    avis = []
    for r in data.get("reviews", []):
        avis.append(AvisClient(
            plateforme="trustpilot",
            auteur=r["consumer"]["displayName"],
            note=r["stars"],
            titre=r.get("title"),
            contenu=r["text"],
            date_publication=r["createdAt"],
            avis_id=r["id"]
        ))
    return avis`,
            filename: "collecte_avis.py",
          },
        ],
      },
      {
        title: "Analyse de sentiment et catégorisation",
        content:
          "L'agent IA analyse chaque avis pour en extraire le sentiment, la thématique principale, le niveau de criticité et les points clés à adresser dans la réponse.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
from collecte_avis import AvisClient
from pydantic import BaseModel, Field
from typing import List

class AnalyseAvis(BaseModel):
    sentiment: str = Field(description="positif, neutre, negatif, critique")
    score_sentiment: float = Field(ge=-1, le=1)
    themes: List[str] = Field(description="Thèmes identifiés")
    points_cles: List[str] = Field(description="Points à adresser")
    niveau_urgence: str = Field(description="faible, moyen, eleve, critique")
    necessite_escalade: bool
    raison_escalade: str = ""

client = anthropic.Anthropic()

def analyser_avis(avis: AvisClient) -> AnalyseAvis:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""Analyse cet avis client et produis un JSON structuré.

AVIS :
- Plateforme : {avis.plateforme}
- Note : {avis.note}/5
- Titre : {avis.titre or "Sans titre"}
- Contenu : {avis.contenu}

Évalue le sentiment, identifie les thèmes (produit, service, livraison, prix, SAV, UX),
les points clés à adresser, le niveau d'urgence, et si une escalade humaine est nécessaire.

Critères d'escalade :
- Note 1/5 avec menace juridique ou médiatique
- Problème de sécurité ou santé mentionné
- Client influenceur (> 50 avis sur la plateforme)
- Accusation de fraude ou tromperie

Retourne un JSON AnalyseAvis."""}]
    )
    result = json.loads(response.content[0].text)
    return AnalyseAvis(**result)`,
            filename: "analyse_avis.py",
          },
        ],
      },
      {
        title: "Génération de réponses personnalisées",
        content:
          "Le moteur de génération crée des réponses empathiques, personnalisées et conformes à la charte éditoriale de la marque. Chaque réponse est adaptée au sentiment, à la plateforme et au contenu de l'avis.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
from collecte_avis import AvisClient
from analyse_avis import AnalyseAvis

client = anthropic.Anthropic()

CHARTE_EDITORIALE = """
Ton : Chaleureux, professionnel, authentique.
Tutoiement : Non, toujours vouvoyer.
Signature : L'équipe [NomMarque].
Règles :
- Toujours remercier pour l'avis
- Personnaliser en reprenant un élément spécifique de l'avis
- Proposer une solution concrète pour les avis négatifs
- Ne jamais être défensif ou argumentatif
- Maximum 150 mots pour Google, 200 pour Trustpilot
"""

def generer_reponse(avis: AvisClient, analyse: AnalyseAvis) -> str:
    max_mots = 150 if avis.plateforme == "google" else 200
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": f"""Rédige une réponse à cet avis client.

CHARTE ÉDITORIALE :
{CHARTE_EDITORIALE}

AVIS :
- Plateforme : {avis.plateforme}
- Auteur : {avis.auteur}
- Note : {avis.note}/5
- Contenu : {avis.contenu}

ANALYSE :
- Sentiment : {analyse.sentiment}
- Thèmes : {', '.join(analyse.themes)}
- Points à adresser : {', '.join(analyse.points_cles)}

CONSIGNES :
- Maximum {max_mots} mots
- Ton empathique et personnalisé
- Si négatif : reconnaître le problème, proposer une solution concrète
- Si positif : renforcer la satisfaction, mentionner un détail spécifique
- Ne jamais inventer de faits ou promesses non vérifiables

Réponse uniquement (pas de guillemets ni préambule) :"""}]
    )
    return response.content[0].text.strip()`,
            filename: "generation_reponse.py",
          },
        ],
      },
      {
        title: "Publication et escalade automatique",
        content:
          "Publiez automatiquement les réponses validées sur chaque plateforme via leurs APIs respectives. Les avis critiques déclenchent une alerte Slack avec le contexte complet et un plan d'action proposé.",
        codeSnippets: [
          {
            language: "python",
            code: `import httpx
import os
from collecte_avis import AvisClient
from analyse_avis import AnalyseAvis

async def publier_reponse_google(avis: AvisClient, reponse: str):
    from google.oauth2 import service_account
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_BUSINESS_CREDENTIALS"),
        scopes=["https://www.googleapis.com/auth/business.manage"]
    )
    account_id = os.getenv("GOOGLE_BUSINESS_ACCOUNT_ID")
    async with httpx.AsyncClient() as client:
        await client.put(
            f"https://mybusiness.googleapis.com/v4/accounts/{account_id}/locations/-/reviews/{avis.avis_id}/reply",
            headers={"Authorization": f"Bearer {credentials.token}"},
            json={"comment": reponse}
        )

async def escalader_avis(avis: AvisClient, analyse: AnalyseAvis, reponse_suggeree: str):
    webhook_url = os.getenv("SLACK_WEBHOOK_ESCALADE")
    message = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"Avis critique - {avis.plateforme.upper()}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Auteur :* {avis.auteur} | *Note :* {'star' * avis.note}/5\\n*Contenu :* {avis.contenu[:300]}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Analyse :* {analyse.sentiment} | *Thèmes :* {', '.join(analyse.themes)}\\n*Raison escalade :* {analyse.raison_escalade}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Réponse suggérée :*\\n{reponse_suggeree}"}},
        ]
    }
    async with httpx.AsyncClient() as client:
        await client.post(webhook_url, json=message)`,
            filename: "publication_avis.py",
          },
        ],
      },
      {
        title: "Pipeline complet et scheduling",
        content:
          "Assemblez le pipeline complet : collecte, analyse, génération, publication et escalade. Un scheduler lance le traitement toutes les 15 minutes pour garantir des réponses rapides.",
        codeSnippets: [
          {
            language: "python",
            code: `import asyncio
from collecte_avis import collecter_google_reviews, collecter_trustpilot_reviews, AvisClient
from analyse_avis import analyser_avis
from generation_reponse import generer_reponse
from publication_avis import publier_reponse_google, escalader_avis
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def traiter_avis(avis: AvisClient):
    if avis.reponse_existante:
        logger.info(f"Avis {avis.avis_id} déjà répondu, skip.")
        return
    analyse = analyser_avis(avis)
    reponse = generer_reponse(avis, analyse)
    if analyse.necessite_escalade:
        await escalader_avis(avis, analyse, reponse)
        logger.warning(f"Avis {avis.avis_id} escaladé - {analyse.raison_escalade}")
    else:
        if avis.plateforme == "google":
            await publier_reponse_google(avis, reponse)
        logger.info(f"Réponse publiée pour avis {avis.avis_id} ({avis.plateforme})")

async def pipeline_avis():
    logger.info("Démarrage du pipeline de gestion des avis...")
    avis_google = await collecter_google_reviews(limit=20)
    avis_trustpilot = await collecter_trustpilot_reviews(limit=20)
    tous_avis = avis_google + avis_trustpilot
    logger.info(f"{len(tous_avis)} avis collectés.")
    for avis in tous_avis:
        await traiter_avis(avis)
    logger.info("Pipeline terminé.")

if __name__ == "__main__":
    asyncio.run(pipeline_avis())`,
            filename: "pipeline_avis.py",
          },
        ],
      },
      {
        title: "Tests et monitoring",
        content:
          "Testez la qualité des réponses générées avec des avis réels anonymisés. Mesurez le taux de satisfaction et mettez en place un dashboard de suivi de la réputation en ligne.",
        codeSnippets: [
          {
            language: "python",
            code: `import pytest
from collecte_avis import AvisClient
from analyse_avis import analyser_avis
from generation_reponse import generer_reponse
from datetime import datetime

def test_avis_positif():
    avis = AvisClient(
        plateforme="google", auteur="Marie L.", note=5,
        contenu="Service exceptionnel ! Livraison rapide et produit conforme. Je recommande vivement.",
        date_publication=datetime.now(), avis_id="test-001"
    )
    analyse = analyser_avis(avis)
    assert analyse.sentiment == "positif"
    assert not analyse.necessite_escalade
    reponse = generer_reponse(avis, analyse)
    assert len(reponse.split()) <= 150
    assert "Marie" in reponse or "merci" in reponse.lower()

def test_avis_negatif_escalade():
    avis = AvisClient(
        plateforme="trustpilot", auteur="Jean P.", note=1,
        contenu="Scandaleux ! Produit défectueux reçu et aucune réponse du SAV depuis 3 semaines. Je vais contacter une association de consommateurs.",
        date_publication=datetime.now(), avis_id="test-002"
    )
    analyse = analyser_avis(avis)
    assert analyse.sentiment in ["negatif", "critique"]
    assert analyse.necessite_escalade`,
            filename: "test_avis.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les noms d'auteurs des avis sont publics, mais les données CRM associées sont pseudonymisées avant envoi au LLM. Aucune donnée personnelle interne (email, téléphone, historique d'achat détaillé) n'est transmise au modèle. Chiffrement AES-256 au repos pour la base de données des avis.",
      auditLog: "Chaque réponse générée est loguée avec : horodatage, avis source, analyse de sentiment, réponse générée, statut de publication, et éventuelle escalade. Rétention des logs pendant 24 mois pour analyse de tendances.",
      humanInTheLoop: "Les avis avec une note de 1/5 ou un sentiment 'critique' sont systématiquement soumis à validation humaine avant publication. Les réponses aux avis mentionnant des problèmes juridiques ou de sécurité ne sont jamais publiées automatiquement.",
      monitoring: "Dashboard temps réel : volume d'avis par plateforme, distribution des sentiments, temps moyen de réponse, taux d'escalade, évolution de la note moyenne, top thématiques négatives. Alertes si la note moyenne chute de plus de 0.2 point sur 7 jours.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (toutes les 15 min) → Node HTTP Request (API Google Business) + Node HTTP Request (API Trustpilot) → Node Merge (consolidation avis) → Node Code (filtrage avis sans réponse) → Node HTTP Request (API LLM analyse sentiment) → Node Switch (escalade ou réponse auto) → Branch auto: Node HTTP Request (API LLM génération réponse) → Node HTTP Request (publication) → Branch escalade: Node Slack (alerte équipe) → Node PostgreSQL (log).",
      nodes: ["Schedule Trigger", "HTTP Request (Google Business)", "HTTP Request (Trustpilot)", "Merge (avis)", "Code (filtrage)", "HTTP Request (analyse LLM)", "Switch (escalade)", "HTTP Request (génération réponse)", "HTTP Request (publication)", "Slack (escalade)", "PostgreSQL (log)"],
      triggerType: "Schedule Trigger (toutes les 15 minutes)",
    },
    estimatedTime: "4-6h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "Retail", "Hôtellerie-Restauration", "SaaS", "Services"],
    metiers: ["Marketing Digital", "Community Management", "Relation Client"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Gestion des Avis Clients — Guide Complet",
    metaDescription:
      "Automatisez la gestion de vos avis clients sur Google, Trustpilot et App Store avec un agent IA. Réponses personnalisées, analyse de sentiment et escalade intelligente.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-analyse-predictive-pannes",
    title: "Agent d'Analyse Prédictive des Pannes IT",
    subtitle: "Anticipez les défaillances de votre infrastructure IT grâce à un agent IA de maintenance prédictive",
    problem:
      "Les pannes d'infrastructure IT (serveurs, réseaux, stockage) surviennent de manière imprévisible, causant des interruptions de service coûteuses. La surveillance traditionnelle basée sur des seuils statiques ne détecte les problèmes qu'une fois qu'ils se produisent. Les équipes IT passent plus de temps à éteindre des incendies qu'à prévenir les incidents. Le coût moyen d'une heure d'indisponibilité dépasse 100 000 EUR pour les entreprises de taille intermédiaire.",
    value:
      "Un agent IA collecte et analyse en continu les métriques d'infrastructure (CPU, RAM, disque, réseau, logs applicatifs), détecte les anomalies et prédit les pannes 2 à 48 heures avant qu'elles ne surviennent. Il génère des alertes contextualisées avec diagnostic probable et recommandations d'action préventive, permettant aux équipes IT d'intervenir proactivement.",
    inputs: [
      "Métriques d'infrastructure en temps réel (Prometheus, Datadog, Zabbix)",
      "Logs système et applicatifs (ELK Stack, Splunk)",
      "Historique des incidents (ServiceNow, Jira Service Management)",
      "Configuration des assets IT (CMDB)",
      "Données de capacité et seuils d'alerte actuels",
    ],
    outputs: [
      "Prédiction de panne avec probabilité et horizon temporel (2h à 48h)",
      "Diagnostic probable de la cause racine",
      "Recommandation d'action préventive détaillée",
      "Score de criticité de l'asset concerné (impact métier)",
      "Rapport hebdomadaire de santé infrastructure avec tendances",
    ],
    risks: [
      "Faux positifs générant de la fatigue d'alerte chez les opérateurs",
      "Faux négatifs manquant une panne critique imminente",
      "Dépendance au LLM pour des décisions opérationnelles sensibles",
      "Modèle entraîné sur des données historiques non représentatives de nouvelles architectures",
    ],
    roiIndicatif:
      "Réduction de 65% des pannes non planifiées. Diminution de 40% du temps moyen de résolution (MTTR). Économie estimée de 500K EUR/an pour une infrastructure de 200 serveurs. Amélioration de la disponibilité de 99.5% à 99.95%.",
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
│  Métriques  │────▶│  Agent IA    │────▶│  Alertes    │
│  Infra      │     │  (Détection  │     │  prédictives│
│  (Prometheus│     │  anomalies + │     │  + actions  │
│  + Logs)    │     │  Prédiction) │     │  préventives│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │ Historique│ │  CMDB    │ │  Modèle  │
       │ incidents │ │  Assets  │ │  ML      │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration des sources de données",
        content:
          "Installez les dépendances et configurez les connexions vers vos sources de métriques (Prometheus) et de logs (Elasticsearch). L'agent a besoin d'un accès en lecture à l'historique d'au moins 3 mois pour entraîner le modèle de détection d'anomalies.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary pandas scikit-learn prophet prometheus-api-client elasticsearch python-dotenv fastapi`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/predictive_db
PROMETHEUS_URL=http://prometheus.internal:9090
ELASTICSEARCH_URL=http://elasticsearch.internal:9200
ELASTICSEARCH_INDEX=syslog-*
SERVICENOW_INSTANCE=https://company.service-now.com
SERVICENOW_USER=...
SERVICENOW_PASSWORD=...
SLACK_WEBHOOK_ALERTS=https://hooks.slack.com/services/...
PAGERDUTY_API_KEY=...
SEUIL_ALERTE_PROBA=0.75`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Collecte et normalisation des métriques",
        content:
          "Créez un module de collecte qui interroge Prometheus pour les métriques infrastructure et Elasticsearch pour les logs système. Les données sont normalisées dans un format commun pour l'analyse par le modèle.",
        codeSnippets: [
          {
            language: "python",
            code: `from prometheus_api_client import PrometheusConnect
from elasticsearch import Elasticsearch
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import pandas as pd
import os

class MetriqueServeur(BaseModel):
    hostname: str
    timestamp: datetime
    cpu_usage_pct: float
    memory_usage_pct: float
    disk_usage_pct: float
    disk_io_read_mbps: float
    disk_io_write_mbps: float
    network_in_mbps: float
    network_out_mbps: float
    load_average_5m: float
    nb_erreurs_log_1h: int = 0
    nb_warnings_log_1h: int = 0
    latence_reseau_ms: float = 0
    nb_connexions_actives: int = 0

prom = PrometheusConnect(url=os.getenv("PROMETHEUS_URL"), disable_ssl=True)
es = Elasticsearch(os.getenv("ELASTICSEARCH_URL"))

def collecter_metriques_serveur(hostname: str, heures: int = 24) -> List[MetriqueServeur]:
    """Collecte les métriques Prometheus d'un serveur sur N heures"""
    fin = datetime.now()
    debut = fin - timedelta(hours=heures)

    queries = {
        "cpu_usage_pct": f'100 - (avg(rate(node_cpu_seconds_total{{mode="idle",instance="{hostname}"}}[5m])) * 100)',
        "memory_usage_pct": f'(1 - node_memory_MemAvailable_bytes{{instance="{hostname}"}} / node_memory_MemTotal_bytes{{instance="{hostname}"}}) * 100',
        "disk_usage_pct": f'(1 - node_filesystem_avail_bytes{{instance="{hostname}",mountpoint="/"}} / node_filesystem_size_bytes{{instance="{hostname}",mountpoint="/"}}) * 100',
        "load_average_5m": f'node_load5{{instance="{hostname}"}}',
    }
    metriques = []
    resultats = {}
    for nom, query in queries.items():
        data = prom.custom_query_range(query, start_time=debut, end_time=fin, step="5m")
        if data:
            resultats[nom] = {datetime.fromtimestamp(v[0]): float(v[1]) for v in data[0]["values"]}

    # Compter les erreurs dans les logs
    log_errors = es.count(
        index=os.getenv("ELASTICSEARCH_INDEX"),
        body={"query": {"bool": {"must": [
            {"match": {"host.name": hostname}},
            {"match": {"log.level": "error"}},
            {"range": {"@timestamp": {"gte": f"now-{heures}h"}}}
        ]}}}
    )["count"]

    timestamps = sorted(resultats.get("cpu_usage_pct", {}).keys())
    for ts in timestamps:
        metriques.append(MetriqueServeur(
            hostname=hostname,
            timestamp=ts,
            cpu_usage_pct=resultats.get("cpu_usage_pct", {}).get(ts, 0),
            memory_usage_pct=resultats.get("memory_usage_pct", {}).get(ts, 0),
            disk_usage_pct=resultats.get("disk_usage_pct", {}).get(ts, 0),
            disk_io_read_mbps=0, disk_io_write_mbps=0,
            network_in_mbps=0, network_out_mbps=0,
            load_average_5m=resultats.get("load_average_5m", {}).get(ts, 0),
            nb_erreurs_log_1h=log_errors // max(heures, 1)
        ))
    return metriques`,
            filename: "collecte_metriques.py",
          },
        ],
      },
      {
        title: "Modèle de détection d'anomalies et prédiction",
        content:
          "Entraînez un modèle de détection d'anomalies (Isolation Forest) sur l'historique des métriques, couplé à Prophet pour la prédiction de tendances. Les anomalies détectées sont enrichies par le LLM pour produire un diagnostic compréhensible.",
        codeSnippets: [
          {
            language: "python",
            code: `from sklearn.ensemble import IsolationForest
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle
from typing import Dict, Tuple

class PredicteurPannes:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=42
        )
        self.prophets: Dict[str, Prophet] = {}
        self.est_entraine = False

    def entrainer(self, df_historique: pd.DataFrame):
        """Entraîne le modèle sur l'historique des métriques"""
        features = ["cpu_usage_pct", "memory_usage_pct", "disk_usage_pct",
                    "load_average_5m", "nb_erreurs_log_1h"]
        X = df_historique[features].fillna(0)
        self.isolation_forest.fit(X)

        # Entraîner Prophet pour chaque métrique
        for col in features:
            prophet_df = df_historique[["timestamp", col]].rename(
                columns={"timestamp": "ds", col: "y"}
            )
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode="multiplicative"
            )
            model.fit(prophet_df)
            self.prophets[col] = model
        self.est_entraine = True
        pickle.dump(self, open("models/predicteur_pannes.pkl", "wb"))

    def detecter_anomalies(self, df_recent: pd.DataFrame) -> pd.DataFrame:
        """Détecte les anomalies dans les métriques récentes"""
        features = ["cpu_usage_pct", "memory_usage_pct", "disk_usage_pct",
                    "load_average_5m", "nb_erreurs_log_1h"]
        X = df_recent[features].fillna(0)
        scores = self.isolation_forest.decision_function(X)
        predictions = self.isolation_forest.predict(X)
        df_recent["anomaly_score"] = scores
        df_recent["is_anomaly"] = predictions == -1
        return df_recent[df_recent["is_anomaly"]]

    def predire_tendances(self, horizon_heures: int = 48) -> Dict[str, pd.DataFrame]:
        """Prédit l'évolution des métriques sur l'horizon donné"""
        predictions = {}
        for metrique, model in self.prophets.items():
            future = model.make_future_dataframe(periods=horizon_heures * 12, freq="5min")
            forecast = model.predict(future)
            predictions[metrique] = forecast[["ds", "yhat", "yhat_upper"]].tail(horizon_heures * 12)
        return predictions`,
            filename: "predicteur_pannes.py",
          },
        ],
      },
      {
        title: "Agent LLM de diagnostic et recommandation",
        content:
          "L'agent LLM reçoit les anomalies détectées et les prédictions, puis génère un diagnostic en langage naturel avec des recommandations d'action concrètes pour l'équipe IT.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
from pydantic import BaseModel, Field
from typing import List, Optional

class DiagnosticPanne(BaseModel):
    hostname: str
    probabilite_panne: float = Field(ge=0, le=1)
    horizon_estime: str = Field(description="Estimation temporelle avant la panne")
    cause_probable: str
    composant_concerne: str = Field(description="CPU, RAM, Disque, Réseau, Application")
    impact_metier: str = Field(description="critique, eleve, moyen, faible")
    recommandations: List[str]
    actions_immediates: List[str]
    metriques_cles: dict

client = anthropic.Anthropic()

def diagnostiquer(hostname: str, anomalies: dict, predictions: dict,
                   historique_incidents: list) -> DiagnosticPanne:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"""Tu es un ingénieur SRE senior spécialisé en maintenance prédictive IT.
Analyse ces données et produis un diagnostic structuré.

SERVEUR : {hostname}

ANOMALIES DÉTECTÉES :
{json.dumps(anomalies, indent=2, default=str)}

PRÉDICTIONS (48 prochaines heures) :
{json.dumps(predictions, indent=2, default=str)}

HISTORIQUE DES INCIDENTS SUR CE SERVEUR :
{json.dumps(historique_incidents, indent=2, default=str)}

Analyse les patterns, corrèle avec l'historique, et produis :
1. La probabilité d'une panne (0 à 1)
2. L'horizon temporel estimé
3. La cause racine probable
4. L'impact métier potentiel
5. Les recommandations d'action préventive
6. Les actions immédiates à entreprendre

Retourne un JSON DiagnosticPanne."""}]
    )
    result = json.loads(response.content[0].text)
    result["hostname"] = hostname
    return DiagnosticPanne(**result)`,
            filename: "agent_diagnostic.py",
          },
        ],
      },
      {
        title: "API et système d'alertes",
        content:
          "Déployez l'API de prédiction avec un système d'alertes multi-canal (Slack, PagerDuty, email). Les alertes sont enrichies avec le diagnostic complet et les recommandations d'action.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from agent_diagnostic import diagnostiquer, DiagnosticPanne
from collecte_metriques import collecter_metriques_serveur
from predicteur_pannes import PredicteurPannes
import httpx
import os
import pickle

app = FastAPI(title="API Maintenance Prédictive IT")
predicteur = pickle.load(open("models/predicteur_pannes.pkl", "rb"))

@app.get("/api/prediction/{hostname}")
async def predire_panne(hostname: str):
    metriques = collecter_metriques_serveur(hostname, heures=24)
    import pandas as pd
    df = pd.DataFrame([m.model_dump() for m in metriques])
    anomalies = predicteur.detecter_anomalies(df)
    predictions = predicteur.predire_tendances(horizon_heures=48)
    if not anomalies.empty:
        diagnostic = diagnostiquer(
            hostname=hostname,
            anomalies=anomalies.to_dict(orient="records"),
            predictions={k: v.to_dict(orient="records") for k, v in predictions.items()},
            historique_incidents=[]
        )
        if diagnostic.probabilite_panne >= float(os.getenv("SEUIL_ALERTE_PROBA", 0.75)):
            await envoyer_alerte(diagnostic)
        return diagnostic.model_dump()
    return {"hostname": hostname, "status": "nominal", "anomalies": 0}

async def envoyer_alerte(diagnostic: DiagnosticPanne):
    webhook = os.getenv("SLACK_WEBHOOK_ALERTS")
    emoji = {"critique": "🔴", "eleve": "🟠", "moyen": "🟡", "faible": "🟢"}
    message = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"Alerte prédictive - {diagnostic.hostname}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Probabilité :* {diagnostic.probabilite_panne:.0%} | *Horizon :* {diagnostic.horizon_estime}\\n*Cause :* {diagnostic.cause_probable}\\n*Impact :* {emoji.get(diagnostic.impact_metier, '')} {diagnostic.impact_metier}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Actions immédiates :*\\n" + "\\n".join(f"• {a}" for a in diagnostic.actions_immediates)}},
        ]
    }
    async with httpx.AsyncClient() as client:
        await client.post(webhook, json=message)`,
            filename: "api_predictive.py",
          },
        ],
      },
      {
        title: "Pipeline de surveillance continue",
        content:
          "Mettez en place le pipeline complet de surveillance continue. Un scheduler analyse chaque serveur périodiquement, stocke les résultats et alimente un dashboard de santé infrastructure.",
        codeSnippets: [
          {
            language: "python",
            code: `import asyncio
import logging
from datetime import datetime
from collecte_metriques import collecter_metriques_serveur
from predicteur_pannes import PredicteurPannes
from agent_diagnostic import diagnostiquer
import pandas as pd
import pickle
import psycopg2
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVEURS = [
    "web-prod-01", "web-prod-02", "api-prod-01",
    "db-master-01", "db-replica-01", "cache-prod-01"
]

predicteur = pickle.load(open("models/predicteur_pannes.pkl", "rb"))

async def surveiller_infrastructure():
    logger.info(f"Scan infrastructure - {datetime.now().isoformat()}")
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    for hostname in SERVEURS:
        try:
            metriques = collecter_metriques_serveur(hostname, heures=6)
            df = pd.DataFrame([m.model_dump() for m in metriques])
            anomalies = predicteur.detecter_anomalies(df)
            status = "anomalie" if not anomalies.empty else "nominal"
            cur.execute(
                "INSERT INTO health_checks (hostname, timestamp, status, nb_anomalies, metriques) VALUES (%s, %s, %s, %s, %s)",
                (hostname, datetime.now(), status, len(anomalies), df.describe().to_json())
            )
            if not anomalies.empty:
                logger.warning(f"{hostname}: {len(anomalies)} anomalies détectées")
        except Exception as e:
            logger.error(f"Erreur surveillance {hostname}: {e}")
    conn.commit()
    conn.close()

if __name__ == "__main__":
    asyncio.run(surveiller_infrastructure())`,
            filename: "surveillance.py",
          },
        ],
      },
      {
        title: "Tests et calibration du modèle",
        content:
          "Testez le système avec des données historiques de pannes réelles pour calibrer les seuils d'alerte et minimiser les faux positifs. Mesurez la précision prédictive sur les 3 derniers mois.",
        codeSnippets: [
          {
            language: "python",
            code: `import pytest
import pandas as pd
import numpy as np
from predicteur_pannes import PredicteurPannes
from datetime import datetime, timedelta

def test_detection_anomalie_cpu():
    predicteur = PredicteurPannes()
    # Générer des données normales
    dates = pd.date_range(end=datetime.now(), periods=1000, freq="5min")
    df_normal = pd.DataFrame({
        "timestamp": dates,
        "cpu_usage_pct": np.random.normal(45, 10, 1000).clip(0, 100),
        "memory_usage_pct": np.random.normal(60, 8, 1000).clip(0, 100),
        "disk_usage_pct": np.random.normal(55, 5, 1000).clip(0, 100),
        "load_average_5m": np.random.normal(2, 0.5, 1000).clip(0, 20),
        "nb_erreurs_log_1h": np.random.poisson(2, 1000),
    })
    predicteur.entrainer(df_normal)
    # Injecter une anomalie (CPU spike)
    df_anomalie = df_normal.tail(10).copy()
    df_anomalie["cpu_usage_pct"] = 98.5
    df_anomalie["nb_erreurs_log_1h"] = 150
    anomalies = predicteur.detecter_anomalies(df_anomalie)
    assert len(anomalies) > 0, "L'anomalie CPU devrait être détectée"

def test_pas_de_faux_positif_normal():
    predicteur = PredicteurPannes()
    dates = pd.date_range(end=datetime.now(), periods=1000, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "cpu_usage_pct": np.random.normal(45, 10, 1000).clip(0, 100),
        "memory_usage_pct": np.random.normal(60, 8, 1000).clip(0, 100),
        "disk_usage_pct": np.random.normal(55, 5, 1000).clip(0, 100),
        "load_average_5m": np.random.normal(2, 0.5, 1000).clip(0, 20),
        "nb_erreurs_log_1h": np.random.poisson(2, 1000),
    })
    predicteur.entrainer(df)
    anomalies = predicteur.detecter_anomalies(df.tail(50))
    taux_faux_positif = len(anomalies) / 50
    assert taux_faux_positif < 0.1, f"Taux de faux positifs trop élevé: {taux_faux_positif:.0%}"`,
            filename: "test_predicteur.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Aucune donnée personnelle n'est transmise au LLM. Seules les métriques techniques agrégées (CPU, RAM, disque, réseau) et les identifiants de serveurs (hostnames) sont envoyés. Les logs système sont filtrés pour retirer toute information sensible (IP internes, credentials) avant analyse.",
      auditLog: "Chaque prédiction est loguée avec : horodatage, serveur concerné, probabilité de panne, diagnostic, recommandations, et résultat réel (panne survenue ou non) pour le réentraînement du modèle. Rétention de 12 mois pour analyse de performance du modèle.",
      humanInTheLoop: "Les alertes avec une probabilité de panne supérieure à 90% et un impact critique déclenchent un appel PagerDuty obligatoire. Aucune action automatique n'est exécutée sur l'infrastructure sans validation humaine. Les recommandations sont consultatives uniquement.",
      monitoring: "Dashboard Grafana : précision du modèle (vrais positifs vs faux positifs), taux de pannes prédites vs non prédites, MTTR avant et après déploiement, coût API LLM par diagnostic, nombre d'alertes par jour et par criticité, tendance de santé globale de l'infrastructure.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Schedule Trigger (toutes les 30 min) → Node HTTP Request (Prometheus API métriques) + Node HTTP Request (Elasticsearch logs) → Node Code (normalisation données) → Node HTTP Request (API modèle ML anomalies) → Node IF (anomalie détectée ?) → Node HTTP Request (API LLM diagnostic) → Node Switch (criticité) → Branch critique: Node PagerDuty (alerte on-call) → Branch élevée: Node Slack (canal ops) → Node PostgreSQL (log prédiction).",
      nodes: ["Schedule Trigger", "HTTP Request (Prometheus)", "HTTP Request (Elasticsearch)", "Code (normalisation)", "HTTP Request (ML anomalies)", "IF (anomalie ?)", "HTTP Request (LLM diagnostic)", "Switch (criticité)", "PagerDuty (alerte)", "Slack (ops)", "PostgreSQL (log)"],
      triggerType: "Schedule Trigger (toutes les 30 minutes)",
    },
    estimatedTime: "8-12h",
    difficulty: "Expert",
    sectors: ["Technologie", "Finance", "Telecom", "E-commerce", "Industrie"],
    metiers: ["SRE", "Infrastructure", "DevOps", "IT Operations"],
    functions: ["IT"],
    metaTitle: "Agent IA d'Analyse Prédictive des Pannes IT — Guide Complet",
    metaDescription:
      "Anticipez les pannes d'infrastructure IT avec un agent IA de maintenance prédictive. Détection d'anomalies, diagnostic automatisé et alertes proactives. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-planification-rendez-vous",
    title: "Agent de Planification de Rendez-vous Commercial",
    subtitle: "Automatisez la prise de rendez-vous et la qualification des prospects avec un agent IA conversationnel",
    problem:
      "La prise de rendez-vous commerciaux est un processus inefficace : les SDR passent 60% de leur temps sur des tâches administratives (relances email, coordination d'agendas, qualification initiale) au lieu de vendre. Les délais de réponse aux demandes entrantes dépassent souvent 24h, entraînant la perte de prospects chauds. La coordination des agendas entre prospects et commerciaux génère des allers-retours interminables.",
    value:
      "Un agent IA conversationnel gère l'intégralité du processus de prise de rendez-vous : il qualifie le prospect via un échange naturel (email ou chat), identifie le bon interlocuteur commercial selon le profil, propose des créneaux disponibles, et confirme le rendez-vous avec rappels automatiques. Le temps de réponse passe de 24h à moins de 2 minutes.",
    inputs: [
      "Demande entrante du prospect (formulaire, email, chatbot)",
      "Données CRM du prospect (si existant)",
      "Calendriers des commerciaux (Google Calendar, Outlook)",
      "Critères de qualification (BANT, secteur, taille d'entreprise)",
      "Règles d'attribution par territoire, secteur ou taille de deal",
    ],
    outputs: [
      "Score de qualification du prospect (0-100)",
      "Profil BANT complété (Budget, Authority, Need, Timeline)",
      "Commercial attribué avec justification",
      "Rendez-vous confirmé avec invitation calendrier",
      "Fiche de briefing commercial avec contexte du prospect",
    ],
    risks: [
      "Qualification trop agressive repoussant des prospects à fort potentiel",
      "Mauvaise attribution du commercial (territoire, expertise)",
      "Créneaux proposés non adaptés au fuseau horaire du prospect",
      "Ton trop robotique dans les échanges dégradant l'image de marque",
    ],
    roiIndicatif:
      "Réduction de 75% du temps administratif des SDR. Augmentation de 40% du taux de prise de rendez-vous. Diminution du délai de réponse de 24h à 2 minutes. Amélioration de 25% du taux de show-up grâce aux rappels automatiques.",
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
│  Prospect   │────▶│  Agent IA    │────▶│  Rendez-vous│
│  (email,    │     │  (Qualif. +  │     │  confirmé + │
│  chat, form)│     │  Planning)   │     │  briefing   │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │  CRM      │ │ Calendar │ │  Règles  │
       │(HubSpot)  │ │ (Google) │ │  attrib. │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration des intégrations",
        content:
          "Installez les dépendances et configurez les connexions vers votre CRM (HubSpot), votre calendrier (Google Calendar) et votre messagerie. L'agent a besoin d'un accès en lecture/écriture au calendrier et en lecture au CRM.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install openai langchain psycopg2-binary python-dotenv fastapi google-auth google-api-python-client hubspot-api-client python-dateutil pytz`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/rdv_db
HUBSPOT_API_KEY=pat-...
GOOGLE_CALENDAR_CREDENTIALS=./calendar_credentials.json
GOOGLE_CALENDAR_IDS=commercial1@company.com,commercial2@company.com
SMTP_HOST=smtp.company.com
SMTP_PORT=587
SMTP_USER=agent-rdv@company.com
SMTP_PASSWORD=...
SLACK_WEBHOOK_SALES=https://hooks.slack.com/services/...
FUSEAU_HORAIRE_DEFAUT=Europe/Paris`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Module de qualification conversationnelle",
        content:
          "Créez l'agent conversationnel qui qualifie le prospect via un échange naturel. L'agent pose des questions pertinentes sans être intrusif et construit progressivement le profil BANT du prospect.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class ProfilBANT(BaseModel):
    budget_estime: Optional[str] = None
    est_decideur: Optional[bool] = None
    role_dans_decision: Optional[str] = None
    besoin_principal: Optional[str] = None
    problematique_identifiee: Optional[str] = None
    timeline: Optional[str] = None
    taille_entreprise: Optional[str] = None
    secteur: Optional[str] = None
    score_qualification: int = Field(default=0, ge=0, le=100)
    questions_posees: List[str] = Field(default_factory=list)
    informations_manquantes: List[str] = Field(default_factory=list)

client = OpenAI()

SYSTEM_PROMPT = """Tu es un assistant commercial pour [NomEntreprise].
Ton rôle est de qualifier les prospects de manière naturelle et bienveillante.

OBJECTIF : Obtenir les informations BANT sans interrogatoire.
- Budget : Fourchette budgétaire ou contraintes financières
- Authority : Rôle dans la décision d'achat
- Need : Besoin principal et problématique
- Timeline : Urgence et calendrier du projet

RÈGLES :
- Sois naturel, empathique, jamais intrusif
- Pose maximum 2 questions par message
- Adapte tes questions aux réponses précédentes
- Si le prospect semble pressé, propose directement un créneau
- Vouvoie toujours le prospect
- Ne révèle jamais que tu es une IA sauf si on te le demande directement

Quand tu as suffisamment d'informations (score >= 60), propose un rendez-vous."""

def qualifier_prospect(historique_conversation: list, contexte_crm: dict = None) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if contexte_crm:
        messages.append({"role": "system", "content": f"Contexte CRM du prospect : {json.dumps(contexte_crm)}"})
    messages.extend(historique_conversation)
    messages.append({"role": "system", "content": "Après ta réponse, fournis aussi un JSON ProfilBANT mis à jour dans un bloc json."})

    response = client.chat.completions.create(
        model="gpt-4.1", temperature=0.7, max_tokens=1024, messages=messages
    )
    contenu = response.choices[0].message.content
    # Extraire la réponse et le profil BANT
    if "---json" in contenu:
        parts = contenu.split("---json")
        reponse_prospect = parts[0].strip()
        profil_json = parts[1].split("---")[0].strip()
        profil = ProfilBANT(**json.loads(profil_json))
    else:
        reponse_prospect = contenu
        profil = ProfilBANT()
    return {"reponse": reponse_prospect, "profil": profil.model_dump()}`,
            filename: "qualification.py",
          },
        ],
      },
      {
        title: "Gestion des calendriers et disponibilités",
        content:
          "Intégrez Google Calendar pour récupérer les disponibilités des commerciaux et proposer des créneaux adaptés au prospect. Le module gère les fuseaux horaires et les préférences de chaque commercial.",
        codeSnippets: [
          {
            language: "python",
            code: `from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from typing import List, Dict
import pytz
import os

SCOPES = ["https://www.googleapis.com/auth/calendar"]

def get_calendar_service():
    credentials = service_account.Credentials.from_service_account_file(
        os.getenv("GOOGLE_CALENDAR_CREDENTIALS"), scopes=SCOPES
    )
    return build("calendar", "v3", credentials=credentials)

def obtenir_disponibilites(
    calendar_id: str,
    jours_ahead: int = 5,
    duree_rdv_min: int = 30,
    fuseau: str = "Europe/Paris"
) -> List[Dict]:
    service = get_calendar_service()
    tz = pytz.timezone(fuseau)
    now = datetime.now(tz)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=jours_ahead)).isoformat()

    events = service.events().list(
        calendarId=calendar_id,
        timeMin=time_min, timeMax=time_max,
        singleEvents=True, orderBy="startTime"
    ).execute().get("items", [])

    # Calculer les créneaux libres (9h-18h, lundi-vendredi)
    creneaux_libres = []
    for jour in range(jours_ahead):
        date = now.date() + timedelta(days=jour + 1)
        if date.weekday() >= 5:
            continue
        debut_journee = tz.localize(datetime.combine(date, datetime.strptime("09:00", "%H:%M").time()))
        fin_journee = tz.localize(datetime.combine(date, datetime.strptime("18:00", "%H:%M").time()))

        occupations = [
            (datetime.fromisoformat(e["start"]["dateTime"]),
             datetime.fromisoformat(e["end"]["dateTime"]))
            for e in events
            if datetime.fromisoformat(e["start"]["dateTime"]).date() == date
        ]
        occupations.sort()

        cursor = debut_journee
        for start_occ, end_occ in occupations:
            if (start_occ - cursor).total_seconds() >= duree_rdv_min * 60:
                creneaux_libres.append({
                    "debut": cursor.isoformat(),
                    "fin": start_occ.isoformat(),
                    "date_lisible": cursor.strftime("%A %d %B à %Hh%M")
                })
            cursor = max(cursor, end_occ)
        if (fin_journee - cursor).total_seconds() >= duree_rdv_min * 60:
            creneaux_libres.append({
                "debut": cursor.isoformat(),
                "fin": fin_journee.isoformat(),
                "date_lisible": cursor.strftime("%A %d %B à %Hh%M")
            })
    return creneaux_libres[:10]

def creer_evenement(calendar_id: str, titre: str, debut: str,
                     duree_min: int, email_prospect: str, notes: str):
    service = get_calendar_service()
    event = {
        "summary": titre,
        "description": notes,
        "start": {"dateTime": debut, "timeZone": "Europe/Paris"},
        "end": {"dateTime": (datetime.fromisoformat(debut) + timedelta(minutes=duree_min)).isoformat(), "timeZone": "Europe/Paris"},
        "attendees": [{"email": email_prospect}, {"email": calendar_id}],
        "conferenceData": {"createRequest": {"requestId": f"rdv-{datetime.now().timestamp()}"}},
        "reminders": {"useDefault": False, "overrides": [
            {"method": "email", "minutes": 60},
            {"method": "popup", "minutes": 15}
        ]}
    }
    return service.events().insert(
        calendarId=calendar_id, body=event,
        conferenceDataVersion=1, sendUpdates="all"
    ).execute()`,
            filename: "calendrier.py",
          },
        ],
      },
      {
        title: "Attribution intelligente des commerciaux",
        content:
          "Le module d'attribution sélectionne le commercial le plus adapté au prospect en fonction du territoire, du secteur d'activité, de la taille du deal et de la charge de travail actuelle de chaque commercial.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import List, Optional
import json
from openai import OpenAI

class Commercial(BaseModel):
    nom: str
    email: str
    calendar_id: str
    territoires: List[str]
    secteurs_expertise: List[str]
    taille_deals: str = Field(description="PME, ETI, GrandCompte")
    charge_actuelle: int = Field(description="Nombre de deals en cours")
    taux_conversion_30j: float = 0
    langues: List[str] = Field(default_factory=lambda: ["fr"])

EQUIPE_COMMERCIALE = [
    Commercial(nom="Sophie Martin", email="sophie@company.com", calendar_id="sophie@company.com",
               territoires=["IDF", "Nord"], secteurs_expertise=["Tech", "SaaS"], taille_deals="ETI",
               charge_actuelle=12, taux_conversion_30j=0.32, langues=["fr", "en"]),
    Commercial(nom="Thomas Dubois", email="thomas@company.com", calendar_id="thomas@company.com",
               territoires=["Sud", "Ouest"], secteurs_expertise=["Industrie", "Retail"], taille_deals="PME",
               charge_actuelle=18, taux_conversion_30j=0.28, langues=["fr"]),
    Commercial(nom="Camille Laurent", email="camille@company.com", calendar_id="camille@company.com",
               territoires=["IDF", "International"], secteurs_expertise=["Finance", "Assurance"], taille_deals="GrandCompte",
               charge_actuelle=6, taux_conversion_30j=0.45, langues=["fr", "en", "de"]),
]

client = OpenAI()

def attribuer_commercial(profil_bant: dict) -> Commercial:
    response = client.chat.completions.create(
        model="gpt-4.1", temperature=0, max_tokens=512,
        messages=[
            {"role": "system", "content": f"""Sélectionne le commercial le plus adapté pour ce prospect.
Équipe : {json.dumps([c.model_dump() for c in EQUIPE_COMMERCIALE], indent=2)}
Critères de sélection (par ordre de priorité) :
1. Territoire géographique compatible
2. Expertise sectorielle
3. Taille de deal appropriée
4. Charge de travail la plus faible
5. Meilleur taux de conversion
Retourne le nom du commercial sélectionné et la justification en JSON."""},
            {"role": "user", "content": f"Profil prospect : {json.dumps(profil_bant)}"}
        ]
    )
    result = json.loads(response.choices[0].message.content)
    return next(c for c in EQUIPE_COMMERCIALE if c.nom == result["commercial"])`,
            filename: "attribution.py",
          },
        ],
      },
      {
        title: "API et pipeline de bout en bout",
        content:
          "Assemblez le pipeline complet : réception de la demande, qualification conversationnelle, attribution du commercial, proposition de créneaux, et confirmation du rendez-vous avec briefing automatique.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from qualification import qualifier_prospect, ProfilBANT
from attribution import attribuer_commercial
from calendrier import obtenir_disponibilites, creer_evenement
import json

app = FastAPI(title="API Planification RDV Commercial")

class MessageProspect(BaseModel):
    session_id: str
    message: str
    email_prospect: Optional[str] = None
    source: str = "chatbot"

# Stockage en mémoire des sessions (utiliser Redis en production)
sessions = {}

@app.post("/api/rdv/conversation")
async def converser(msg: MessageProspect):
    if msg.session_id not in sessions:
        sessions[msg.session_id] = {"historique": [], "profil": {}, "etape": "qualification"}
    session = sessions[msg.session_id]
    session["historique"].append({"role": "user", "content": msg.message})

    if session["etape"] == "qualification":
        resultat = qualifier_prospect(session["historique"])
        session["profil"] = resultat["profil"]
        session["historique"].append({"role": "assistant", "content": resultat["reponse"]})

        if resultat["profil"].get("score_qualification", 0) >= 60:
            commercial = attribuer_commercial(resultat["profil"])
            session["commercial"] = commercial.model_dump()
            creneaux = obtenir_disponibilites(commercial.calendar_id)
            session["creneaux"] = creneaux
            session["etape"] = "proposition_creneau"
            creneaux_texte = "\\n".join([f"- {c['date_lisible']}" for c in creneaux[:5]])
            reponse_finale = f"{resultat['reponse']}\\n\\nJe vous propose un échange avec {commercial.nom}, spécialiste de votre secteur. Voici les prochains créneaux disponibles :\\n{creneaux_texte}\\n\\nQuel créneau vous conviendrait le mieux ?"
            session["historique"][-1]["content"] = reponse_finale
            return {"reponse": reponse_finale, "etape": "proposition_creneau", "creneaux": creneaux[:5]}

        return {"reponse": resultat["reponse"], "etape": "qualification", "score": resultat["profil"].get("score_qualification", 0)}

    elif session["etape"] == "proposition_creneau":
        # Confirmer le créneau choisi
        commercial = session["commercial"]
        creneau_choisi = session["creneaux"][0]  # Simplifié: prendre le premier
        briefing = f"Prospect qualifié via chatbot. Profil BANT : {json.dumps(session['profil'], indent=2)}"
        event = creer_evenement(
            calendar_id=commercial["calendar_id"],
            titre=f"RDV Commercial - {msg.email_prospect or 'Prospect'}",
            debut=creneau_choisi["debut"],
            duree_min=30,
            email_prospect=msg.email_prospect or "",
            notes=briefing
        )
        session["etape"] = "confirme"
        return {
            "reponse": f"Parfait ! Votre rendez-vous avec {commercial['nom']} est confirmé pour le {creneau_choisi['date_lisible']}. Vous recevrez une invitation par email avec le lien de visioconférence. A bientôt !",
            "etape": "confirme",
            "event_id": event.get("id")
        }`,
            filename: "api_rdv.py",
          },
        ],
      },
      {
        title: "Tests et métriques de performance",
        content:
          "Testez le pipeline complet avec des scénarios de prospects variés. Mesurez le taux de conversion, le temps moyen de qualification et la satisfaction des prospects.",
        codeSnippets: [
          {
            language: "python",
            code: `import pytest
from qualification import qualifier_prospect, ProfilBANT

def test_qualification_prospect_chaud():
    historique = [
        {"role": "user", "content": "Bonjour, je suis directeur commercial chez TechCorp (150 employés). Nous cherchons une solution de CRM IA pour Q2, budget autour de 50K EUR."},
    ]
    resultat = qualifier_prospect(historique)
    profil = resultat["profil"]
    assert profil["score_qualification"] >= 60, "Un prospect avec budget, timeline et autorité devrait scorer haut"
    assert profil["est_decideur"] is True
    assert profil["budget_estime"] is not None

def test_qualification_prospect_froid():
    historique = [
        {"role": "user", "content": "Bonjour, je voulais juste des informations sur vos tarifs."},
    ]
    resultat = qualifier_prospect(historique)
    profil = resultat["profil"]
    assert profil["score_qualification"] < 60, "Un prospect sans info BANT devrait scorer bas"
    assert len(profil["informations_manquantes"]) > 0

def test_reponse_naturelle():
    historique = [
        {"role": "user", "content": "Salut, je suis intéressé par votre offre."},
    ]
    resultat = qualifier_prospect(historique)
    reponse = resultat["reponse"]
    assert len(reponse) > 20, "La réponse doit être substantielle"
    assert "vous" in reponse.lower(), "L'agent doit vouvoyer"`,
            filename: "test_rdv.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données prospects (nom, email, entreprise) sont nécessaires au processus mais ne sont jamais stockées dans les logs LLM. Les conversations sont pseudonymisées avant archivage. Les données CRM sont accédées en lecture seule via API sécurisée avec token à durée limitée.",
      auditLog: "Chaque session de qualification est loguée : horodatage, source du prospect, score de qualification, commercial attribué, créneau proposé, résultat (RDV confirmé, abandonné, escaladé). Traçabilité complète pour analyse du funnel de conversion.",
      humanInTheLoop: "Les prospects stratégiques (entreprises > 500 employés ou deal > 100K EUR) sont automatiquement escaladés vers un manager commercial. Les conversations où le prospect exprime une insatisfaction sont transférées à un humain en temps réel.",
      monitoring: "Dashboard commercial : taux de qualification, taux de prise de RDV, délai moyen de réponse, taux de show-up, NPS post-interaction, répartition des attributions par commercial, coût par lead qualifié.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouveau lead formulaire/chat) → Node HTTP Request (API CRM HubSpot enrichissement) → Node HTTP Request (API LLM qualification) → Node IF (score >= 60 ?) → Node HTTP Request (Google Calendar disponibilités) → Node HTTP Request (API LLM choix commercial) → Node Google Calendar (création événement) → Node Email (confirmation prospect) → Node Slack (notification commercial) → Node HubSpot (mise à jour deal).",
      nodes: ["Webhook (nouveau lead)", "HTTP Request (HubSpot)", "HTTP Request (LLM qualification)", "IF (score qualif)", "HTTP Request (Calendar)", "HTTP Request (attribution)", "Google Calendar (RDV)", "Email (confirmation)", "Slack (notification)", "HubSpot (update deal)"],
      triggerType: "Webhook (soumission formulaire ou message chatbot)",
    },
    estimatedTime: "6-8h",
    difficulty: "Moyen",
    sectors: ["SaaS", "Services B2B", "Conseil", "Technologie", "Immobilier"],
    metiers: ["Sales Development", "Inside Sales", "Business Development"],
    functions: ["Sales"],
    metaTitle: "Agent IA de Planification de Rendez-vous Commercial — Guide Complet",
    metaDescription:
      "Automatisez la prise de rendez-vous et la qualification de prospects avec un agent IA conversationnel. Intégration CRM, Calendar et scoring BANT. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-traduction-localisation",
    title: "Agent de Traduction et Localisation de Contenu",
    subtitle: "Localisez automatiquement vos contenus marketing pour les marchés internationaux grâce à l'IA",
    problem:
      "Les entreprises françaises qui s'internationalisent font face à un défi majeur : traduire et localiser des volumes importants de contenu (site web, emails marketing, fiches produit, documentation) dans plusieurs langues. La traduction humaine est coûteuse (0.15-0.25 EUR/mot) et lente (2-5 jours par document). Les outils de traduction automatique classiques produisent des résultats littéraux qui ne respectent ni le ton de la marque, ni les spécificités culturelles du marché cible.",
    value:
      "Un agent IA spécialisé traduit et localise les contenus en adaptant le message aux spécificités culturelles, réglementaires et marketing de chaque marché cible. Il respecte le glossaire de la marque, adapte les références culturelles, convertit les formats (dates, devises, unités) et produit un contenu qui semble natif. La qualité approche celle d'un traducteur professionnel à un coût 10x inférieur.",
    inputs: [
      "Contenu source en français (texte, HTML, Markdown, JSON)",
      "Langue et marché cible (ex: anglais US, allemand Allemagne, espagnol Mexique)",
      "Glossaire de marque et terminologie spécifique",
      "Guide de style et ton par marché",
      "Contexte marketing (type de contenu, audience, objectif)",
    ],
    outputs: [
      "Contenu traduit et localisé dans la langue cible",
      "Rapport de localisation (adaptations culturelles effectuées, termes du glossaire appliqués)",
      "Score de qualité de la traduction (fluency, accuracy, style)",
      "Liste des segments nécessitant une relecture humaine",
      "Contenu au format original préservé (HTML, Markdown, JSON)",
    ],
    risks: [
      "Contresens ou nuance culturelle manquée pouvant offenser le marché cible",
      "Non-respect des contraintes réglementaires locales (mentions légales, RGPD vs CCPA)",
      "Perte du ton et de la personnalité de la marque dans la traduction",
      "Hallucination du LLM ajoutant ou omettant des informations du texte source",
    ],
    roiIndicatif:
      "Réduction de 85% du coût de traduction (de 0.20 EUR/mot à 0.03 EUR/mot). Accélération du time-to-market international de 5 jours à 2 heures. Capacité de localiser en 10+ langues simultanément. Cohérence terminologique de 98% grâce au glossaire automatisé.",
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
│  Contenu    │────▶│  Agent IA    │────▶│  Contenu    │
│  source FR  │     │  (Traduction │     │  localisé   │
│  (texte,    │     │  + Adaptation│     │  (multi-    │
│  HTML, JSON)│     │  culturelle) │     │  langues)   │
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌───────────┐ ┌──────────┐ ┌──────────┐
       │ Glossaire │ │  Guide   │ │ Mémoire  │
       │  marque   │ │  style   │ │ traduct. │
       └───────────┘ └──────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Prérequis et configuration",
        content:
          "Installez les dépendances et configurez l'environnement. Préparez votre glossaire de marque et vos guides de style par marché cible pour garantir la cohérence terminologique.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install anthropic langchain psycopg2-binary python-dotenv fastapi beautifulsoup4 markdown pyyaml deepl`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=postgresql://user:pass@localhost:5432/traduction_db
DEEPL_API_KEY=...  # Optionnel: pour comparaison qualité
SLACK_WEBHOOK_REVIEW=https://hooks.slack.com/services/...
LANGUES_CIBLES=en-US,de-DE,es-ES,it-IT,pt-BR,ja-JP
SEUIL_QUALITE_AUTO=0.85`,
            filename: ".env",
          },
        ],
      },
      {
        title: "Gestion du glossaire et de la mémoire de traduction",
        content:
          "Créez un système de glossaire et de mémoire de traduction qui assure la cohérence terminologique à travers tous les contenus. Le glossaire stocke les traductions validées des termes clés de la marque.",
        codeSnippets: [
          {
            language: "python",
            code: `from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import json
import psycopg2
import os

class EntreeGlossaire(BaseModel):
    terme_source: str
    traductions: Dict[str, str]  # {"en-US": "...", "de-DE": "..."}
    contexte: str = ""
    ne_pas_traduire: bool = False  # Pour les noms propres, marques

class MemoireTraduction(BaseModel):
    segment_source: str
    traductions: Dict[str, str]
    valide_par_humain: bool = False
    date_validation: Optional[str] = None

# Glossaire de marque
GLOSSAIRE = [
    EntreeGlossaire(
        terme_source="intelligence artificielle agentique",
        traductions={"en-US": "agentic AI", "de-DE": "agentische KI", "es-ES": "IA agéntica", "it-IT": "IA agentica"},
        contexte="Terme technique central de la marque"
    ),
    EntreeGlossaire(
        terme_source="automatisation intelligente",
        traductions={"en-US": "intelligent automation", "de-DE": "intelligente Automatisierung", "es-ES": "automatización inteligente"},
        contexte="Feature principale du produit"
    ),
    EntreeGlossaire(
        terme_source="NomMarque",
        traductions={},
        ne_pas_traduire=True,
        contexte="Nom de la marque - ne jamais traduire"
    ),
]

def charger_glossaire(langue_cible: str) -> Dict[str, str]:
    """Charge le glossaire pour une langue cible donnée"""
    glossaire = {}
    for entree in GLOSSAIRE:
        if entree.ne_pas_traduire:
            glossaire[entree.terme_source] = entree.terme_source
        elif langue_cible in entree.traductions:
            glossaire[entree.terme_source] = entree.traductions[langue_cible]
    return glossaire

def chercher_memoire_traduction(segment: str, langue_cible: str) -> Optional[str]:
    """Recherche un segment déjà traduit dans la mémoire"""
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()
    cur.execute(
        "SELECT traductions FROM memoire_traduction WHERE segment_source = %s AND valide_par_humain = true",
        (segment,)
    )
    result = cur.fetchone()
    conn.close()
    if result:
        traductions = json.loads(result[0])
        return traductions.get(langue_cible)
    return None`,
            filename: "glossaire.py",
          },
        ],
      },
      {
        title: "Moteur de traduction et localisation",
        content:
          "Le coeur de l'agent : un moteur de traduction qui segmente le contenu, applique le glossaire, traduit avec le LLM en respectant le contexte culturel, puis reconstitue le format original.",
        codeSnippets: [
          {
            language: "python",
            code: `import anthropic
import json
import re
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from glossaire import charger_glossaire, chercher_memoire_traduction

class SegmentTraduit(BaseModel):
    source: str
    traduction: str
    score_confiance: float = Field(ge=0, le=1)
    adaptations_culturelles: List[str] = Field(default_factory=list)
    necessite_relecture: bool = False
    raison_relecture: str = ""

class ResultatTraduction(BaseModel):
    langue_source: str
    langue_cible: str
    contenu_traduit: str
    segments: List[SegmentTraduit]
    score_qualite_global: float
    glossaire_applique: Dict[str, str]
    adaptations_culturelles: List[str]
    nb_segments_relecture: int

GUIDES_STYLE = {
    "en-US": "Ton direct et action-oriented. Phrases courtes. Utiliser 'you' fréquemment. Éviter le passif.",
    "de-DE": "Ton formel (Sie). Précision technique valorisée. Phrases structurées. Respecter la capitalisation des noms.",
    "es-ES": "Ton chaleureux mais professionnel. Utiliser 'usted' en B2B. Adapter les expressions idiomatiques.",
    "it-IT": "Ton élégant et engageant. Forme de politesse 'Lei'. Adapter les références culturelles.",
    "pt-BR": "Ton moderne et accessible. Utiliser 'você'. Adapter au marché brésilien, pas portugais.",
    "ja-JP": "Niveau de politesse keigo en B2B. Adapter la structure (sujet souvent omis). Formats: YYYY年MM月DD日.",
}

client = anthropic.Anthropic()

def traduire_contenu(
    contenu: str,
    langue_cible: str,
    type_contenu: str = "page_web",
    contexte_marketing: str = ""
) -> ResultatTraduction:
    glossaire = charger_glossaire(langue_cible)
    guide_style = GUIDES_STYLE.get(langue_cible, "")

    # Segmenter le contenu
    segments = segmenter_contenu(contenu)
    segments_traduits = []

    for segment in segments:
        # Vérifier la mémoire de traduction
        traduction_existante = chercher_memoire_traduction(segment, langue_cible)
        if traduction_existante:
            segments_traduits.append(SegmentTraduit(
                source=segment, traduction=traduction_existante,
                score_confiance=1.0, necessite_relecture=False
            ))
            continue

        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": f"""Tu es un traducteur-localiseur professionnel spécialisé dans le marketing B2B.

LANGUE SOURCE : Français (France)
LANGUE CIBLE : {langue_cible}
TYPE DE CONTENU : {type_contenu}
CONTEXTE : {contexte_marketing}

GUIDE DE STYLE ({langue_cible}) :
{guide_style}

GLOSSAIRE OBLIGATOIRE (utilise ces traductions exactes) :
{json.dumps(glossaire, indent=2, ensure_ascii=False)}

SEGMENT À TRADUIRE :
{segment}

RÈGLES :
1. Traduis le sens, pas les mots. Adapte les expressions idiomatiques.
2. Respecte le glossaire de marque (termes imposés ci-dessus).
3. Adapte les formats : dates, devises, unités de mesure.
4. Adapte les références culturelles au marché cible.
5. Préserve le formatage (Markdown, HTML) intact.
6. Ne jamais ajouter ni omettre d'information par rapport au source.
7. Signale si un segment nécessite une relecture humaine.

Retourne un JSON SegmentTraduit."""}]
        )
        result = json.loads(response.content[0].text)
        result["source"] = segment
        segments_traduits.append(SegmentTraduit(**result))

    contenu_final = " ".join([s.traduction for s in segments_traduits])
    score_global = sum(s.score_confiance for s in segments_traduits) / max(len(segments_traduits), 1)
    adaptations = [a for s in segments_traduits for a in s.adaptations_culturelles]

    return ResultatTraduction(
        langue_source="fr-FR", langue_cible=langue_cible,
        contenu_traduit=contenu_final, segments=segments_traduits,
        score_qualite_global=round(score_global, 3),
        glossaire_applique=glossaire, adaptations_culturelles=adaptations,
        nb_segments_relecture=sum(1 for s in segments_traduits if s.necessite_relecture)
    )

def segmenter_contenu(contenu: str) -> List[str]:
    """Segmente le contenu en unités de traduction"""
    segments = re.split(r'\\n\\n+', contenu)
    return [s.strip() for s in segments if s.strip()]`,
            filename: "traducteur.py",
          },
        ],
      },
      {
        title: "Traitement par lots et formats multiples",
        content:
          "Gérez la traduction par lots de fichiers entiers (HTML, Markdown, JSON de localisation) en préservant la structure et le formatage d'origine. Le module supporte les formats i18n standards.",
        codeSnippets: [
          {
            language: "python",
            code: `import json
import yaml
from bs4 import BeautifulSoup
from typing import Dict, List
from traducteur import traduire_contenu, ResultatTraduction
import os

def traduire_fichier_json_i18n(
    fichier_source: str,
    langue_cible: str
) -> Dict:
    """Traduit un fichier JSON i18n (format clé-valeur)"""
    with open(fichier_source, "r", encoding="utf-8") as f:
        source = json.load(f)

    resultat = {}
    def traduire_recursif(obj, prefix=""):
        if isinstance(obj, str):
            trad = traduire_contenu(obj, langue_cible, type_contenu="ui_string")
            return trad.contenu_traduit
        elif isinstance(obj, dict):
            return {k: traduire_recursif(v, f"{prefix}.{k}") for k, v in obj.items()}
        elif isinstance(obj, list):
            return [traduire_recursif(item, f"{prefix}[{i}]") for i, item in enumerate(obj)]
        return obj

    return traduire_recursif(source)

def traduire_html(html_source: str, langue_cible: str) -> str:
    """Traduit le contenu textuel d'un fichier HTML en préservant la structure"""
    soup = BeautifulSoup(html_source, "html.parser")

    # Éléments contenant du texte à traduire
    for element in soup.find_all(text=True):
        if element.parent.name in ["script", "style", "code", "pre"]:
            continue
        texte = element.strip()
        if texte and len(texte) > 2:
            trad = traduire_contenu(texte, langue_cible, type_contenu="page_web")
            element.replace_with(trad.contenu_traduit)

    # Traduire les attributs alt, title, placeholder
    for tag in soup.find_all(True):
        for attr in ["alt", "title", "placeholder", "aria-label"]:
            if tag.get(attr):
                trad = traduire_contenu(tag[attr], langue_cible, type_contenu="ui_string")
                tag[attr] = trad.contenu_traduit

    return str(soup)

def traduire_lot(
    dossier_source: str,
    langue_cible: str,
    dossier_sortie: str
) -> List[Dict]:
    """Traduit un dossier complet de fichiers"""
    resultats = []
    os.makedirs(dossier_sortie, exist_ok=True)
    for fichier in os.listdir(dossier_source):
        chemin = os.path.join(dossier_source, fichier)
        if fichier.endswith(".json"):
            traduit = traduire_fichier_json_i18n(chemin, langue_cible)
            with open(os.path.join(dossier_sortie, fichier), "w", encoding="utf-8") as f:
                json.dump(traduit, f, ensure_ascii=False, indent=2)
        elif fichier.endswith(".html"):
            with open(chemin, "r", encoding="utf-8") as f:
                traduit = traduire_html(f.read(), langue_cible)
            with open(os.path.join(dossier_sortie, fichier), "w", encoding="utf-8") as f:
                f.write(traduit)
        resultats.append({"fichier": fichier, "langue": langue_cible, "status": "traduit"})
    return resultats`,
            filename: "traduction_lots.py",
          },
        ],
      },
      {
        title: "API et contrôle qualité",
        content:
          "Déployez l'API de traduction avec un système de contrôle qualité intégré. Les traductions sous le seuil de qualité sont automatiquement envoyées pour relecture humaine.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from traducteur import traduire_contenu, ResultatTraduction
from traduction_lots import traduire_fichier_json_i18n, traduire_html
import httpx
import os

app = FastAPI(title="API Traduction & Localisation IA")

class DemandeTraduction(BaseModel):
    contenu: str
    langue_cible: str
    type_contenu: str = "page_web"
    contexte: str = ""
    auto_publish: bool = False

@app.post("/api/traduction/traduire")
async def traduire(demande: DemandeTraduction) -> dict:
    resultat = traduire_contenu(
        contenu=demande.contenu,
        langue_cible=demande.langue_cible,
        type_contenu=demande.type_contenu,
        contexte_marketing=demande.contexte
    )
    seuil = float(os.getenv("SEUIL_QUALITE_AUTO", 0.85))
    if resultat.score_qualite_global < seuil or resultat.nb_segments_relecture > 0:
        await notifier_relecture(resultat)
        return {**resultat.model_dump(), "status": "en_relecture",
                "message": f"Qualité {resultat.score_qualite_global:.0%} sous le seuil de {seuil:.0%}. Envoyé en relecture."}
    if demande.auto_publish:
        return {**resultat.model_dump(), "status": "publie"}
    return {**resultat.model_dump(), "status": "traduit"}

@app.post("/api/traduction/lot")
async def traduire_en_lot(langues: List[str], contenu: str, type_contenu: str = "page_web"):
    resultats = {}
    for langue in langues:
        resultat = traduire_contenu(contenu, langue, type_contenu)
        resultats[langue] = {
            "contenu": resultat.contenu_traduit,
            "score": resultat.score_qualite_global,
            "adaptations": resultat.adaptations_culturelles
        }
    return resultats

async def notifier_relecture(resultat: ResultatTraduction):
    webhook = os.getenv("SLACK_WEBHOOK_REVIEW")
    segments_a_revoir = [s for s in resultat.segments if s.necessite_relecture]
    message = {
        "blocks": [
            {"type": "header", "text": {"type": "plain_text", "text": f"Relecture requise - {resultat.langue_cible}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Score qualité :* {resultat.score_qualite_global:.0%}\\n*Segments à revoir :* {len(segments_a_revoir)}"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "\\n".join([f"• _{s.source[:80]}..._ → {s.raison_relecture}" for s in segments_a_revoir[:5]])}},
        ]
    }
    async with httpx.AsyncClient() as client:
        await client.post(webhook, json=message)`,
            filename: "api_traduction.py",
          },
        ],
      },
      {
        title: "Tests de qualité et benchmarks",
        content:
          "Testez la qualité des traductions en comparant avec des traductions de référence. Mesurez la précision du glossaire, la fluidité et la fidélité au texte source.",
        codeSnippets: [
          {
            language: "python",
            code: `import pytest
from traducteur import traduire_contenu

def test_traduction_anglais_qualite():
    contenu = "Notre solution d'intelligence artificielle agentique permet aux entreprises françaises d'automatiser leurs processus métier en toute sécurité."
    resultat = traduire_contenu(contenu, "en-US", type_contenu="page_web")
    assert resultat.score_qualite_global >= 0.8
    assert "agentic AI" in resultat.contenu_traduit, "Le glossaire doit être respecté"
    assert "French" in resultat.contenu_traduit or "companies" in resultat.contenu_traduit

def test_glossaire_respecte():
    contenu = "L'automatisation intelligente transforme les processus métier."
    resultat = traduire_contenu(contenu, "de-DE")
    assert "intelligente Automatisierung" in resultat.contenu_traduit, "Le terme du glossaire DE doit être utilisé"

def test_preservation_formatage_html():
    contenu = "<h1>Bienvenue</h1><p>Découvrez notre <strong>solution IA</strong> pour les entreprises.</p>"
    from traduction_lots import traduire_html
    resultat = traduire_html(contenu, "en-US")
    assert "<h1>" in resultat and "</h1>" in resultat, "Les balises HTML doivent être préservées"
    assert "<strong>" in resultat

def test_traduction_japonais():
    contenu = "Contactez-nous pour une démonstration gratuite de notre plateforme."
    resultat = traduire_contenu(contenu, "ja-JP", type_contenu="page_web")
    assert resultat.score_qualite_global >= 0.7
    assert len(resultat.contenu_traduit) > 0`,
            filename: "test_traduction.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contenus marketing ne contiennent généralement pas de données personnelles. En cas de données nominatives dans le contenu source (témoignages clients, études de cas), elles sont transmises au LLM uniquement pour traduction et ne sont pas stockées dans les logs. Le glossaire et la mémoire de traduction sont chiffrés au repos.",
      auditLog: "Chaque traduction est loguée avec : horodatage, contenu source (hash), langue cible, score qualité, segments nécessitant relecture, glossaire appliqué, et validation humaine éventuelle. Rétention 12 mois pour amélioration continue du modèle.",
      humanInTheLoop: "Les traductions avec un score qualité inférieur à 85% sont automatiquement envoyées à un traducteur humain pour relecture. Les contenus juridiques (CGV, mentions légales, contrats) nécessitent toujours une validation humaine. Les traducteurs peuvent enrichir le glossaire et la mémoire de traduction.",
      monitoring: "Dashboard traduction : volume de mots traduits par langue, score qualité moyen par langue, taux de relecture humaine, coût par mot, temps moyen de traduction, couverture du glossaire, comparaison qualité IA vs humain sur échantillons.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouveau contenu à traduire) → Node Code (détection format et segmentation) → Node Loop (pour chaque langue cible) → Node HTTP Request (API LLM traduction) → Node Code (reconstruction format) → Node IF (score qualité >= seuil ?) → Branch OK: Node HTTP Request (CMS publication) → Branch relecture: Node Slack (notification traducteur) → Node PostgreSQL (log et mémoire de traduction).",
      nodes: ["Webhook (contenu)", "Code (segmentation)", "Loop (langues cibles)", "HTTP Request (LLM traduction)", "Code (reconstruction)", "IF (qualité)", "HTTP Request (CMS)", "Slack (relecture)", "PostgreSQL (log)"],
      triggerType: "Webhook (nouveau contenu ou mise à jour CMS)",
    },
    estimatedTime: "6-8h",
    difficulty: "Moyen",
    sectors: ["SaaS", "E-commerce", "Tourisme", "Luxe", "Industrie"],
    metiers: ["Marketing International", "Content Marketing", "Localisation"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Traduction et Localisation de Contenu — Guide Complet",
    metaDescription:
      "Localisez automatiquement vos contenus marketing pour l'international avec un agent IA. Glossaire de marque, adaptation culturelle et contrôle qualité intégré. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-automatisation-emails",
    title: "Agent d'Automatisation et Tri Intelligent des Emails",
    subtitle: "Catégorisez, priorisez et rédigez des réponses automatiques à vos emails entrants grâce à l'IA",
    problem:
      "Les équipes support et commerciales reçoivent des centaines d'emails par jour. Le tri manuel est chronophage, des messages urgents passent inaperçus, et la qualité des réponses varie selon les agents. Le temps moyen de première réponse dépasse souvent les 24 heures.",
    value:
      "Un agent IA analyse chaque email entrant, le catégorise automatiquement (demande technique, réclamation, demande commerciale, spam), attribue un niveau de priorité, et génère un brouillon de réponse personnalisé. Les emails critiques sont escaladés instantanément.",
    inputs: [
      "Contenu de l'email (sujet, corps, pièces jointes)",
      "Historique de correspondance avec l'expéditeur",
      "Base de connaissances interne (FAQ, procédures)",
      "Règles de routage et de priorité métier",
      "Modèles de réponses existants",
    ],
    outputs: [
      "Catégorie de l'email (support, commercial, administratif, spam)",
      "Niveau de priorité (urgent, normal, faible)",
      "Brouillon de réponse personnalisé",
      "Résumé de l'email en une ligne",
      "Suggestions d'actions (escalade, transfert, archivage)",
    ],
    risks: [
      "Mauvaise catégorisation entraînant la perte d'emails critiques",
      "Réponses automatiques inappropriées envoyées sans validation",
      "Non-respect du RGPD lors de l'analyse des pièces jointes",
      "Dépendance excessive à l'automatisation pour des sujets sensibles",
    ],
    roiIndicatif:
      "Réduction de 70% du temps de tri des emails. Temps de première réponse divisé par 4. Augmentation de 40% de la satisfaction client sur le canal email.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Small", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: "+-----------------+     +------------------+     +----------------+\n|   Boite Email   |---->|   Agent LLM      |---->|   CRM / Help   |\n|   (IMAP/API)    |     |   (Tri + Rédac.) |     |   Desk         |\n+-----------------+     +--------+---------+     +----------------+\n                                 |\n                        +--------v---------+\n                        |  Base de         |\n                        |  Connaissances   |\n                        +------------------+",
    tutorial: [
      {
        title: "Prérequis et installation",
        content:
          "Installez les dépendances nécessaires et configurez les accès à l'API Anthropic ainsi qu'à votre serveur de messagerie IMAP ou API Gmail/Outlook.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain imapclient pydantic fastapi",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modèle de données pour la classification",
        content:
          "Définissez les structures de données pour la catégorisation et la priorisation des emails. Le modèle inclut la catégorie, la priorité, le résumé et le brouillon de réponse.",
        codeSnippets: [
          {
            language: "python",
            code: "from pydantic import BaseModel, Field\nfrom enum import Enum\nfrom typing import Optional\n\nclass EmailCategory(str, Enum):\n    SUPPORT = \"support_technique\"\n    COMMERCIAL = \"demande_commerciale\"\n    RECLAMATION = \"reclamation\"\n    ADMINISTRATIF = \"administratif\"\n    SPAM = \"spam\"\n\nclass Priority(str, Enum):\n    URGENT = \"urgent\"\n    NORMAL = \"normal\"\n    LOW = \"faible\"\n\nclass EmailAnalysis(BaseModel):\n    category: EmailCategory\n    priority: Priority\n    summary: str = Field(max_length=200)\n    draft_response: str\n    suggested_action: str\n    confidence: float = Field(ge=0.0, le=1.0)\n    needs_human_review: bool = False",
            filename: "models.py",
          },
        ],
      },
      {
        title: "Récupération des emails via IMAP",
        content:
          "Connectez-vous au serveur de messagerie pour récupérer les emails non lus. Cette étape utilise imapclient pour un accès IMAP sécurisé.",
        codeSnippets: [
          {
            language: "python",
            code: "from imapclient import IMAPClient\nimport email\nfrom email.header import decode_header\n\ndef fetch_unread_emails(host: str, user: str, password: str) -> list[dict]:\n    with IMAPClient(host, ssl=True) as client:\n        client.login(user, password)\n        client.select_folder(\"INBOX\")\n        messages = client.search([\"UNSEEN\"])\n        emails = []\n        for uid, data in client.fetch(messages, [\"RFC822\"]).items():\n            msg = email.message_from_bytes(data[b\"RFC822\"])\n            subject = decode_header(msg[\"Subject\"])[0][0]\n            if isinstance(subject, bytes):\n                subject = subject.decode(\"utf-8\", errors=\"replace\")\n            body = \"\"\n            if msg.is_multipart():\n                for part in msg.walk():\n                    if part.get_content_type() == \"text/plain\":\n                        body = part.get_payload(decode=True).decode(\"utf-8\", errors=\"replace\")\n            else:\n                body = msg.get_payload(decode=True).decode(\"utf-8\", errors=\"replace\")\n            emails.append({\"uid\": uid, \"from\": msg[\"From\"], \"subject\": subject, \"body\": body})\n        return emails",
            filename: "email_fetcher.py",
          },
        ],
      },
      {
        title: "Agent de classification et rédaction",
        content:
          "Construisez l'agent IA qui analyse chaque email, le catégorise, lui attribue une priorité et génère un brouillon de réponse. L'agent utilise le contexte de votre base de connaissances.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nimport json\n\nclient = anthropic.Anthropic()\n\ndef analyze_email(email_data: dict, knowledge_base: str) -> EmailAnalysis:\n    prompt = (\n        \"Tu es un agent de tri d'emails professionnel. \"\n        \"Analyse l'email suivant et retourne un JSON structuré.\\n\\n\"\n        \"Base de connaissances:\\n{kb}\\n\\n\"\n        \"Email:\\nDe: {sender}\\nSujet: {subject}\\nCorps: {body}\\n\\n\"\n        \"Retourne un JSON avec: category, priority, summary, \"\n        \"draft_response, suggested_action, confidence, needs_human_review\"\n    ).format(\n        kb=knowledge_base,\n        sender=email_data[\"from\"],\n        subject=email_data[\"subject\"],\n        body=email_data[\"body\"][:3000]\n    )\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=2048,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    return EmailAnalysis.model_validate_json(message.content[0].text)",
            filename: "classifier.py",
          },
        ],
      },
      {
        title: "Pipeline de traitement automatisé",
        content:
          "Créez le pipeline complet qui orchestre la récupération, l'analyse et le routage des emails. Le pipeline tourne en boucle et traite les nouveaux messages toutes les minutes.",
        codeSnippets: [
          {
            language: "python",
            code: "import time\nimport logging\n\nlogger = logging.getLogger(__name__)\n\ndef process_email_pipeline(config: dict):\n    emails = fetch_unread_emails(\n        config[\"imap_host\"], config[\"imap_user\"], config[\"imap_pass\"]\n    )\n    knowledge_base = load_knowledge_base(config[\"kb_path\"])\n    for email_data in emails:\n        try:\n            analysis = analyze_email(email_data, knowledge_base)\n            save_analysis(email_data[\"uid\"], analysis)\n            if analysis.priority == Priority.URGENT:\n                send_slack_alert(email_data, analysis)\n            if analysis.category == EmailCategory.SPAM:\n                move_to_spam(email_data[\"uid\"])\n                continue\n            if not analysis.needs_human_review and analysis.confidence > 0.85:\n                send_draft_response(email_data, analysis.draft_response)\n            else:\n                assign_to_agent(email_data, analysis)\n            logger.info(\"Email %s traite: %s / %s\", email_data[\"uid\"], analysis.category, analysis.priority)\n        except Exception as e:\n            logger.error(\"Erreur traitement email %s: %s\", email_data[\"uid\"], e)\n\ndef run_continuous(config: dict, interval: int = 60):\n    while True:\n        process_email_pipeline(config)\n        time.sleep(interval)",
            filename: "pipeline.py",
          },
        ],
      },
      {
        title: "API REST pour le dashboard",
        content:
          "Exposez une API FastAPI pour consulter les statistiques de tri, rechercher des emails analysés et ajuster les règles de classification en temps réel.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, Query\nfrom typing import Optional\n\napp = FastAPI(title=\"Email Automation Agent\")\n\n@app.get(\"/api/stats\")\nasync def get_stats():\n    return {\n        \"total_processed\": await count_processed_today(),\n        \"by_category\": await count_by_category(),\n        \"by_priority\": await count_by_priority(),\n        \"auto_responded\": await count_auto_responded(),\n        \"avg_confidence\": await avg_confidence_score()\n    }\n\n@app.get(\"/api/emails\")\nasync def list_emails(\n    category: Optional[str] = Query(None),\n    priority: Optional[str] = Query(None),\n    limit: int = Query(50, le=200)\n):\n    filters = {}\n    if category:\n        filters[\"category\"] = category\n    if priority:\n        filters[\"priority\"] = priority\n    return await fetch_analyzed_emails(filters, limit)",
            filename: "api.py",
          },
        ],
      },
      {
        title: "Tests unitaires",
        content:
          "Validez le bon fonctionnement de l'agent avec des tests couvrant chaque catégorie d'email et les cas limites (emails vides, pièces jointes, emails multilingues).",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nfrom models import EmailAnalysis, EmailCategory, Priority\nfrom classifier import analyze_email\n\ndef test_support_email_classification():\n    email = {\n        \"from\": \"client@example.com\",\n        \"subject\": \"Bug sur la page de paiement\",\n        \"body\": \"Bonjour, je n'arrive plus a finaliser mon achat. L'erreur 500 apparait.\"\n    }\n    result = analyze_email(email, \"FAQ: Les erreurs 500 sont liees au service de paiement.\")\n    assert result.category == EmailCategory.SUPPORT\n    assert result.priority in [Priority.URGENT, Priority.NORMAL]\n    assert result.confidence >= 0.7\n\ndef test_spam_detection():\n    email = {\n        \"from\": \"promo@spam.xyz\",\n        \"subject\": \"GAGNEZ 10000 EUR MAINTENANT\",\n        \"body\": \"Cliquez ici pour recevoir votre prix. Offre limitee.\"\n    }\n    result = analyze_email(email, \"\")\n    assert result.category == EmailCategory.SPAM\n\ndef test_urgent_email_flagged():\n    email = {\n        \"from\": \"directeur@enterprise.fr\",\n        \"subject\": \"URGENT - Systeme en panne\",\n        \"body\": \"Le systeme de production est hors service depuis 2 heures.\"\n    }\n    result = analyze_email(email, \"\")\n    assert result.priority == Priority.URGENT\n    assert result.needs_human_review is True",
            filename: "test_classifier.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les emails contiennent souvent des données personnelles (noms, adresses, numéros de téléphone). Le contenu est pseudonymisé avant envoi au LLM : les emails, numéros de téléphone et adresses sont remplacés par des tokens. Les pièces jointes ne sont jamais envoyées au modèle. Les données sont chiffrées au repos (AES-256) et en transit (TLS 1.3).",
      auditLog: "Chaque email traité génère une entrée d'audit : identifiant unique, horodatage de réception, catégorie attribuée, priorité, score de confiance, action prise (réponse auto, escalade, archivage), identifiant de l'agent humain si intervention. Conservation des logs pendant 3 ans.",
      humanInTheLoop: "Les emails classés comme réclamation ou avec un score de confiance inférieur à 0.85 sont systématiquement soumis à un agent humain pour validation avant envoi de la réponse. Les emails marqués urgents déclenchent une notification immédiate au responsable d'équipe. Un bouton de correction permet de réajuster la classification.",
      monitoring: "Dashboard temps réel : volume d'emails traités par heure, taux de classification automatique, taux de réponse automatique, temps moyen de traitement, distribution par catégorie, score de confiance moyen, taux de correction humaine, emails en attente de validation.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Trigger Email (IMAP/Gmail) toutes les minutes → Node Code (extraction contenu et métadonnées) → Node HTTP Request (API Claude pour classification) → Node Switch (catégorie) → Branch support: Node HTTP Request (recherche base de connaissances) → Node HTTP Request (génération réponse) → Branch urgent: Node Slack (alerte équipe) → Branch spam: Node Email (déplacement dossier spam) → Node PostgreSQL (sauvegarde analyse et audit).",
      nodes: ["Email Trigger (IMAP)", "Code (extraction)", "HTTP Request (classification LLM)", "Switch (catégorie)", "HTTP Request (KB search)", "HTTP Request (réponse LLM)", "Slack (alerte urgent)", "Email (move spam)", "PostgreSQL (audit)"],
      triggerType: "Email Trigger (IMAP polling toutes les 60 secondes)",
    },
    estimatedTime: "4-6h",
    difficulty: "Facile",
    sectors: ["Services", "E-commerce", "B2B SaaS", "Technologie"],
    metiers: ["Support Client", "Commercial", "Administration"],
    functions: ["Support"],
    metaTitle: "Agent IA d'Automatisation des Emails — Guide Complet",
    metaDescription:
      "Automatisez le tri et la réponse à vos emails avec un agent IA. Classification intelligente, priorisation automatique et brouillons de réponse personnalisés. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-analyse-appels-telephoniques",
    title: "Agent d'Analyse des Appels Téléphoniques",
    subtitle: "Transcrivez et analysez vos appels commerciaux et support avec Whisper et un LLM",
    problem:
      "Les entreprises perdent des insights précieux contenus dans leurs appels téléphoniques. Les managers n'ont pas le temps d'écouter des heures d'enregistrements, les bonnes pratiques ne sont pas partagées, et les signaux faibles (insatisfaction client, objections récurrentes) passent inaperçus.",
    value:
      "Un agent IA transcrit automatiquement chaque appel via Whisper, puis analyse la transcription avec un LLM pour extraire les points clés, le sentiment, les objections, les engagements pris et un score de qualité. Les managers obtiennent un tableau de bord synthétique de chaque conversation.",
    inputs: [
      "Enregistrement audio de l'appel (WAV, MP3, M4A)",
      "Métadonnées de l'appel (date, durée, participants)",
      "Fiche client CRM associée",
      "Grille d'évaluation qualité (critères métier)",
      "Historique des interactions précédentes",
    ],
    outputs: [
      "Transcription complète horodatée",
      "Résumé structuré de l'appel (3-5 points clés)",
      "Analyse de sentiment par segment",
      "Liste des objections et réponses apportées",
      "Score de qualité de l'appel (0-100)",
      "Engagements et prochaines étapes identifiées",
    ],
    risks: [
      "Erreurs de transcription sur les termes techniques ou noms propres",
      "Non-conformité RGPD si les participants n'ont pas consenti à l'enregistrement",
      "Biais dans l'analyse de sentiment selon l'accent ou la langue",
      "Utilisation abusive pour la surveillance excessive des employés",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de revue des appels par les managers. Amélioration de 25% du taux de conversion grâce au coaching ciblé. Détection 3x plus rapide des clients à risque de churn.",
    recommendedStack: [
      { name: "OpenAI Whisper Large V3", category: "Other" },
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS S3", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Whisper.cpp (local)", category: "Other", isFree: true },
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "MinIO (S3 self-hosted)", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-----------------+     +------------------+     +------------------+\n|  Enregistrement |---->|   Whisper        |---->|   Agent LLM      |\n|  Audio (S3)     |     |   (Transcription)|     |   (Analyse)      |\n+-----------------+     +------------------+     +--------+---------+\n                                                          |\n                        +------------------+     +--------v---------+\n                        |   Dashboard      |<----|   PostgreSQL     |\n                        |   (Résultats)    |     |   (Stockage)     |\n                        +------------------+     +------------------+",
    tutorial: [
      {
        title: "Prérequis et installation",
        content:
          "Installez Whisper pour la transcription audio et les bibliothèques nécessaires pour l'analyse LLM. Vous aurez besoin de ffmpeg pour le traitement audio.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai-whisper anthropic langchain pydantic fastapi pydub psycopg2-binary\nbrew install ffmpeg  # macOS\n# apt install ffmpeg  # Linux",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modèles de données",
        content:
          "Définissez les structures pour la transcription, l'analyse et le score de qualité. Ces modèles garantissent une sortie structurée et validée.",
        codeSnippets: [
          {
            language: "python",
            code: "from pydantic import BaseModel, Field\nfrom enum import Enum\nfrom typing import Optional\n\nclass Sentiment(str, Enum):\n    POSITIF = \"positif\"\n    NEUTRE = \"neutre\"\n    NEGATIF = \"negatif\"\n\nclass TranscriptSegment(BaseModel):\n    start_time: float\n    end_time: float\n    speaker: str\n    text: str\n    sentiment: Optional[Sentiment] = None\n\nclass Objection(BaseModel):\n    text: str\n    response_given: str\n    was_handled: bool\n\nclass CallAnalysis(BaseModel):\n    summary: str = Field(max_length=500)\n    key_points: list[str] = Field(min_length=1, max_length=5)\n    overall_sentiment: Sentiment\n    objections: list[Objection]\n    commitments: list[str]\n    next_steps: list[str]\n    quality_score: int = Field(ge=0, le=100)\n    coaching_tips: list[str]\n    churn_risk: bool = False",
            filename: "models.py",
          },
        ],
      },
      {
        title: "Transcription audio avec Whisper",
        content:
          "Utilisez Whisper pour transcrire l'audio en texte avec horodatage. Le modèle large-v3 offre la meilleure précision pour le français.",
        codeSnippets: [
          {
            language: "python",
            code: "import whisper\nfrom models import TranscriptSegment\n\ndef transcribe_audio(audio_path: str, model_size: str = \"large-v3\") -> list[TranscriptSegment]:\n    model = whisper.load_model(model_size)\n    result = model.transcribe(\n        audio_path,\n        language=\"fr\",\n        task=\"transcribe\",\n        verbose=False\n    )\n    segments = []\n    for seg in result[\"segments\"]:\n        segments.append(TranscriptSegment(\n            start_time=seg[\"start\"],\n            end_time=seg[\"end\"],\n            speaker=\"inconnu\",  # diarisation separee\n            text=seg[\"text\"].strip()\n        ))\n    return segments\n\ndef format_transcript(segments: list[TranscriptSegment]) -> str:\n    lines = []\n    for seg in segments:\n        minutes = int(seg.start_time // 60)\n        seconds = int(seg.start_time % 60)\n        timestamp = \"{:02d}:{:02d}\".format(minutes, seconds)\n        lines.append(\"[{}] {}: {}\".format(timestamp, seg.speaker, seg.text))\n    return \"\\n\".join(lines)",
            filename: "transcriber.py",
          },
        ],
      },
      {
        title: "Analyse de l'appel par le LLM",
        content:
          "Envoyez la transcription au LLM avec votre grille d'évaluation pour obtenir une analyse structurée : résumé, sentiment, objections, score de qualité et conseils de coaching.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nfrom models import CallAnalysis\n\nclient = anthropic.Anthropic()\n\ndef analyze_call(transcript: str, evaluation_grid: str, client_context: str) -> CallAnalysis:\n    prompt = (\n        \"Tu es un expert en analyse d'appels commerciaux et support.\\n\"\n        \"Analyse cette transcription et retourne un JSON structure.\\n\\n\"\n        \"Grille d'evaluation:\\n{grid}\\n\\n\"\n        \"Contexte client:\\n{ctx}\\n\\n\"\n        \"Transcription:\\n{transcript}\\n\\n\"\n        \"Retourne un JSON avec: summary, key_points (3-5), \"\n        \"overall_sentiment, objections (avec text, response_given, was_handled), \"\n        \"commitments, next_steps, quality_score (0-100), \"\n        \"coaching_tips, churn_risk\"\n    ).format(\n        grid=evaluation_grid,\n        ctx=client_context,\n        transcript=transcript[:8000]\n    )\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=4096,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    return CallAnalysis.model_validate_json(message.content[0].text)",
            filename: "analyzer.py",
          },
        ],
      },
      {
        title: "Pipeline complet de traitement",
        content:
          "Orchestrez le pipeline complet : récupération de l'audio depuis S3, transcription, analyse, et sauvegarde des résultats en base de données.",
        codeSnippets: [
          {
            language: "python",
            code: "import boto3\nimport tempfile\nimport logging\nfrom pathlib import Path\n\nlogger = logging.getLogger(__name__)\ns3 = boto3.client(\"s3\")\n\ndef process_call(bucket: str, audio_key: str, call_metadata: dict) -> dict:\n    # Telecharger l'audio depuis S3\n    with tempfile.NamedTemporaryFile(suffix=\".wav\", delete=False) as tmp:\n        s3.download_file(bucket, audio_key, tmp.name)\n        audio_path = tmp.name\n    try:\n        # Etape 1 : Transcription\n        logger.info(\"Transcription de %s\", audio_key)\n        segments = transcribe_audio(audio_path)\n        transcript_text = format_transcript(segments)\n        # Etape 2 : Recuperer contexte CRM\n        client_context = get_crm_context(call_metadata[\"client_id\"])\n        evaluation_grid = load_evaluation_grid(call_metadata.get(\"type\", \"commercial\"))\n        # Etape 3 : Analyse LLM\n        logger.info(\"Analyse LLM de l'appel\")\n        analysis = analyze_call(transcript_text, evaluation_grid, client_context)\n        # Etape 4 : Sauvegarde\n        result = {\n            \"call_id\": call_metadata[\"call_id\"],\n            \"transcript\": transcript_text,\n            \"segments\": [s.model_dump() for s in segments],\n            \"analysis\": analysis.model_dump()\n        }\n        save_to_database(result)\n        # Etape 5 : Alertes\n        if analysis.churn_risk:\n            send_churn_alert(call_metadata, analysis)\n        if analysis.quality_score < 40:\n            notify_manager(call_metadata, analysis)\n        return result\n    finally:\n        Path(audio_path).unlink(missing_ok=True)",
            filename: "pipeline.py",
          },
        ],
      },
      {
        title: "API et dashboard",
        content:
          "Créez une API REST pour accéder aux analyses et alimenter le dashboard des managers. L'API permet de filtrer par commercial, période et score de qualité.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, Query, BackgroundTasks\nfrom typing import Optional\nfrom datetime import date\n\napp = FastAPI(title=\"Call Analysis Agent\")\n\n@app.post(\"/api/calls/analyze\")\nasync def submit_call(call: dict, bg: BackgroundTasks):\n    bg.add_task(process_call, call[\"bucket\"], call[\"audio_key\"], call[\"metadata\"])\n    return {\"status\": \"processing\", \"call_id\": call[\"metadata\"][\"call_id\"]}\n\n@app.get(\"/api/calls\")\nasync def list_calls(\n    agent: Optional[str] = None,\n    date_from: Optional[date] = None,\n    date_to: Optional[date] = None,\n    min_score: Optional[int] = Query(None, ge=0, le=100)\n):\n    filters = {}\n    if agent:\n        filters[\"agent\"] = agent\n    if date_from:\n        filters[\"date_from\"] = date_from\n    if date_to:\n        filters[\"date_to\"] = date_to\n    if min_score is not None:\n        filters[\"min_score\"] = min_score\n    return await fetch_call_analyses(filters)\n\n@app.get(\"/api/calls/{call_id}\")\nasync def get_call(call_id: str):\n    return await fetch_call_analysis(call_id)\n\n@app.get(\"/api/stats/coaching\")\nasync def coaching_stats(agent: Optional[str] = None):\n    return {\n        \"avg_quality_score\": await avg_quality_by_agent(agent),\n        \"top_objections\": await top_objections(agent),\n        \"sentiment_distribution\": await sentiment_dist(agent),\n        \"improvement_trend\": await quality_trend(agent)\n    }",
            filename: "api.py",
          },
        ],
      },
      {
        title: "Tests et validation",
        content:
          "Testez le pipeline complet avec des enregistrements de test pour valider la qualité de transcription et la pertinence de l'analyse.",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nfrom models import CallAnalysis, Sentiment\nfrom analyzer import analyze_call\n\nSAMPLE_TRANSCRIPT = (\n    \"[00:00] Commercial: Bonjour, merci d'avoir accepte cet appel.\\n\"\n    \"[00:05] Client: Bonjour, j'aimerais en savoir plus sur votre offre Enterprise.\\n\"\n    \"[00:15] Commercial: Bien sur. Quel est votre budget pour ce projet ?\\n\"\n    \"[00:22] Client: Nous avons un budget de 50000 euros annuels.\\n\"\n    \"[00:30] Commercial: Parfait, notre offre Enterprise est a 45000 par an.\\n\"\n    \"[00:40] Client: C'est interessant mais j'ai une objection sur le delai de mise en place.\\n\"\n    \"[00:50] Commercial: Nous garantissons un deploiement en 4 semaines.\"\n)\n\nEVAL_GRID = \"Criteres: accueil, decouverte des besoins, traitement des objections, closing\"\n\ndef test_call_analysis_structure():\n    result = analyze_call(SAMPLE_TRANSCRIPT, EVAL_GRID, \"Client Enterprise, secteur Finance\")\n    assert isinstance(result, CallAnalysis)\n    assert 0 <= result.quality_score <= 100\n    assert len(result.key_points) >= 1\n    assert result.overall_sentiment in list(Sentiment)\n\ndef test_objection_detection():\n    result = analyze_call(SAMPLE_TRANSCRIPT, EVAL_GRID, \"\")\n    assert len(result.objections) >= 1\n    assert any(\"delai\" in obj.text.lower() or \"mise en place\" in obj.text.lower() for obj in result.objections)\n\ndef test_commitment_extraction():\n    result = analyze_call(SAMPLE_TRANSCRIPT, EVAL_GRID, \"\")\n    assert len(result.commitments) >= 0  # Peut ne pas y avoir d'engagement formel\n    assert len(result.next_steps) >= 1",
            filename: "test_analyzer.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les enregistrements audio contiennent des données personnelles sensibles. Les fichiers sont stockés chiffrés sur S3 (SSE-KMS). La transcription est traitée en mémoire et seuls les résumés anonymisés sont envoyés au LLM. Les noms et numéros de compte sont masqués avant l'analyse. Consentement obligatoire des deux parties avant enregistrement (conformité RGPD et CNIL).",
      auditLog: "Traçabilité complète : horodatage de l'appel, durée, participants, hash de l'enregistrement audio, résultat de transcription, résultat d'analyse, score de qualité, actions déclenchées (alertes, notifications). Conservation des enregistrements selon la politique interne (6 mois par défaut). Logs d'accès aux transcriptions.",
      humanInTheLoop: "Les appels avec un score de qualité inférieur à 50 sont escaladés au manager pour revue manuelle. Les alertes churn déclenchent une action du responsable compte. Les commerciaux peuvent contester le score et demander une ré-évaluation. Le manager valide les coaching tips avant partage avec l'agent.",
      monitoring: "Dashboard temps réel : nombre d'appels analysés par jour, score de qualité moyen par agent, tendance de qualité sur 30 jours, top 5 des objections récurrentes, répartition des sentiments, taux de détection de churn, durée moyenne des appels, corrélation score/conversion.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Trigger Webhook (nouvel enregistrement déposé sur S3) → Node AWS S3 (téléchargement audio) → Node HTTP Request (API Whisper transcription) → Node Code (formatage transcription) → Node HTTP Request (API Claude analyse) → Node PostgreSQL (sauvegarde résultats) → Node Switch (score qualité) → Branch score < 50: Node Slack (alerte manager) → Branch churn détecté: Node Email (alerte responsable compte) → Node HTTP Request (mise à jour CRM).",
      nodes: ["Webhook (S3 event)", "AWS S3 (download)", "HTTP Request (Whisper)", "Code (formatage)", "HTTP Request (Claude analyse)", "PostgreSQL (sauvegarde)", "Switch (score)", "Slack (alerte manager)", "Email (alerte churn)", "HTTP Request (CRM update)"],
      triggerType: "Webhook (événement S3 - nouvel enregistrement audio déposé)",
    },
    estimatedTime: "10-16h",
    difficulty: "Expert",
    sectors: ["B2B SaaS", "Assurance", "Banque", "Telecom", "Services"],
    metiers: ["Commercial", "Support Client", "Direction Commerciale", "Formation"],
    functions: ["Sales"],
    metaTitle: "Agent IA d'Analyse des Appels Téléphoniques — Guide Expert",
    metaDescription:
      "Transcrivez et analysez vos appels commerciaux et support avec un agent IA. Whisper pour la transcription, Claude pour l'analyse de sentiment, détection d'objections et coaching. Tutoriel complet.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-generation-rapports",
    title: "Agent de Génération Automatique de Rapports",
    subtitle: "Générez automatiquement des rapports hebdomadaires et mensuels à partir de sources de données multiples",
    problem:
      "Les équipes finance et direction passent des heures chaque semaine à consolider des données provenant de multiples sources (ERP, CRM, comptabilité, RH) pour produire des rapports. Le processus est manuel, sujet aux erreurs de copier-coller, et les rapports arrivent souvent en retard.",
    value:
      "Un agent IA collecte automatiquement les données depuis vos sources, les consolide, détecte les anomalies et les tendances, puis génère un rapport structuré avec des visualisations et des commentaires analytiques en langage naturel. Les rapports sont livrés à l'heure, chaque semaine.",
    inputs: [
      "Données financières (ERP, comptabilité)",
      "Données commerciales (CRM, pipeline)",
      "Données RH (effectifs, absentéisme)",
      "KPIs et objectifs définis par la direction",
      "Modèle de rapport (template configurable)",
      "Rapports précédents pour comparaison",
    ],
    outputs: [
      "Rapport PDF/HTML structuré avec graphiques",
      "Tableau de synthèse des KPIs avec évolution",
      "Commentaires analytiques générés par IA",
      "Alertes sur anomalies et écarts significatifs",
      "Fichier Excel annexe avec données brutes",
    ],
    risks: [
      "Erreurs de calcul ou d'agrégation des données",
      "Interprétation erronée des tendances par le LLM",
      "Indisponibilité d'une source de données bloquant le rapport",
      "Diffusion de données confidentielles si le rapport est mal routé",
    ],
    roiIndicatif:
      "Réduction de 90% du temps de préparation des rapports. Livraison systématique à l'heure (vs 60% avant). Détection automatique de 30% d'anomalies supplémentaires grâce à l'analyse IA.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "WeasyPrint", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
    ],
    architectureDiagram: "+-----------+  +-----------+  +-----------+\n|   ERP     |  |   CRM     |  |   RH      |\n+-----+-----+  +-----+-----+  +-----+-----+\n      |              |              |\n      v              v              v\n+------------------------------------------+\n|        Agent LLM (Consolidation           |\n|        + Analyse + Rédaction)             |\n+---------------------+--------------------+\n                      |\n              +-------v--------+\n              |  Rapport PDF   |\n              |  + Email auto  |\n              +----------------+",
    tutorial: [
      {
        title: "Prérequis et installation",
        content:
          "Installez les bibliothèques pour la connexion aux sources de données, la génération de graphiques et la création de PDF. Configurez les accès aux différentes APIs.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain pandas matplotlib weasyprint jinja2 sqlalchemy psycopg2-binary requests schedule",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Connecteurs de données",
        content:
          "Créez des connecteurs pour chaque source de données. Chaque connecteur implémente une interface commune et retourne un DataFrame pandas standardisé.",
        codeSnippets: [
          {
            language: "python",
            code: "import pandas as pd\nfrom abc import ABC, abstractmethod\nfrom sqlalchemy import create_engine\nimport requests\n\nclass DataConnector(ABC):\n    @abstractmethod\n    def fetch_data(self, date_from: str, date_to: str) -> pd.DataFrame:\n        pass\n\nclass ERPConnector(DataConnector):\n    def __init__(self, connection_string: str):\n        self.engine = create_engine(connection_string)\n\n    def fetch_data(self, date_from: str, date_to: str) -> pd.DataFrame:\n        query = (\n            \"SELECT date_comptable, compte, libelle, montant_debit, montant_credit \"\n            \"FROM ecritures_comptables \"\n            \"WHERE date_comptable BETWEEN '{}' AND '{}'\"\n        ).format(date_from, date_to)\n        return pd.read_sql(query, self.engine)\n\nclass CRMConnector(DataConnector):\n    def __init__(self, api_url: str, api_key: str):\n        self.api_url = api_url\n        self.headers = {\"Authorization\": \"Bearer \" + api_key}\n\n    def fetch_data(self, date_from: str, date_to: str) -> pd.DataFrame:\n        response = requests.get(\n            self.api_url + \"/deals\",\n            headers=self.headers,\n            params={\"date_from\": date_from, \"date_to\": date_to}\n        )\n        response.raise_for_status()\n        return pd.DataFrame(response.json()[\"deals\"])",
            filename: "connectors.py",
          },
        ],
      },
      {
        title: "Agrégation et calcul des KPIs",
        content:
          "Consolidez les données de toutes les sources et calculez les KPIs définis. Comparez avec la période précédente pour détecter les tendances et anomalies.",
        codeSnippets: [
          {
            language: "python",
            code: "import pandas as pd\nfrom dataclasses import dataclass\nfrom typing import Optional\n\n@dataclass\nclass KPI:\n    name: str\n    value: float\n    previous_value: Optional[float]\n    target: Optional[float]\n    unit: str\n\n    @property\n    def variation_pct(self) -> Optional[float]:\n        if self.previous_value and self.previous_value != 0:\n            return ((self.value - self.previous_value) / abs(self.previous_value)) * 100\n        return None\n\n    @property\n    def is_on_target(self) -> Optional[bool]:\n        if self.target:\n            return self.value >= self.target\n        return None\n\ndef compute_financial_kpis(erp_data: pd.DataFrame, previous_erp: pd.DataFrame, targets: dict) -> list[KPI]:\n    ca_current = erp_data[erp_data[\"compte\"].str.startswith(\"70\")][\"montant_credit\"].sum()\n    ca_previous = previous_erp[previous_erp[\"compte\"].str.startswith(\"70\")][\"montant_credit\"].sum()\n    charges_current = erp_data[erp_data[\"compte\"].str.startswith(\"6\")][\"montant_debit\"].sum()\n    charges_previous = previous_erp[previous_erp[\"compte\"].str.startswith(\"6\")][\"montant_debit\"].sum()\n    marge = ca_current - charges_current\n    return [\n        KPI(\"Chiffre d'affaires\", ca_current, ca_previous, targets.get(\"ca\"), \"EUR\"),\n        KPI(\"Charges totales\", charges_current, charges_previous, None, \"EUR\"),\n        KPI(\"Marge brute\", marge, ca_previous - charges_previous, targets.get(\"marge\"), \"EUR\"),\n        KPI(\"Taux de marge\", (marge / ca_current * 100) if ca_current else 0, None, targets.get(\"taux_marge\"), \"%\"),\n    ]",
            filename: "kpis.py",
          },
        ],
      },
      {
        title: "Analyse et commentaires par le LLM",
        content:
          "Envoyez les KPIs calculés et les données agrégées au LLM pour générer des commentaires analytiques en langage naturel. L'agent identifie les tendances, les risques et les recommandations.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nimport json\n\nclient = anthropic.Anthropic()\n\ndef generate_analysis(kpis: list[KPI], raw_data_summary: str, previous_report_summary: str) -> dict:\n    kpi_text = \"\\n\".join(\n        \"{name}: {value} {unit} (variation: {var}%, cible: {target})\".format(\n            name=k.name, value=k.value, unit=k.unit,\n            var=round(k.variation_pct, 1) if k.variation_pct else \"N/A\",\n            target=k.target or \"N/A\"\n        )\n        for k in kpis\n    )\n    prompt = (\n        \"Tu es un analyste financier senior. Analyse ces KPIs et genere :\\n\"\n        \"1. Un commentaire executif (3-5 phrases)\\n\"\n        \"2. Les points positifs (max 3)\\n\"\n        \"3. Les points d'attention (max 3)\\n\"\n        \"4. Les recommandations (max 3)\\n\"\n        \"5. Les anomalies detectees\\n\\n\"\n        \"KPIs:\\n{kpis}\\n\\n\"\n        \"Resume des donnees:\\n{data}\\n\\n\"\n        \"Rapport precedent:\\n{previous}\\n\\n\"\n        \"Retourne un JSON structure.\"\n    ).format(kpis=kpi_text, data=raw_data_summary, previous=previous_report_summary)\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=4096,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    return json.loads(message.content[0].text)",
            filename: "analysis.py",
          },
        ],
      },
      {
        title: "Génération du rapport PDF",
        content:
          "Utilisez Jinja2 et WeasyPrint pour générer un rapport PDF professionnel intégrant les KPIs, graphiques et commentaires analytiques.",
        codeSnippets: [
          {
            language: "python",
            code: "from jinja2 import Environment, FileSystemLoader\nfrom weasyprint import HTML\nimport matplotlib\nmatplotlib.use(\"Agg\")\nimport matplotlib.pyplot as plt\nimport io\nimport base64\n\ndef create_kpi_chart(kpis: list[KPI]) -> str:\n    fig, ax = plt.subplots(figsize=(10, 4))\n    names = [k.name for k in kpis if k.unit == \"EUR\"]\n    values = [k.value for k in kpis if k.unit == \"EUR\"]\n    prev_values = [k.previous_value or 0 for k in kpis if k.unit == \"EUR\"]\n    x = range(len(names))\n    ax.bar([i - 0.2 for i in x], prev_values, 0.4, label=\"Precedent\", color=\"#94a3b8\")\n    ax.bar([i + 0.2 for i in x], values, 0.4, label=\"Actuel\", color=\"#3b82f6\")\n    ax.set_xticks(list(x))\n    ax.set_xticklabels(names, rotation=15)\n    ax.legend()\n    ax.set_title(\"Comparaison des KPIs financiers\")\n    buf = io.BytesIO()\n    fig.savefig(buf, format=\"png\", bbox_inches=\"tight\")\n    plt.close(fig)\n    return base64.b64encode(buf.getvalue()).decode()\n\ndef generate_pdf_report(kpis: list[KPI], analysis: dict, chart_b64: str, period: str) -> bytes:\n    env = Environment(loader=FileSystemLoader(\"templates\"))\n    template = env.get_template(\"report.html\")\n    html_content = template.render(\n        period=period, kpis=kpis, analysis=analysis, chart_image=chart_b64\n    )\n    return HTML(string=html_content).write_pdf()",
            filename: "report_generator.py",
          },
        ],
      },
      {
        title: "Orchestration et planification",
        content:
          "Planifiez la génération automatique des rapports avec un scheduler. Le pipeline complet s'exécute à heure fixe et envoie le rapport par email aux destinataires configurés.",
        codeSnippets: [
          {
            language: "python",
            code: "import schedule\nimport time\nimport smtplib\nfrom email.mime.multipart import MIMEMultipart\nfrom email.mime.base import MIMEBase\nfrom email.mime.text import MIMEText\nfrom email import encoders\nfrom datetime import datetime, timedelta\n\ndef run_weekly_report(config: dict):\n    date_to = datetime.now().strftime(\"%Y-%m-%d\")\n    date_from = (datetime.now() - timedelta(days=7)).strftime(\"%Y-%m-%d\")\n    prev_from = (datetime.now() - timedelta(days=14)).strftime(\"%Y-%m-%d\")\n    prev_to = date_from\n    # Collecte des donnees\n    erp = ERPConnector(config[\"erp_dsn\"]).fetch_data(date_from, date_to)\n    prev_erp = ERPConnector(config[\"erp_dsn\"]).fetch_data(prev_from, prev_to)\n    crm = CRMConnector(config[\"crm_url\"], config[\"crm_key\"]).fetch_data(date_from, date_to)\n    # Calcul KPIs\n    kpis = compute_financial_kpis(erp, prev_erp, config[\"targets\"])\n    # Analyse LLM\n    analysis = generate_analysis(kpis, erp.describe().to_string(), \"\")\n    # Graphique\n    chart = create_kpi_chart(kpis)\n    # Generation PDF\n    pdf_bytes = generate_pdf_report(kpis, analysis, chart, date_from + \" au \" + date_to)\n    # Envoi email\n    send_report_email(config[\"recipients\"], pdf_bytes, date_from + \" au \" + date_to)\n\ndef send_report_email(recipients: list[str], pdf_bytes: bytes, period: str):\n    msg = MIMEMultipart()\n    msg[\"Subject\"] = \"Rapport hebdomadaire - \" + period\n    msg[\"From\"] = \"rapports@entreprise.fr\"\n    msg[\"To\"] = \", \".join(recipients)\n    msg.attach(MIMEText(\"Veuillez trouver ci-joint le rapport hebdomadaire.\", \"plain\"))\n    attachment = MIMEBase(\"application\", \"pdf\")\n    attachment.set_payload(pdf_bytes)\n    encoders.encode_base64(attachment)\n    attachment.add_header(\"Content-Disposition\", \"attachment\", filename=\"rapport.pdf\")\n    msg.attach(attachment)\n    with smtplib.SMTP_SSL(\"smtp.entreprise.fr\", 465) as server:\n        server.login(\"rapports@entreprise.fr\", \"password\")\n        server.send_message(msg)\n\nschedule.every().monday.at(\"08:00\").do(run_weekly_report, config=CONFIG)\n\nwhile True:\n    schedule.run_pending()\n    time.sleep(60)",
            filename: "scheduler.py",
          },
        ],
      },
      {
        title: "Tests et validation",
        content:
          "Validez le pipeline de bout en bout avec des données de test. Vérifiez le calcul des KPIs, la qualité de l'analyse LLM et la génération correcte du PDF.",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nimport pandas as pd\nfrom kpis import KPI, compute_financial_kpis\nfrom analysis import generate_analysis\n\ndef test_kpi_variation():\n    kpi = KPI(\"CA\", 120000, 100000, 110000, \"EUR\")\n    assert kpi.variation_pct == 20.0\n    assert kpi.is_on_target is True\n\ndef test_kpi_no_previous():\n    kpi = KPI(\"CA\", 120000, None, 110000, \"EUR\")\n    assert kpi.variation_pct is None\n\ndef test_financial_kpis_computation():\n    erp_current = pd.DataFrame({\n        \"compte\": [\"701000\", \"701000\", \"601000\", \"602000\"],\n        \"montant_credit\": [50000, 70000, 0, 0],\n        \"montant_debit\": [0, 0, 30000, 20000],\n        \"libelle\": [\"Vente A\", \"Vente B\", \"Achat X\", \"Achat Y\"],\n        \"date_comptable\": [\"2025-02-01\"] * 4\n    })\n    erp_previous = pd.DataFrame({\n        \"compte\": [\"701000\", \"601000\"],\n        \"montant_credit\": [90000, 0],\n        \"montant_debit\": [0, 40000],\n        \"libelle\": [\"Vente\", \"Achat\"],\n        \"date_comptable\": [\"2025-01-25\"] * 2\n    })\n    kpis = compute_financial_kpis(erp_current, erp_previous, {\"ca\": 100000})\n    assert kpis[0].name == \"Chiffre d'affaires\"\n    assert kpis[0].value == 120000\n    assert kpis[0].is_on_target is True\n\ndef test_llm_analysis_structure():\n    kpis = [\n        KPI(\"CA\", 120000, 100000, 110000, \"EUR\"),\n        KPI(\"Marge\", 70000, 60000, 65000, \"EUR\"),\n    ]\n    result = generate_analysis(kpis, \"Donnees resume test\", \"Rapport precedent OK\")\n    assert \"commentaire\" in result or \"executive_summary\" in result\n    assert isinstance(result, dict)",
            filename: "test_reports.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données financières sont sensibles et confidentielles. Seules des données agrégées (KPIs, totaux par catégorie) sont envoyées au LLM, jamais les écritures comptables détaillées ni les noms de clients. Les rapports générés sont chiffrés et l'accès est contrôlé par rôle (RBAC). Les données transitent exclusivement via des canaux chiffrés (TLS 1.3).",
      auditLog: "Chaque exécution du pipeline génère un log complet : horodatage, sources interrogées, nombre de lignes collectées, KPIs calculés, modèle LLM utilisé, tokens consommés, rapport généré (hash SHA-256), destinataires notifiés. Les logs sont conservés 5 ans pour conformité comptable.",
      humanInTheLoop: "Le rapport est envoyé en mode brouillon au DAF ou contrôleur de gestion pour validation avant diffusion au comité de direction. Les anomalies détectées avec un score de confiance inférieur à 0.8 sont signalées pour vérification manuelle. Un workflow d'approbation permet de corriger et republier le rapport.",
      monitoring: "Dashboard de suivi : taux de succès des exécutions, temps de génération, nombre de sources connectées, volume de données traitées, tokens LLM consommés, nombre de rapports générés par semaine, taux d'anomalies détectées, feedback des destinataires (utile/inutile).",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (lundi 8h) → Node PostgreSQL (données ERP) → Node HTTP Request (données CRM) → Node Code (calcul KPIs et agrégation) → Node HTTP Request (API Claude analyse) → Node Code (génération graphiques matplotlib) → Node Code (génération HTML Jinja2) → Node HTTP Request (WeasyPrint PDF) → Node Email (envoi rapport aux destinataires) → Node PostgreSQL (log d'audit).",
      nodes: ["Cron Trigger (lundi 8h)", "PostgreSQL (ERP data)", "HTTP Request (CRM data)", "Code (calcul KPIs)", "HTTP Request (Claude analyse)", "Code (graphiques)", "Code (HTML Jinja2)", "HTTP Request (PDF)", "Email (envoi rapport)", "PostgreSQL (audit log)"],
      triggerType: "Cron Trigger (planification hebdomadaire lundi 8h00)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Finance", "Services", "Industrie", "Retail", "Technologie"],
    metiers: ["Direction Financière", "Contrôle de Gestion", "Direction Générale", "Comptabilité"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Génération Automatique de Rapports — Guide Complet",
    metaDescription:
      "Automatisez la génération de vos rapports financiers hebdomadaires et mensuels avec un agent IA. Consolidation multi-sources, analyse intelligente et PDF automatique. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-gestion-faq-dynamique",
    title: "Agent de Gestion de FAQ Dynamique",
    subtitle: "Mettez à jour automatiquement votre FAQ à partir des tickets de support et détectez les nouvelles questions émergentes",
    problem:
      "Les FAQ deviennent rapidement obsolètes car leur mise à jour est manuelle. Les nouvelles questions récurrentes ne sont pas détectées à temps, les clients ne trouvent pas de réponses à jour, et le volume de tickets augmente inutilement sur des sujets déjà documentés mais mal référencés.",
    value:
      "Un agent IA analyse en continu les tickets de support entrants, identifie les questions récurrentes non couvertes par la FAQ, génère automatiquement de nouvelles entrées, et propose la mise à jour des réponses existantes devenues obsolètes. Le taux de self-service augmente significativement.",
    inputs: [
      "Tickets de support résolus (texte question + réponse)",
      "FAQ existante (questions, réponses, catégories)",
      "Base de connaissances interne",
      "Logs de recherche sur le site (requêtes sans résultat)",
      "Feedback utilisateurs sur les articles FAQ",
    ],
    outputs: [
      "Nouvelles entrées FAQ générées (question + réponse)",
      "Mises à jour proposées pour les entrées existantes",
      "Rapport de détection de questions émergentes",
      "Score de couverture FAQ (% de sujets couverts)",
      "Entrées FAQ à archiver (obsolètes)",
    ],
    risks: [
      "Génération de réponses incorrectes ou imprécises dans la FAQ",
      "Doublons de questions formulées différemment",
      "Perte de cohérence de ton entre entrées manuelles et générées",
      "Publication automatique d'informations erronées sans validation",
    ],
    roiIndicatif:
      "Augmentation de 50% du taux de self-service. Réduction de 35% du volume de tickets de niveau 1. FAQ toujours à jour avec un effort de maintenance réduit de 80%.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL + pgvector", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Small", category: "LLM", isFree: false },
      { name: "ChromaDB", category: "Database", isFree: true },
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "Make.com", category: "Orchestration", isFree: false },
    ],
    architectureDiagram: "+------------------+     +------------------+     +------------------+\n|   Tickets        |---->|   Agent LLM      |---->|   FAQ CMS        |\n|   Support        |     |   (Analyse +     |     |   (Publication)  |\n+------------------+     |   Generation)    |     +------------------+\n                          +--------+---------+\n+------------------+               |\n|   Recherches     |------->-------+\n|   sans resultat  |\n+------------------+",
    tutorial: [
      {
        title: "Prérequis et installation",
        content:
          "Installez les dépendances nécessaires pour l'analyse sémantique des tickets, la recherche vectorielle et la génération de contenu FAQ.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain pgvector psycopg2-binary sentence-transformers pydantic fastapi numpy",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modèles de données",
        content:
          "Définissez les structures pour les entrées FAQ, les clusters de questions et les propositions de mises à jour. Le modèle inclut le statut de validation et le score de pertinence.",
        codeSnippets: [
          {
            language: "python",
            code: "from pydantic import BaseModel, Field\nfrom enum import Enum\nfrom typing import Optional\nfrom datetime import datetime\n\nclass FAQStatus(str, Enum):\n    DRAFT = \"brouillon\"\n    PENDING_REVIEW = \"en_attente_validation\"\n    PUBLISHED = \"publie\"\n    ARCHIVED = \"archive\"\n\nclass FAQEntry(BaseModel):\n    id: Optional[str] = None\n    question: str\n    answer: str\n    category: str\n    tags: list[str]\n    status: FAQStatus = FAQStatus.DRAFT\n    relevance_score: float = Field(ge=0.0, le=1.0)\n    source_ticket_ids: list[str] = []\n    created_at: Optional[datetime] = None\n    updated_at: Optional[datetime] = None\n\nclass QuestionCluster(BaseModel):\n    representative_question: str\n    similar_questions: list[str]\n    ticket_count: int\n    existing_faq_match: Optional[str] = None\n    match_score: float = Field(ge=0.0, le=1.0, default=0.0)\n\nclass FAQUpdateProposal(BaseModel):\n    action: str  # \"create\", \"update\", \"archive\"\n    entry: FAQEntry\n    reason: str\n    confidence: float = Field(ge=0.0, le=1.0)",
            filename: "models.py",
          },
        ],
      },
      {
        title: "Détection de questions récurrentes",
        content:
          "Utilisez les embeddings pour regrouper les questions similaires provenant des tickets de support. Les clusters de questions permettent d'identifier les sujets récurrents non couverts par la FAQ.",
        codeSnippets: [
          {
            language: "python",
            code: "import numpy as np\nfrom sentence_transformers import SentenceTransformer\nfrom sklearn.cluster import DBSCAN\n\nembedder = SentenceTransformer(\"paraphrase-multilingual-MiniLM-L12-v2\")\n\ndef cluster_questions(tickets: list[dict], eps: float = 0.3) -> list[QuestionCluster]:\n    questions = [t[\"subject\"] + \" \" + t[\"body\"][:200] for t in tickets]\n    embeddings = embedder.encode(questions, normalize_embeddings=True)\n    clustering = DBSCAN(eps=eps, min_samples=3, metric=\"cosine\").fit(embeddings)\n    clusters = []\n    for label in set(clustering.labels_):\n        if label == -1:\n            continue\n        indices = np.where(clustering.labels_ == label)[0]\n        cluster_questions_list = [questions[i] for i in indices]\n        # Choisir la question la plus representative (proche du centroide)\n        centroid = embeddings[indices].mean(axis=0)\n        distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)\n        representative_idx = indices[np.argmin(distances)]\n        clusters.append(QuestionCluster(\n            representative_question=questions[representative_idx],\n            similar_questions=cluster_questions_list[:10],\n            ticket_count=len(indices)\n        ))\n    return sorted(clusters, key=lambda c: c.ticket_count, reverse=True)",
            filename: "clustering.py",
          },
        ],
      },
      {
        title: "Matching avec la FAQ existante",
        content:
          "Comparez chaque cluster de questions avec les entrées FAQ existantes pour identifier les lacunes (questions sans réponse) et les mises à jour nécessaires.",
        codeSnippets: [
          {
            language: "python",
            code: "from pgvector.psycopg2 import register_vector\nimport psycopg2\n\ndef match_clusters_to_faq(clusters: list[QuestionCluster], db_config: dict) -> list[QuestionCluster]:\n    conn = psycopg2.connect(**db_config)\n    register_vector(conn)\n    cur = conn.cursor()\n    for cluster in clusters:\n        embedding = embedder.encode([cluster.representative_question], normalize_embeddings=True)[0]\n        cur.execute(\n            \"SELECT id, question, 1 - (embedding <=> %s::vector) as similarity \"\n            \"FROM faq_entries WHERE status = 'publie' \"\n            \"ORDER BY embedding <=> %s::vector LIMIT 1\",\n            (embedding.tolist(), embedding.tolist())\n        )\n        result = cur.fetchone()\n        if result and result[2] > 0.75:\n            cluster.existing_faq_match = result[1]\n            cluster.match_score = float(result[2])\n        else:\n            cluster.match_score = 0.0\n    conn.close()\n    return clusters\n\ndef identify_gaps(clusters: list[QuestionCluster]) -> list[QuestionCluster]:\n    return [c for c in clusters if c.match_score < 0.75 and c.ticket_count >= 5]",
            filename: "matcher.py",
          },
        ],
      },
      {
        title: "Génération de contenu FAQ par le LLM",
        content:
          "Pour chaque lacune identifiée, générez automatiquement une entrée FAQ avec question formatée, réponse complète, catégorie et tags. Le LLM utilise les tickets résolus comme source de vérité.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\n\nclient = anthropic.Anthropic()\n\ndef generate_faq_entry(cluster: QuestionCluster, resolved_tickets: list[dict], existing_faq: list[dict]) -> FAQEntry:\n    tickets_text = \"\\n---\\n\".join(\n        \"Q: {q}\\nR: {r}\".format(q=t[\"subject\"], r=t[\"resolution\"][:500])\n        for t in resolved_tickets[:5]\n    )\n    faq_context = \"\\n\".join(\n        \"- {q}\".format(q=f[\"question\"]) for f in existing_faq[:20]\n    )\n    prompt = (\n        \"Tu es un redacteur de FAQ professionnel.\\n\"\n        \"A partir des tickets de support resolus ci-dessous, genere une entree FAQ.\\n\\n\"\n        \"Tickets resolus sur ce sujet:\\n{tickets}\\n\\n\"\n        \"FAQ existante (pour eviter les doublons):\\n{faq}\\n\\n\"\n        \"Question representative: {question}\\n\\n\"\n        \"Genere un JSON avec: question (reformulee clairement), \"\n        \"answer (reponse complete et structuree), category, tags (liste), \"\n        \"relevance_score (0-1 selon la pertinence)\"\n    ).format(\n        tickets=tickets_text,\n        faq=faq_context,\n        question=cluster.representative_question\n    )\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=2048,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    data = FAQEntry.model_validate_json(message.content[0].text)\n    data.source_ticket_ids = [t[\"id\"] for t in resolved_tickets[:5]]\n    data.status = FAQStatus.PENDING_REVIEW\n    return data",
            filename: "generator.py",
          },
        ],
      },
      {
        title: "Pipeline complet et API",
        content:
          "Orchestrez le pipeline complet de détection, matching et génération. Exposez une API pour consulter les propositions et les valider via un workflow d'approbation.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, Query\nfrom typing import Optional\n\napp = FastAPI(title=\"Dynamic FAQ Agent\")\n\ndef run_faq_pipeline(config: dict) -> list[FAQUpdateProposal]:\n    # Recuperer les tickets recents non traites\n    tickets = fetch_recent_tickets(config[\"helpdesk_api\"], days=7)\n    # Regrouper les questions similaires\n    clusters = cluster_questions(tickets)\n    # Comparer avec la FAQ existante\n    clusters = match_clusters_to_faq(clusters, config[\"db\"])\n    # Identifier les lacunes\n    gaps = identify_gaps(clusters)\n    proposals = []\n    for cluster in gaps:\n        resolved = get_resolved_tickets_for_cluster(cluster, tickets)\n        entry = generate_faq_entry(cluster, resolved, fetch_existing_faq(config[\"db\"]))\n        proposals.append(FAQUpdateProposal(\n            action=\"create\",\n            entry=entry,\n            reason=\"{count} tickets sur ce sujet sans entree FAQ correspondante\".format(count=cluster.ticket_count),\n            confidence=entry.relevance_score\n        ))\n    save_proposals(proposals, config[\"db\"])\n    return proposals\n\n@app.get(\"/api/faq/proposals\")\nasync def list_proposals(status: Optional[str] = Query(None)):\n    return await fetch_proposals(status)\n\n@app.post(\"/api/faq/proposals/{proposal_id}/approve\")\nasync def approve_proposal(proposal_id: str):\n    proposal = await get_proposal(proposal_id)\n    await publish_faq_entry(proposal.entry)\n    await update_proposal_status(proposal_id, \"approved\")\n    return {\"status\": \"published\"}\n\n@app.post(\"/api/faq/proposals/{proposal_id}/reject\")\nasync def reject_proposal(proposal_id: str, reason: str = \"\"):\n    await update_proposal_status(proposal_id, \"rejected\", reason)\n    return {\"status\": \"rejected\"}\n\n@app.get(\"/api/faq/coverage\")\nasync def faq_coverage():\n    return {\n        \"total_topics_detected\": await count_topic_clusters(),\n        \"topics_covered\": await count_covered_topics(),\n        \"coverage_rate\": await compute_coverage_rate(),\n        \"top_uncovered_topics\": await top_uncovered(limit=10)\n    }",
            filename: "api.py",
          },
        ],
      },
      {
        title: "Tests et validation",
        content:
          "Testez le clustering de questions, le matching avec la FAQ existante et la qualité des entrées générées. Validez que les doublons sont correctement détectés.",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nfrom models import QuestionCluster, FAQEntry, FAQStatus\nfrom clustering import cluster_questions\nfrom generator import generate_faq_entry\n\ndef test_question_clustering():\n    tickets = [\n        {\"subject\": \"Comment reinitialiser mon mot de passe ?\", \"body\": \"Je n'arrive plus a me connecter\"},\n        {\"subject\": \"Mot de passe oublie\", \"body\": \"J'ai oublie mon mot de passe\"},\n        {\"subject\": \"Reset password\", \"body\": \"Comment changer mon mot de passe\"},\n        {\"subject\": \"Probleme connexion mot de passe\", \"body\": \"Mon mot de passe ne fonctionne plus\"},\n        {\"subject\": \"Facture introuvable\", \"body\": \"Je ne trouve pas ma facture\"},\n    ]\n    clusters = cluster_questions(tickets, eps=0.4)\n    # Les questions sur le mot de passe doivent etre regroupees\n    password_cluster = [c for c in clusters if \"mot de passe\" in c.representative_question.lower() or \"password\" in c.representative_question.lower()]\n    assert len(password_cluster) >= 1\n    assert password_cluster[0].ticket_count >= 3\n\ndef test_faq_generation_quality():\n    cluster = QuestionCluster(\n        representative_question=\"Comment reinitialiser mon mot de passe ?\",\n        similar_questions=[\"Mot de passe oublie\", \"Reset password\"],\n        ticket_count=15\n    )\n    resolved = [\n        {\"id\": \"T001\", \"subject\": \"Mot de passe oublie\", \"resolution\": \"Allez sur la page de connexion, cliquez sur Mot de passe oublie, entrez votre email, suivez le lien recu.\"},\n        {\"id\": \"T002\", \"subject\": \"Reset password\", \"resolution\": \"Utilisez le lien de reinitialisation disponible sur la page login.\"},\n    ]\n    entry = generate_faq_entry(cluster, resolved, [])\n    assert isinstance(entry, FAQEntry)\n    assert entry.status == FAQStatus.PENDING_REVIEW\n    assert len(entry.answer) > 50\n    assert len(entry.tags) >= 1\n    assert entry.relevance_score >= 0.5",
            filename: "test_faq.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les tickets de support peuvent contenir des données personnelles (noms, emails, numéros de commande). Le contenu est anonymisé avant envoi au LLM : les identifiants personnels sont remplacés par des placeholders. Les entrées FAQ générées ne contiennent jamais de données spécifiques à un client. Les embeddings vectoriels ne permettent pas de reconstituer le texte original.",
      auditLog: "Traçabilité de chaque proposition : horodatage, tickets sources, cluster identifié, entrée FAQ générée, valideur, date de publication ou rejet, motif de rejet le cas échéant. Historique des modifications de chaque entrée FAQ. Conservation des logs pendant 2 ans.",
      humanInTheLoop: "Toutes les entrées FAQ générées passent par un statut 'en attente de validation' avant publication. Un expert métier valide le contenu, corrige si nécessaire, et approuve la publication. Les mises à jour d'entrées existantes sont signalées au responsable documentation. Un workflow Slack/Email notifie les valideurs des nouvelles propositions.",
      monitoring: "Dashboard de suivi : nombre de clusters détectés par semaine, taux de couverture FAQ, nombre de propositions générées/approuvées/rejetées, taux de self-service (tickets évités), top 10 des recherches sans résultat, score de pertinence moyen des entrées générées, feedback utilisateurs par article.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien 6h) → Node HTTP Request (API Helpdesk - tickets résolus) → Node Code (extraction questions et clustering) → Node HTTP Request (API pgvector - matching FAQ) → Node Switch (gap détecté ?) → Branch gap: Node HTTP Request (API Claude - génération FAQ) → Node PostgreSQL (sauvegarde proposition) → Node Slack (notification validateur) → Branch pas de gap: Node PostgreSQL (log audit) → Node HTTP Request (mise à jour score de couverture).",
      nodes: ["Cron Trigger (quotidien)", "HTTP Request (Helpdesk API)", "Code (clustering)", "HTTP Request (pgvector matching)", "Switch (gap?)", "HTTP Request (Claude génération)", "PostgreSQL (propositions)", "Slack (notification)", "PostgreSQL (audit)", "HTTP Request (coverage update)"],
      triggerType: "Cron Trigger (exécution quotidienne à 6h00)",
    },
    estimatedTime: "4-6h",
    difficulty: "Facile",
    sectors: ["B2B SaaS", "E-commerce", "Services", "Technologie", "Telecom"],
    metiers: ["Support Client", "Documentation", "Product Management"],
    functions: ["Support"],
    metaTitle: "Agent IA de Gestion de FAQ Dynamique — Guide Complet",
    metaDescription:
      "Automatisez la mise à jour de votre FAQ avec un agent IA. Détection de questions récurrentes, génération automatique d'entrées et workflow de validation. Tutoriel pas-à-pas.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-scoring-risque-credit",
    title: "Agent de Scoring de Risque Crédit",
    subtitle: "Évaluez automatiquement le risque crédit de vos clients avec une analyse IA multi-sources",
    problem:
      "L'évaluation du risque crédit repose sur des modèles statistiques rigides et des analyses manuelles chronophages. Les analystes crédit passent des heures à compiler des données provenant de multiples sources (bilans, flux bancaires, données sectorielles) et les décisions sont souvent retardées, ce qui impacte la relation commerciale.",
    value:
      "Un agent IA agrège automatiquement les données financières multi-sources, analyse les bilans et comptes de résultat, intègre les signaux faibles (actualités, contentieux, évolution sectorielle), et produit un score de risque argumenté avec des recommandations. Le temps de décision passe de plusieurs jours à quelques minutes.",
    inputs: [
      "Bilans et comptes de résultat (3 derniers exercices)",
      "Données Banque de France (cotation, incidents de paiement)",
      "Flux bancaires et relevés de compte",
      "Données sectorielles et benchmarks",
      "Informations légales (Kbis, dirigeants, contentieux)",
      "Données internes (historique de paiement, encours)",
    ],
    outputs: [
      "Score de risque crédit (0-1000) avec grade (A à E)",
      "Analyse détaillée des ratios financiers",
      "Synthèse des points forts et points de vigilance",
      "Recommandation de limite de crédit",
      "Plan de surveillance (fréquence de revue, alertes)",
      "Rapport PDF conforme aux exigences réglementaires",
    ],
    risks: [
      "Erreurs d'analyse pouvant mener à des pertes financières significatives",
      "Biais algorithmique discriminant certaines catégories d'entreprises",
      "Non-conformité avec les réglementations bancaires (Bâle III/IV, EBA)",
      "Hallucination du LLM sur des données financières critiques",
      "Dépendance à la qualité des données sources",
    ],
    roiIndicatif:
      "Réduction de 75% du temps d'analyse par dossier. Diminution de 20% du taux de défaut grâce à la détection de signaux faibles. Augmentation de 30% du volume de dossiers traités sans recrutement.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS (Lambda + S3)", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "Evidently AI", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "Ollama + Mixtral", category: "LLM", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
    ],
    architectureDiagram: "+-----------+  +-----------+  +-----------+  +-----------+\n|  Bilans   |  |  Banque   |  |  Donnees  |  |  Donnees  |\n|  Comptes  |  |  de France|  |  Legales  |  |  Internes |\n+-----+-----+  +-----+-----+  +-----+-----+  +-----+-----+\n      |              |              |              |\n      v              v              v              v\n+------------------------------------------------------+\n|            Agent LLM (Analyse + Scoring)              |\n|         + Modele Statistique (XGBoost)                |\n+----------------------------+-------------------------+\n                             |\n                     +-------v--------+\n                     |  Score + Rapport|\n                     |  + Alerte       |\n                     +----------------+",
    tutorial: [
      {
        title: "Prérequis et installation",
        content:
          "Installez les bibliothèques nécessaires pour l'analyse financière, le scoring statistique et la génération de rapports. Configurez les accès aux API de données financières.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain pandas numpy scikit-learn xgboost pydantic fastapi weasyprint jinja2 psycopg2-binary requests",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modèles de données financières",
        content:
          "Définissez les structures pour les données financières, les ratios et le résultat du scoring. Ces modèles garantissent la cohérence et la traçabilité de chaque évaluation.",
        codeSnippets: [
          {
            language: "python",
            code: "from pydantic import BaseModel, Field\nfrom enum import Enum\nfrom typing import Optional\nfrom datetime import date\n\nclass RiskGrade(str, Enum):\n    A = \"A\"  # Risque tres faible\n    B = \"B\"  # Risque faible\n    C = \"C\"  # Risque modere\n    D = \"D\"  # Risque eleve\n    E = \"E\"  # Risque tres eleve\n\nclass FinancialRatios(BaseModel):\n    ratio_endettement: float  # Dettes / Capitaux propres\n    ratio_liquidite: float  # Actifs circulants / Passifs circulants\n    ratio_solvabilite: float  # Capitaux propres / Total bilan\n    marge_nette: float  # Resultat net / CA\n    rotation_stocks: float  # CA / Stocks moyens\n    delai_paiement_clients: float  # (Creances clients / CA) * 365\n    delai_paiement_fournisseurs: float  # (Dettes fournisseurs / Achats) * 365\n    capacite_autofinancement: float\n    taux_croissance_ca: float\n\nclass CreditRiskScore(BaseModel):\n    score: int = Field(ge=0, le=1000)\n    grade: RiskGrade\n    financial_ratios: FinancialRatios\n    strengths: list[str] = Field(min_length=1, max_length=5)\n    warnings: list[str] = Field(max_length=5)\n    recommended_credit_limit: float\n    recommended_payment_terms: int  # en jours\n    review_frequency: str  # \"mensuel\", \"trimestriel\", \"annuel\"\n    detailed_analysis: str\n    confidence: float = Field(ge=0.0, le=1.0)\n    model_version: str = \"1.0\"",
            filename: "models.py",
          },
        ],
      },
      {
        title: "Calcul des ratios financiers",
        content:
          "Extrayez et calculez les ratios financiers clés à partir des bilans et comptes de résultat. Ces ratios alimentent à la fois le modèle statistique et l'analyse LLM.",
        codeSnippets: [
          {
            language: "python",
            code: "import pandas as pd\nfrom models import FinancialRatios\n\ndef compute_ratios(bilan: dict, compte_resultat: dict) -> FinancialRatios:\n    # Extraction des postes cles\n    capitaux_propres = bilan.get(\"capitaux_propres\", 0)\n    total_bilan = bilan.get(\"total_actif\", 1)\n    dettes_totales = bilan.get(\"dettes_totales\", 0)\n    actifs_circulants = bilan.get(\"actifs_circulants\", 0)\n    passifs_circulants = bilan.get(\"passifs_circulants\", 1)\n    creances_clients = bilan.get(\"creances_clients\", 0)\n    dettes_fournisseurs = bilan.get(\"dettes_fournisseurs\", 0)\n    stocks = bilan.get(\"stocks\", 1)\n    ca = compte_resultat.get(\"chiffre_affaires\", 1)\n    resultat_net = compte_resultat.get(\"resultat_net\", 0)\n    achats = compte_resultat.get(\"achats\", 1)\n    dotations = compte_resultat.get(\"dotations_amortissements\", 0)\n    ca_precedent = compte_resultat.get(\"ca_precedent\", ca)\n    return FinancialRatios(\n        ratio_endettement=dettes_totales / max(capitaux_propres, 1),\n        ratio_liquidite=actifs_circulants / max(passifs_circulants, 1),\n        ratio_solvabilite=capitaux_propres / max(total_bilan, 1),\n        marge_nette=resultat_net / max(ca, 1),\n        rotation_stocks=ca / max(stocks, 1),\n        delai_paiement_clients=(creances_clients / max(ca, 1)) * 365,\n        delai_paiement_fournisseurs=(dettes_fournisseurs / max(achats, 1)) * 365,\n        capacite_autofinancement=resultat_net + dotations,\n        taux_croissance_ca=((ca - ca_precedent) / max(abs(ca_precedent), 1)) * 100\n    )\n\ndef ratios_to_features(ratios: FinancialRatios) -> list[float]:\n    return [\n        ratios.ratio_endettement,\n        ratios.ratio_liquidite,\n        ratios.ratio_solvabilite,\n        ratios.marge_nette,\n        ratios.rotation_stocks,\n        ratios.delai_paiement_clients,\n        ratios.delai_paiement_fournisseurs,\n        ratios.capacite_autofinancement,\n        ratios.taux_croissance_ca,\n    ]",
            filename: "ratios.py",
          },
        ],
      },
      {
        title: "Modèle de scoring statistique",
        content:
          "Entraînez un modèle XGBoost sur vos données historiques pour produire un score quantitatif. Ce score est ensuite enrichi par l'analyse qualitative du LLM.",
        codeSnippets: [
          {
            language: "python",
            code: "import xgboost as xgb\nimport numpy as np\nfrom sklearn.model_selection import cross_val_score\nimport joblib\n\ndef train_scoring_model(features: np.ndarray, labels: np.ndarray, model_path: str) -> xgb.XGBClassifier:\n    model = xgb.XGBClassifier(\n        n_estimators=200,\n        max_depth=6,\n        learning_rate=0.05,\n        objective=\"multi:softprob\",\n        num_class=5,  # Grades A a E\n        eval_metric=\"mlogloss\",\n        random_state=42\n    )\n    scores = cross_val_score(model, features, labels, cv=5, scoring=\"accuracy\")\n    print(\"Accuracy CV: {:.3f} (+/- {:.3f})\".format(scores.mean(), scores.std()))\n    model.fit(features, labels)\n    joblib.dump(model, model_path)\n    return model\n\ndef predict_risk(model: xgb.XGBClassifier, features: list[float]) -> tuple[int, str]:\n    features_array = np.array([features])\n    probas = model.predict_proba(features_array)[0]\n    # Score sur 1000 : moyenne ponderee des probabilites\n    grade_scores = {0: 900, 1: 700, 2: 500, 3: 300, 4: 100}\n    score = sum(probas[i] * grade_scores[i] for i in range(5))\n    predicted_grade = model.predict(features_array)[0]\n    grade_map = {0: \"A\", 1: \"B\", 2: \"C\", 3: \"D\", 4: \"E\"}\n    return int(score), grade_map[predicted_grade]",
            filename: "scoring_model.py",
          },
        ],
      },
      {
        title: "Analyse qualitative par le LLM",
        content:
          "Le LLM enrichit le score quantitatif avec une analyse qualitative : interprétation des tendances, signaux faibles détectés dans les actualités, et recommandations argumentées.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nfrom models import CreditRiskScore, RiskGrade, FinancialRatios\n\nclient = anthropic.Anthropic()\n\ndef analyze_credit_risk(\n    ratios: FinancialRatios,\n    statistical_score: int,\n    statistical_grade: str,\n    company_info: dict,\n    sector_benchmarks: dict,\n    payment_history: dict,\n    legal_info: dict\n) -> CreditRiskScore:\n    ratios_text = \"\\n\".join(\n        \"- {}: {}\".format(k, round(v, 3)) for k, v in ratios.model_dump().items()\n    )\n    prompt = (\n        \"Tu es un analyste credit senior dans une institution financiere.\\n\"\n        \"Analyse ce dossier de risque credit et produis une evaluation detaillee.\\n\\n\"\n        \"Score statistique: {score}/1000 (grade {grade})\\n\\n\"\n        \"Ratios financiers:\\n{ratios}\\n\\n\"\n        \"Informations entreprise:\\n\"\n        \"- Raison sociale: {name}\\n\"\n        \"- SIREN: {siren}\\n\"\n        \"- Secteur: {sector}\\n\"\n        \"- Effectif: {employees}\\n\\n\"\n        \"Benchmarks sectoriels: {benchmarks}\\n\\n\"\n        \"Historique de paiement: {history}\\n\\n\"\n        \"Informations legales: {legal}\\n\\n\"\n        \"Retourne un JSON avec: score (0-1000), grade (A-E), \"\n        \"strengths (points forts), warnings (points de vigilance), \"\n        \"recommended_credit_limit, recommended_payment_terms (jours), \"\n        \"review_frequency, detailed_analysis, confidence\"\n    ).format(\n        score=statistical_score, grade=statistical_grade,\n        ratios=ratios_text,\n        name=company_info.get(\"name\", \"\"),\n        siren=company_info.get(\"siren\", \"\"),\n        sector=company_info.get(\"sector\", \"\"),\n        employees=company_info.get(\"employees\", \"\"),\n        benchmarks=str(sector_benchmarks),\n        history=str(payment_history),\n        legal=str(legal_info)\n    )\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=4096,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    result = CreditRiskScore.model_validate_json(message.content[0].text)\n    result.financial_ratios = ratios\n    return result",
            filename: "credit_analyzer.py",
          },
        ],
      },
      {
        title: "API et pipeline complet",
        content:
          "Exposez le scoring via une API REST sécurisée. Le pipeline orchestre la collecte de données, le calcul des ratios, le scoring statistique, l'analyse LLM et la génération du rapport.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, HTTPException, Depends\nfrom fastapi.security import HTTPBearer\nimport joblib\n\napp = FastAPI(title=\"Credit Risk Scoring Agent\")\nsecurity = HTTPBearer()\nscoring_model = joblib.load(\"models/xgb_credit_v1.joblib\")\n\n@app.post(\"/api/credit/score\")\nasync def score_company(request: dict, token=Depends(security)):\n    siren = request.get(\"siren\")\n    if not siren:\n        raise HTTPException(400, \"SIREN requis\")\n    # Collecte des donnees multi-sources\n    bilan = await fetch_financial_data(siren)\n    company_info = await fetch_company_info(siren)\n    payment_history = await fetch_payment_history(siren)\n    legal_info = await fetch_legal_info(siren)\n    sector_benchmarks = await fetch_sector_benchmarks(company_info[\"sector\"])\n    # Calcul des ratios\n    ratios = compute_ratios(bilan[\"bilan\"], bilan[\"compte_resultat\"])\n    features = ratios_to_features(ratios)\n    # Scoring statistique\n    stat_score, stat_grade = predict_risk(scoring_model, features)\n    # Analyse LLM\n    result = analyze_credit_risk(\n        ratios, stat_score, stat_grade,\n        company_info, sector_benchmarks, payment_history, legal_info\n    )\n    # Sauvegarde et audit\n    await save_scoring_result(siren, result)\n    # Alertes si risque eleve\n    if result.grade in [RiskGrade.D, RiskGrade.E]:\n        await send_risk_alert(siren, result)\n    return result.model_dump()\n\n@app.get(\"/api/credit/history/{siren}\")\nasync def scoring_history(siren: str, token=Depends(security)):\n    return await fetch_scoring_history(siren)\n\n@app.get(\"/api/credit/portfolio\")\nasync def portfolio_overview(token=Depends(security)):\n    return {\n        \"total_expositions\": await total_exposure(),\n        \"distribution_grades\": await grade_distribution(),\n        \"top_risques\": await top_risks(limit=20),\n        \"evolution_mensuelle\": await monthly_trend()\n    }",
            filename: "api.py",
          },
        ],
      },
      {
        title: "Tests et validation réglementaire",
        content:
          "Testez le pipeline complet avec des données de test couvrant tous les grades de risque. Validez la conformité avec les exigences réglementaires (traçabilité, non-discrimination, explicabilité).",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nimport numpy as np\nfrom models import CreditRiskScore, RiskGrade, FinancialRatios\nfrom ratios import compute_ratios\nfrom scoring_model import predict_risk\nfrom credit_analyzer import analyze_credit_risk\n\ndef test_ratios_computation():\n    bilan = {\n        \"capitaux_propres\": 500000, \"total_actif\": 1200000,\n        \"dettes_totales\": 700000, \"actifs_circulants\": 600000,\n        \"passifs_circulants\": 400000, \"creances_clients\": 150000,\n        \"dettes_fournisseurs\": 100000, \"stocks\": 80000\n    }\n    cr = {\n        \"chiffre_affaires\": 2000000, \"resultat_net\": 120000,\n        \"achats\": 800000, \"dotations_amortissements\": 50000,\n        \"ca_precedent\": 1800000\n    }\n    ratios = compute_ratios(bilan, cr)\n    assert ratios.ratio_endettement == pytest.approx(1.4, rel=0.01)\n    assert ratios.ratio_liquidite == pytest.approx(1.5, rel=0.01)\n    assert ratios.marge_nette == pytest.approx(0.06, rel=0.01)\n    assert ratios.taux_croissance_ca == pytest.approx(11.11, rel=0.1)\n\ndef test_scoring_model_output_range():\n    features = [1.2, 1.5, 0.42, 0.06, 25.0, 27.4, 45.6, 170000, 11.1]\n    score, grade = predict_risk(scoring_model, features)\n    assert 0 <= score <= 1000\n    assert grade in [\"A\", \"B\", \"C\", \"D\", \"E\"]\n\ndef test_high_risk_detection():\n    # Entreprise en difficulte financiere\n    bilan = {\n        \"capitaux_propres\": -50000, \"total_actif\": 300000,\n        \"dettes_totales\": 350000, \"actifs_circulants\": 80000,\n        \"passifs_circulants\": 250000, \"creances_clients\": 60000,\n        \"dettes_fournisseurs\": 120000, \"stocks\": 40000\n    }\n    cr = {\n        \"chiffre_affaires\": 500000, \"resultat_net\": -80000,\n        \"achats\": 300000, \"dotations_amortissements\": 20000,\n        \"ca_precedent\": 600000\n    }\n    ratios = compute_ratios(bilan, cr)\n    assert ratios.ratio_liquidite < 1.0\n    assert ratios.marge_nette < 0\n    assert ratios.taux_croissance_ca < 0\n\ndef test_credit_analysis_completeness():\n    ratios = FinancialRatios(\n        ratio_endettement=1.4, ratio_liquidite=1.5, ratio_solvabilite=0.42,\n        marge_nette=0.06, rotation_stocks=25.0, delai_paiement_clients=27.4,\n        delai_paiement_fournisseurs=45.6, capacite_autofinancement=170000,\n        taux_croissance_ca=11.1\n    )\n    result = analyze_credit_risk(\n        ratios, 720, \"B\",\n        {\"name\": \"Test SAS\", \"siren\": \"123456789\", \"sector\": \"tech\", \"employees\": 50},\n        {\"marge_nette_median\": 0.05}, {\"retards_30j\": 0}, {\"contentieux\": 0}\n    )\n    assert isinstance(result, CreditRiskScore)\n    assert 0 <= result.score <= 1000\n    assert result.grade in list(RiskGrade)\n    assert len(result.strengths) >= 1\n    assert result.recommended_credit_limit > 0\n    assert result.confidence >= 0.5",
            filename: "test_credit_scoring.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données financières des entreprises sont hautement confidentielles. Les bilans complets ne sont jamais envoyés au LLM : seuls les ratios calculés et les données agrégées sont transmis. Les données nominatives des dirigeants sont pseudonymisées. Le stockage respecte les normes bancaires (chiffrement AES-256, clés gérées par HSM). Conformité avec le secret bancaire et le RGPD pour les données des dirigeants personnes physiques.",
      auditLog: "Piste d'audit complète conforme Bâle III/IV : horodatage de chaque scoring, données sources utilisées, version du modèle statistique, ratios calculés, score LLM, score final retenu, grade attribué, limite de crédit recommandée, identité de l'analyste valideur, décision finale. Conservation 10 ans minimum (exigence réglementaire). Traçabilité des modifications de modèle.",
      humanInTheLoop: "Tout scoring aboutissant à un grade D ou E est obligatoirement revu par un analyste crédit senior avant notification au client. Les décisions d'octroi supérieures à 500K EUR nécessitent une double validation (analyste + responsable engagements). Un comité de crédit mensuel revoit les dossiers sensibles. L'analyste peut ajuster le score avec justification obligatoire tracée.",
      monitoring: "Dashboard réglementaire : distribution des grades du portefeuille, taux de défaut observé vs prédit par grade (backtesting), performance du modèle (Gini, KS, AUC), concentration sectorielle et géographique, évolution des encours par grade, alertes de dégradation, suivi de la calibration du modèle, rapport de conformité EBA automatisé.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (demande de scoring via API) → Node HTTP Request (API Infogreffe - données légales) → Node HTTP Request (API Banque de France - cotation) → Node Code (calcul ratios financiers) → Node HTTP Request (API modèle XGBoost - scoring statistique) → Node HTTP Request (API Claude - analyse qualitative) → Node Code (score final pondéré) → Node Switch (grade D/E ?) → Branch risque élevé: Node Email (alerte analyste senior) → Node PostgreSQL (sauvegarde scoring + audit) → Node HTTP Request (génération rapport PDF).",
      nodes: ["Webhook (demande scoring)", "HTTP Request (Infogreffe)", "HTTP Request (Banque de France)", "Code (calcul ratios)", "HTTP Request (XGBoost scoring)", "HTTP Request (Claude analyse)", "Code (score final)", "Switch (grade)", "Email (alerte risque)", "PostgreSQL (audit)", "HTTP Request (rapport PDF)"],
      triggerType: "Webhook (demande de scoring crédit via API sécurisée)",
    },
    estimatedTime: "14-20h",
    difficulty: "Expert",
    sectors: ["Banque", "Assurance", "Finance", "Leasing", "Affacturage"],
    metiers: ["Analyse Crédit", "Risk Management", "Direction des Risques", "Engagements"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Scoring de Risque Crédit — Guide Expert",
    metaDescription:
      "Automatisez l'évaluation du risque crédit avec un agent IA. Analyse financière multi-sources, scoring statistique XGBoost, analyse qualitative LLM et conformité Bâle III. Tutoriel complet.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-chatbot-whatsapp-business",
    title: "Agent Chatbot WhatsApp Business",
    subtitle: "Engagez vos clients et suivez leurs commandes via un chatbot IA sur WhatsApp Business",
    problem:
      "Les entreprises B2B recoivent un volume croissant de demandes clients via WhatsApp : suivi de commandes, questions produits, demandes de devis. Les equipes commerciales et support ne peuvent pas repondre en temps reel 24h/24, ce qui entraine des delais de reponse eleves, une insatisfaction client et des opportunites commerciales manquees. Le copier-coller de reponses types ne suffit plus face a la diversite des demandes.",
    value:
      "Un agent IA connecte a WhatsApp Business API analyse chaque message entrant, identifie l'intention du client (suivi commande, demande de devis, question produit, reclamation), interroge les systemes internes (ERP, CRM, logistique) et repond de maniere personnalisee et instantanee. Les demandes complexes sont escaladees vers un humain avec tout le contexte. Le taux de reponse passe a 100% et le delai moyen chute sous les 30 secondes.",
    inputs: [
      "Messages WhatsApp entrants (texte, images, documents)",
      "Catalogue produits et tarifs",
      "Donnees ERP (commandes, stocks, expeditions)",
      "Historique client (CRM)",
      "FAQ et base de connaissances",
    ],
    outputs: [
      "Reponses automatiques personnalisees sur WhatsApp",
      "Statut de commande en temps reel",
      "Devis generes automatiquement",
      "Escalade vers un agent humain avec contexte complet",
      "Rapport d'interactions et metriques d'engagement",
    ],
    risks: [
      "Reponses incorrectes sur le statut de commande ou les tarifs",
      "Non-conformite RGPD sur les donnees personnelles echangees via WhatsApp",
      "Hallucination du LLM sur des informations produit critiques",
      "Dependance a la disponibilite de l'API WhatsApp Business",
      "Ton inapproprie dans un contexte B2B formel",
    ],
    roiIndicatif:
      "Reduction de 70% du temps de reponse moyen. Augmentation de 40% du taux d'engagement client. Diminution de 50% de la charge du support niveau 1.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS (Lambda + API Gateway)", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-------------+     +----------------+     +-------------+\n|  WhatsApp   |---->|  Webhook API   |---->|  Agent LLM  |\n|  Business   |     |  (FastAPI)     |     |  (Routage)  |\n+-------------+     +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  ERP / CRM     |<----|  Fonctions  |\n                    |  (Donnees)     |     |  (Outils)   |\n                    +----------------+     +-------------+\n                                                  |\n                                           +------v------+\n                                           |  Reponse    |\n                                           |  WhatsApp   |\n                                           +-------------+",
    tutorial: [
      {
        title: "Prerequis et configuration WhatsApp Business API",
        content:
          "Pour commencer, vous devez configurer un compte WhatsApp Business API via Meta Business Suite. Creez une application Meta, activez le produit WhatsApp et obtenez vos identifiants. Le processus de verification de l'entreprise peut prendre quelques jours.\n\nInstallez les dependances Python necessaires pour le projet. Vous aurez besoin de FastAPI pour l'API webhook, de LangChain pour l'orchestration de l'agent, et de la bibliotheque OpenAI pour le LLM. Configurez les variables d'environnement pour securiser vos cles API.\n\nLa configuration du webhook est cruciale : WhatsApp enverra tous les messages entrants a votre endpoint. Assurez-vous que votre serveur est accessible via HTTPS avec un certificat SSL valide. Pour le developpement local, utilisez ngrok pour creer un tunnel securise.\n\nConfigurez les modeles de messages (templates) dans Meta Business Suite. Ces templates sont necessaires pour initier des conversations et envoyer des notifications proactives comme les confirmations de commande.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install fastapi uvicorn langchain openai httpx python-dotenv pydantic psycopg2-binary",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Webhook et traitement des messages entrants",
        content:
          "Le coeur du systeme est le webhook qui recoit les messages WhatsApp. Meta envoie une requete POST a votre endpoint chaque fois qu'un client envoie un message. Vous devez verifier la signature de la requete pour garantir son authenticite, puis extraire le contenu du message.\n\nLe webhook doit gerer differents types de messages : texte simple, images, documents et reponses interactives (boutons, listes). Chaque type necessite un traitement specifique avant d'etre envoye a l'agent LLM pour analyse et generation de reponse.\n\nIl est important de repondre rapidement au webhook (sous 5 secondes) pour eviter les timeouts de Meta. Utilisez un systeme de file d'attente asynchrone pour traiter les messages en arriere-plan tout en accusant reception immediatement.\n\nImplementez un mecanisme de deduplication car Meta peut renvoyer le meme message plusieurs fois. Stockez les identifiants de messages dans votre base de donnees pour eviter les reponses en double.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, Request, HTTPException\nimport httpx\nimport os\n\napp = FastAPI()\n\n@app.get(\"/webhook\")\nasync def verify_webhook(hub_mode: str = \"\", hub_verify_token: str = \"\", hub_challenge: str = \"\"):\n    if hub_mode == \"subscribe\" and hub_verify_token == os.getenv(\"WHATSAPP_VERIFY_TOKEN\"):\n        return int(hub_challenge)\n    raise HTTPException(403, \"Token invalide\")\n\n@app.post(\"/webhook\")\nasync def receive_message(request: Request):\n    data = await request.json()\n    entry = data.get(\"entry\", [{}])[0]\n    changes = entry.get(\"changes\", [{}])[0]\n    value = changes.get(\"value\", {})\n    if \"messages\" in value:\n        message = value[\"messages\"][0]\n        sender = message[\"from\"]\n        text = message.get(\"text\", {}).get(\"body\", \"\")\n        response = await process_message(sender, text)\n        await send_whatsapp_reply(sender, response)\n    return {\"status\": \"ok\"}",
            filename: "webhook.py",
          },
        ],
      },
      {
        title: "Agent conversationnel avec outils metier",
        content:
          "L'agent LLM est le cerveau du chatbot. Il analyse l'intention du client, decide quel outil utiliser (recherche commande, catalogue produit, generation devis, escalade), et formule une reponse naturelle et professionnelle. L'utilisation du pattern ReAct permet a l'agent de raisonner etape par etape.\n\nDefinissez les outils metier que l'agent peut appeler : recherche de commande par numero ou nom client, consultation du catalogue et des stocks, generation de devis, et escalade vers un humain. Chaque outil est une fonction Python qui interroge vos systemes internes.\n\nLa gestion du contexte conversationnel est essentielle. Stockez l'historique des conversations par client dans PostgreSQL pour que l'agent puisse maintenir le fil de la discussion. Limitez le contexte aux 10 derniers messages pour optimiser les couts et la latence.\n\nParametrez le prompt systeme avec soin : ton professionnel B2B, reponses concises adaptees au format mobile, vouvoiement systematique, et limites claires sur les actions autonomes de l'agent.",
        codeSnippets: [
          {
            language: "python",
            code: "from langchain.chat_models import ChatOpenAI\nfrom langchain.agents import AgentExecutor, create_openai_tools_agent\nfrom langchain.tools import tool\nfrom langchain.prompts import ChatPromptTemplate, MessagesPlaceholder\n\n@tool\ndef rechercher_commande(numero_commande: str) -> str:\n    \"\"\"Recherche le statut d'une commande par son numero.\"\"\"\n    commande = query_erp_order(numero_commande)\n    if not commande:\n        return \"Aucune commande trouvee avec ce numero.\"\n    return \"Commande {num}: statut={s}, livraison prevue le {d}\".format(\n        num=commande[\"numero\"], s=commande[\"statut\"], d=commande[\"date_livraison\"]\n    )\n\n@tool\ndef consulter_catalogue(recherche: str) -> str:\n    \"\"\"Recherche un produit dans le catalogue.\"\"\"\n    produits = search_catalog(recherche)\n    if not produits:\n        return \"Aucun produit trouve.\"\n    return \"\\n\".join(\n        \"- {n} : {p} EUR HT (stock: {s})\".format(n=p[\"nom\"], p=p[\"prix\"], s=p[\"stock\"]) for p in produits[:5]\n    )\n\nllm = ChatOpenAI(model=\"gpt-4.1\", temperature=0.1)\ntools = [rechercher_commande, consulter_catalogue]",
            filename: "agent_whatsapp.py",
          },
        ],
      },
      {
        title: "Tests et deploiement en production",
        content:
          "Avant le deploiement, testez exhaustivement chaque scenario conversationnel. Simulez des demandes de suivi commande, des questions produit, des reclamations et des cas limites (messages vides, spam, langues non supportees). Mesurez la qualite des reponses avec un jeu de test annote.\n\nDeployez l'API sur AWS Lambda avec API Gateway pour beneficier du scaling automatique. WhatsApp peut envoyer des pics de messages importants, notamment apres l'envoi de campagnes promotionnelles. Lambda gere ces pics sans intervention manuelle.\n\nMettez en place un monitoring complet avec Langfuse : tracez chaque conversation, mesurez le temps de reponse, le taux d'escalade vers un humain, et la satisfaction client. Configurez des alertes si le taux d'escalade depasse un seuil configurable.\n\nPrevoyez une phase de rodage de 2 semaines ou toutes les reponses sont verifiees par un humain avant envoi (mode shadow). Cela permet d'affiner le prompt et de corriger les cas non couverts avant de passer en mode autonome.",
        codeSnippets: [
          {
            language: "python",
            code: "import pytest\nfrom agent_whatsapp import executor\n\ndef test_suivi_commande():\n    result = executor.invoke({\n        \"input\": \"Bonjour, ou en est ma commande CMD-2024-1234 ?\",\n        \"chat_history\": [],\n    })\n    assert \"CMD-2024-1234\" in result[\"output\"]\n\ndef test_escalade():\n    result = executor.invoke({\n        \"input\": \"Je souhaite annuler ma commande immediatement\",\n        \"chat_history\": [],\n    })\n    assert any(mot in result[\"output\"].lower() for mot in [\"conseiller\", \"humain\", \"equipe\"])",
            filename: "test_agent.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les messages WhatsApp contiennent des donnees personnelles (numeros de telephone, noms, adresses). Anonymisez ces donnees avant envoi au LLM. Les numeros sont pseudonymises dans les logs. Stockage conforme RGPD avec chiffrement AES-256. Politique de retention limitee a 12 mois avec purge automatique.",
      auditLog: "Chaque interaction est tracee : horodatage, numero pseudonymise, intention detectee, outil appele, reponse generee, temps de traitement, score de confiance, indicateur d'escalade. Logs stockes dans PostgreSQL avec retention de 24 mois.",
      humanInTheLoop: "Les demandes de modification de commande, reclamations financieres et messages a faible confiance sont escalades vers un agent humain. Le mode shadow est active pendant les 2 premieres semaines. Un superviseur peut prendre le controle de n'importe quelle conversation.",
      monitoring: "Dashboard temps reel : volume de messages par heure, temps de reponse median, taux d'escalade, taux de resolution autonome, satisfaction client, top 10 des intentions, alertes sur les anomalies.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (message WhatsApp entrant) -> Node Code (extraction du message) -> Node HTTP Request (API LLM agent) -> Node Switch (intention) -> Branch commande: Node HTTP Request (ERP) -> Branch escalade: Node Slack (notification) -> Node HTTP Request (reponse WhatsApp API).",
      nodes: ["Webhook (WhatsApp)", "Code (extraction)", "HTTP Request (Agent LLM)", "Switch (intention)", "HTTP Request (ERP)", "Slack (escalade)", "HTTP Request (reponse WhatsApp)"],
      triggerType: "Webhook (message WhatsApp entrant)",
    },
    estimatedTime: "6-10h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "Distribution", "Industrie", "Services B2B"],
    metiers: ["Service Client", "Commerce", "Logistique"],
    functions: ["Support", "Sales"],
    metaTitle: "Agent Chatbot WhatsApp Business IA -- Guide Complet",
    metaDescription:
      "Deployez un chatbot IA sur WhatsApp Business pour automatiser le suivi de commandes et l'engagement client. Stack technique, tutoriel et ROI detaille.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-analyse-sentiments-reseaux",
    title: "Agent d'Analyse de Sentiments sur les Reseaux Sociaux",
    subtitle: "Surveillez la reputation de votre marque en temps reel grace a l'analyse de sentiments IA",
    problem:
      "Les entreprises B2B sont mentionnees sur les reseaux sociaux (LinkedIn, X/Twitter, forums specialises) sans en avoir conscience. Les equipes marketing ne peuvent pas surveiller manuellement des milliers de publications quotidiennes. Les crises reputationnelles sont detectees trop tard, et les opportunites d'engagement positif sont manquees. Les outils de social listening classiques offrent une analyse de sentiment basique et peu fiable sur le vocabulaire B2B francophone.",
    value:
      "Un agent IA collecte en continu les mentions de votre marque, de vos produits et de vos concurrents sur les reseaux sociaux et forums professionnels. Il analyse le sentiment avec une comprehension fine du contexte B2B, detecte les tendances emergentes, identifie les influenceurs cles, et declenche des alertes en temps reel en cas de crise. Les equipes marketing disposent d'un tableau de bord actionnable avec des recommandations de reponse.",
    inputs: [
      "Flux de donnees reseaux sociaux (LinkedIn, X/Twitter, forums)",
      "Liste de mots-cles et mentions a surveiller",
      "Profils concurrents a analyser",
      "Historique des campagnes marketing",
      "Base de connaissances marque (ton, valeurs, FAQ)",
    ],
    outputs: [
      "Score de sentiment par mention (-1 a +1)",
      "Classification thematique des mentions",
      "Alertes en temps reel pour les mentions critiques",
      "Tableau de bord d'evolution du sentiment",
      "Recommandations de reponse contextualisees",
      "Rapport hebdomadaire de veille reputationnelle",
    ],
    risks: [
      "Mauvaise interpretation du sarcasme ou de l'ironie en contexte francophone",
      "Biais dans l'analyse de sentiment selon les secteurs",
      "Non-respect des conditions d'utilisation des API reseaux sociaux",
      "Volume de donnees trop important degradant la qualite d'analyse",
      "Faux positifs declenchant des alertes inutiles",
    ],
    roiIndicatif:
      "Detection des crises reputationnelles 6x plus rapide. Augmentation de 45% du taux d'engagement sur les reponses aux mentions. Economie de 2 ETP sur la veille manuelle.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Mistral Large", category: "LLM", isFree: false },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-------------+  +-------------+  +-------------+\n|  LinkedIn   |  |  X/Twitter  |  |  Forums     |\n|  API        |  |  API        |  |  RSS/Scrape |\n+------+------+  +------+------+  +------+------+\n       |                |                |\n       v                v                v\n+----------------------------------------------+\n|         Collecteur de Mentions               |\n|         (File d'attente Redis)               |\n+-----------------------+----------------------+\n                        |\n                +-------v--------+\n                |   Agent LLM    |\n                |   (Analyse)    |\n                +-------+--------+\n                        |\n              +---------+---------+\n              |                   |\n       +------v------+     +------v------+\n       |  Dashboard  |     |  Alertes    |\n       |  (Metabase) |     |  (Slack)    |\n       +-------------+     +-------------+",
    tutorial: [
      {
        title: "Prerequis et configuration des sources de donnees",
        content:
          "Commencez par configurer les acces aux API des reseaux sociaux. Pour LinkedIn, vous aurez besoin d'une application LinkedIn Developer avec les permissions Marketing API. Pour X/Twitter, creez un projet dans le Developer Portal avec un acces Business pour beneficier du volume de requetes necessaire.\n\nInstallez les dependances Python du projet. Le collecteur de mentions utilise des bibliotheques specifiques pour chaque plateforme, ainsi que Redis pour la file d'attente de traitement. LangChain orchestre l'agent d'analyse et Anthropic Claude fournit la comprehension fine du contexte.\n\nConfigurez les mots-cles de surveillance : nom de votre entreprise et ses variantes, noms de produits, noms des dirigeants, et termes specifiques a votre secteur. Incluez egalement les noms de vos principaux concurrents pour une analyse comparative.\n\nMettez en place un scheduler (cron ou APScheduler) pour collecter les mentions a intervalles reguliers. La frequence depend de votre volume : toutes les 5 minutes pour les grandes marques, toutes les heures pour les PME.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install anthropic langchain httpx redis apscheduler pydantic fastapi psycopg2-binary tweepy",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Collecteur de mentions multi-plateformes",
        content:
          "Le collecteur interroge chaque plateforme et normalise les donnees dans un format unifie. Chaque mention est stockee avec ses metadonnees (auteur, date, plateforme, engagement) dans une file Redis pour traitement asynchrone par l'agent d'analyse.\n\nPour X/Twitter, utilisez l'API v2 avec les operateurs de recherche avances. Filtrez par langue (lang:fr) et excluez les retweets pour eviter les doublons. Pour LinkedIn, l'API Marketing permet de recuperer les mentions de votre page entreprise.\n\nPour les forums specialises (forums sectoriels, Reddit, communautes publiques), utilisez un scraper RSS ou un outil comme Apify. Normalisez les donnees dans le meme schema que les reseaux sociaux pour un traitement uniforme.\n\nImplementez un systeme de deduplication base sur le hash du contenu et l'identifiant de la plateforme. Les memes contenus peuvent apparaitre sur plusieurs plateformes (cross-posting), et il est important de les regrouper.",
        codeSnippets: [
          {
            language: "python",
            code: "import tweepy\nimport redis\nimport json\nfrom datetime import datetime\nfrom pydantic import BaseModel\nfrom typing import Optional\n\nclass Mention(BaseModel):\n    platform: str\n    author: str\n    content: str\n    url: str\n    timestamp: datetime\n    engagement: int\n    language: str = \"fr\"\n\nredis_client = redis.from_url(\"redis://localhost:6379\")\n\ndef collect_twitter_mentions(keywords: list[str], since_id: str = None) -> list[Mention]:\n    client = tweepy.Client(bearer_token=\"...\")\n    query = \" OR \".join(keywords) + \" lang:fr -is:retweet\"\n    tweets = client.search_recent_tweets(\n        query=query, max_results=100, since_id=since_id,\n        tweet_fields=[\"created_at\", \"public_metrics\", \"author_id\"]\n    )\n    mentions = []\n    if tweets.data:\n        for tweet in tweets.data:\n            m = Mention(\n                platform=\"twitter\",\n                author=str(tweet.author_id),\n                content=tweet.text,\n                url=\"https://x.com/i/status/{}\".format(tweet.id),\n                timestamp=tweet.created_at,\n                engagement=tweet.public_metrics[\"like_count\"] + tweet.public_metrics[\"retweet_count\"]\n            )\n            mentions.append(m)\n            redis_client.lpush(\"mentions_queue\", m.model_dump_json())\n    return mentions",
            filename: "collector.py",
          },
        ],
      },
      {
        title: "Agent d'analyse de sentiment contextuel",
        content:
          "L'agent LLM analyse chaque mention avec une comprehension fine du contexte B2B francophone. Contrairement aux outils de sentiment classiques bases sur des lexiques, le LLM comprend le sarcasme, les references sectorielles, et les nuances du langage professionnel.\n\nLe prompt systeme est calibre pour le contexte B2B francais. Il inclut les specificites de votre marque, vos produits, et les sujets sensibles a surveiller en priorite. L'agent produit un score de sentiment continu (-1 a +1) plus nuance qu'une simple classification.\n\nPour les mentions a fort engagement ou a sentiment tres negatif, l'agent genere automatiquement une recommandation de reponse adaptee au ton de votre marque. Cette reponse est envoyee a l'equipe marketing pour validation avant publication.\n\nLe traitement par batch permet d'optimiser les couts API. Regroupez les mentions par lots de 10-20 pour une analyse en une seule requete LLM, tout en maintenant la qualite individuelle de chaque analyse.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nfrom pydantic import BaseModel, Field\n\nclient = anthropic.Anthropic()\n\nclass SentimentAnalysis(BaseModel):\n    sentiment_score: float = Field(ge=-1.0, le=1.0)\n    sentiment_label: str\n    themes: list[str]\n    urgency: str\n    summary: str\n    recommended_action: str\n    suggested_response: str = \"\"\n    confidence: float = Field(ge=0.0, le=1.0)\n\ndef analyze_mention(mention_content: str, author_info: str, brand_context: str) -> SentimentAnalysis:\n    prompt = (\n        \"Tu es un analyste de reputation de marque specialise B2B.\\n\"\n        \"Analyse cette mention de notre marque sur les reseaux sociaux.\\n\\n\"\n        \"Mention: {content}\\n\"\n        \"Auteur: {author}\\n\"\n        \"Contexte marque: {context}\\n\\n\"\n        \"Analyse le sentiment, les themes, l'urgence, et recommande une action.\\n\"\n        \"Retourne un JSON avec: sentiment_score, sentiment_label, themes, \"\n        \"urgency, summary, recommended_action, suggested_response, confidence.\"\n    ).format(content=mention_content, author=author_info, context=brand_context)\n    message = client.messages.create(\n        model=\"claude-sonnet-4-5-20250514\",\n        max_tokens=1024,\n        messages=[{\"role\": \"user\", \"content\": prompt}]\n    )\n    return SentimentAnalysis.model_validate_json(message.content[0].text)",
            filename: "sentiment_agent.py",
          },
        ],
      },
      {
        title: "Alertes, dashboard et deploiement",
        content:
          "Configurez un systeme d'alertes multi-niveaux. Les mentions critiques (sentiment tres negatif, fort engagement, auteur influent) declenchent une notification Slack immediate avec le contexte complet et la reponse suggeree. Les tendances negatives detectees sur 24h generent un rapport de synthese par email.\n\nDeployez un dashboard Metabase connecte a PostgreSQL pour visualiser les metriques en temps reel : evolution du sentiment moyen, repartition par theme, volume de mentions par plateforme, top auteurs et influenceurs, et comparaison avec les concurrents.\n\nAutomatisez la generation de rapports hebdomadaires. L'agent LLM synthetise les evenements marquants de la semaine, les tendances emergentes, et propose des recommandations strategiques. Ce rapport est envoye automatiquement au directeur marketing.\n\nPour le deploiement en production, utilisez Vercel pour l'API et un worker Redis dedie pour le traitement des mentions. Mettez en place des health checks et des alertes d'infrastructure pour garantir la continuite du service de veille.",
        codeSnippets: [
          {
            language: "python",
            code: "import httpx\nfrom datetime import datetime\n\nasync def send_critical_alert(mention: dict, analysis: dict):\n    payload = {\n        \"blocks\": [\n            {\"type\": \"header\", \"text\": {\"type\": \"plain_text\", \"text\": \"Alerte Reputation Critique\"}},\n            {\"type\": \"section\", \"text\": {\"type\": \"mrkdwn\", \"text\": (\n                \"*Plateforme:* {plat}\\n*Sentiment:* {label} ({score})\\n*Urgence:* {urg}\\n\\n\"\n                \">{content}\\n\\n*Reponse suggeree:*\\n{resp}\"\n            ).format(\n                plat=mention[\"platform\"], label=analysis[\"sentiment_label\"],\n                score=analysis[\"sentiment_score\"], urg=analysis[\"urgency\"],\n                content=mention[\"content\"][:500], resp=analysis[\"suggested_response\"]\n            )}}\n        ]\n    }\n    async with httpx.AsyncClient() as client:\n        await client.post(SLACK_WEBHOOK_URL, json=payload)",
            filename: "alerts.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les donnees collectees sur les reseaux sociaux peuvent contenir des informations personnelles. Seul le contenu textuel est envoye au LLM, sans les metadonnees d'identification. Les profils auteurs sont pseudonymises dans la base de donnees. Conformite RGPD avec droit d'opposition. Donnees stockees en UE uniquement.",
      auditLog: "Chaque analyse est tracee : horodatage, plateforme source, mention originale (hash), score de sentiment, themes detectes, action recommandee, et identite de l'operateur ayant valide ou rejete la recommandation. Conservation 18 mois avec export automatique.",
      humanInTheLoop: "Les mentions critiques (sentiment < -0.7 ou urgence critique) sont validees par un humain avant toute action de reponse. Les reponses suggerees ne sont jamais publiees automatiquement : un membre de l'equipe marketing doit les valider. Le mode automatique peut etre active uniquement pour les mentions positives a faible engagement.",
      monitoring: "Dashboard de monitoring : volume de mentions collectees par heure, taux de couverture par plateforme, latence d'analyse, distribution des sentiments en temps reel, taux d'alertes critiques, temps de reaction moyen de l'equipe, et evolution comparative du sentiment vs concurrents.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 15 min) -> Node HTTP Request (API Twitter) -> Node HTTP Request (API LinkedIn) -> Node Code (normalisation et deduplication) -> Node HTTP Request (API Claude - analyse sentiment) -> Node Switch (urgence critique ?) -> Branch critique: Node Slack (alerte) -> Node PostgreSQL (sauvegarde) -> Cron hebdomadaire: Node HTTP Request (rapport) -> Node Email (envoi rapport).",
      nodes: ["Cron Trigger (15 min)", "HTTP Request (Twitter API)", "HTTP Request (LinkedIn API)", "Code (normalisation)", "HTTP Request (Claude analyse)", "Switch (urgence)", "Slack (alerte critique)", "PostgreSQL (sauvegarde)", "Cron (hebdomadaire)", "Email (rapport)"],
      triggerType: "Cron (toutes les 15 minutes)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Services B2B", "Industrie", "Tech", "Conseil", "Finance"],
    metiers: ["Marketing Digital", "Communication", "Relations Publiques"],
    functions: ["Marketing"],
    metaTitle: "Agent IA d'Analyse de Sentiments Reseaux Sociaux -- Guide Complet",
    metaDescription:
      "Surveillez votre reputation de marque en temps reel avec un agent IA d'analyse de sentiments. Collecte multi-plateformes, analyse contextuelle et alertes automatiques. Tutoriel complet.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-prevision-demande-stock",
    title: "Agent de Prevision de Demande et Optimisation des Stocks",
    subtitle: "Anticipez la demande et optimisez vos niveaux de stock grace a l'IA predictive",
    problem:
      "Les entreprises B2B souffrent d'un double probleme de gestion des stocks : le surstockage immobilise du capital et genere des couts de stockage, tandis que les ruptures de stock entrainent des pertes de ventes et une insatisfaction client. Les methodes de prevision traditionnelles (moyennes mobiles, saisonnalite simple) ne captent pas les signaux faibles comme les tendances marche, les evenements macroeconomiques ou les comportements d'achat emergents.",
    value:
      "Un agent IA combine des modeles de prevision statistiques avec l'analyse contextuelle d'un LLM pour produire des previsions de demande precises. Il integre les donnees de vente historiques, les tendances marche, les evenements sectoriels et les signaux faibles pour generer des recommandations de reapprovisionnement optimales. Le taux de rupture chute et le capital immobilise diminue significativement.",
    inputs: [
      "Historique des ventes (3 ans minimum)",
      "Niveaux de stock actuels par reference",
      "Delais d'approvisionnement fournisseurs",
      "Calendrier promotionnel et evenements prevus",
      "Donnees marche et tendances sectorielles",
      "Donnees meteo (pour les produits saisonniers)",
    ],
    outputs: [
      "Previsions de demande par reference (horizon 1-12 semaines)",
      "Recommandations de reapprovisionnement avec quantites",
      "Alertes de risque de rupture de stock",
      "Analyse des surstocks avec recommandations de destockage",
      "Rapport de performance des previsions (MAPE, biais)",
      "Scenarios what-if pour les decisions strategiques",
    ],
    risks: [
      "Previsions erronees causant des ruptures ou du surstockage",
      "Surconfiance dans les predictions IA sans validation humaine",
      "Biais saisonnier mal calibre pour les nouveaux produits sans historique",
      "Dependance a la qualite et la completude des donnees historiques",
      "Impact financier eleve en cas de recommandation incorrecte",
    ],
    roiIndicatif:
      "Reduction de 35% du stock moyen. Diminution de 60% des ruptures de stock. Amelioration de 25% de la precision des previsions vs methodes traditionnelles. ROI typique de 300% la premiere annee.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS (Lambda + S3)", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "Evidently AI", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-------------+  +-------------+  +-------------+\n|  Historique |  |  Donnees    |  |  Signaux    |\n|  Ventes     |  |  Marche     |  |  Externes   |\n+------+------+  +------+------+  +------+------+\n       |                |                |\n       v                v                v\n+----------------------------------------------+\n|        Modele Statistique (Prophet)          |\n|        + Agent LLM (Contexte)                |\n+-----------------------+----------------------+\n                        |\n              +---------+---------+\n              |                   |\n       +------v------+     +------v------+\n       |  Previsions |     |  Alertes    |\n       |  + Reappro  |     |  Ruptures   |\n       +-------------+     +-------------+",
    tutorial: [
      {
        title: "Prerequis et preparation des donnees",
        content:
          "La qualite des previsions depend directement de la qualite des donnees. Commencez par extraire l'historique de ventes de votre ERP sur au moins 3 ans. Nettoyez les donnees : supprimez les commandes annulees, corrigez les valeurs aberrantes, et identifiez les periodes anormales pour les marquer comme anomalies.\n\nInstallez les dependances Python. Prophet (Meta) est un excellent modele de base pour les series temporelles avec saisonnalite. Combinez-le avec LangChain et un LLM pour ajouter une couche d'analyse contextuelle que les modeles statistiques seuls ne peuvent pas fournir.\n\nStructurez vos donnees dans un format standardise : une ligne par couple (reference, date) avec les colonnes quantite vendue, prix, categorie produit, et indicateurs contextuels. Ce format alimentera a la fois le modele statistique et l'agent LLM.\n\nConfigurez egalement les sources de donnees externes : API meteo pour les produits saisonniers, flux RSS sectoriels pour les tendances marche, et calendrier des salons professionnels. Ces signaux enrichissent la precision des previsions.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install openai langchain prophet pandas numpy scikit-learn psycopg2-binary fastapi pydantic httpx",
            filename: "terminal",
          },
        ],
      },
      {
        title: "Modele de prevision statistique avec Prophet",
        content:
          "Prophet est particulierement adapte aux series temporelles B2B car il gere nativement la saisonnalite multiple (hebdomadaire, mensuelle, annuelle), les jours feries, et les changements de tendance. Entrainez un modele par reference produit ou par famille de produits.\n\nConfigurez les parametres de Prophet en fonction de vos donnees : ajoutez les jours feries francais, definissez les periodes de changement de tendance, et incluez les regresseurs externes (promotions, meteo) qui impactent la demande.\n\nLe modele produit non seulement une prevision ponctuelle mais aussi un intervalle de confiance. Utilisez cet intervalle pour calibrer vos niveaux de stock de securite : un produit avec un intervalle large necessite un stock de securite plus important.\n\nEvaluez les performances avec un backtesting rigoureux : entrainez sur 80% de l'historique et testez sur les 20% restants. Mesurez le MAPE (erreur moyenne absolue en pourcentage) et le biais. Un MAPE inferieur a 20% est un bon objectif pour les produits a demande reguliere.",
        codeSnippets: [
          {
            language: "python",
            code: "from prophet import Prophet\nimport pandas as pd\n\ndef train_forecast_model(sales_df: pd.DataFrame, reference: str, forecast_weeks: int = 12) -> dict:\n    ref_data = sales_df[sales_df[\"reference\"] == reference].copy()\n    ref_data = ref_data[ref_data[\"is_anomaly\"] == False]\n    prophet_df = ref_data.rename(columns={\"date\": \"ds\", \"quantite\": \"y\"})\n    model = Prophet(\n        yearly_seasonality=True,\n        weekly_seasonality=False,\n        daily_seasonality=False,\n        changepoint_prior_scale=0.05,\n        interval_width=0.9\n    )\n    model.add_country_holidays(country_name=\"FR\")\n    model.fit(prophet_df[[\"ds\", \"y\"]])\n    future = model.make_future_dataframe(periods=forecast_weeks, freq=\"W\")\n    forecast = model.predict(future)\n    future_forecast = forecast.tail(forecast_weeks)\n    return {\n        \"reference\": reference,\n        \"forecasts\": [\n            {\n                \"date\": row[\"ds\"].isoformat(),\n                \"predicted\": max(0, round(row[\"yhat\"])),\n                \"lower\": max(0, round(row[\"yhat_lower\"])),\n                \"upper\": max(0, round(row[\"yhat_upper\"]))\n            }\n            for _, row in future_forecast.iterrows()\n        ]\n    }",
            filename: "forecast_model.py",
          },
        ],
      },
      {
        title: "Agent LLM pour l'analyse contextuelle",
        content:
          "Le LLM apporte la dimension contextuelle que les modeles statistiques ne captent pas. Il analyse les tendances marche, les evenements sectoriels, et les signaux faibles pour ajuster les previsions statistiques et generer des recommandations actionnables.\n\nL'agent recoit les previsions Prophet, les niveaux de stock actuels, les delais fournisseurs, et le contexte marche. Il produit des recommandations de reapprovisionnement en tenant compte de contraintes invisibles au modele statistique : MOQ, remises volume, contraintes logistiques, et priorites strategiques.\n\nPour les nouveaux produits sans historique, l'agent LLM est particulierement precieux. Il peut estimer la demande initiale en se basant sur des produits similaires, les tendances du marche, et les retours qualitatifs de l'equipe commerciale.\n\nL'agent genere egalement des scenarios what-if : que se passe-t-il si un fournisseur est en retard de 2 semaines ? Si une promotion est lancee ? Si un concurrent baisse ses prix ? Ces scenarios aident les responsables supply chain a prendre des decisions eclairees.",
        codeSnippets: [
          {
            language: "python",
            code: "import openai\nfrom pydantic import BaseModel, Field\n\nclient = openai.OpenAI()\n\nclass ReplenishmentRecommendation(BaseModel):\n    reference: str\n    current_stock: int\n    predicted_demand_4w: int\n    recommended_order_qty: int\n    order_urgency: str\n    rupture_risk: str\n    reasoning: str\n    confidence: float = Field(ge=0.0, le=1.0)\n\ndef generate_recommendations(\n    forecasts: list[dict], stock_levels: dict,\n    lead_times: dict, market_context: str\n) -> list[ReplenishmentRecommendation]:\n    forecast_summary = \"\\n\".join(\n        \"- {ref} : prevision 4 sem={pred}, stock actuel={stock}, delai={lt} jours\".format(\n            ref=f[\"reference\"],\n            pred=sum(fc[\"predicted\"] for fc in f[\"forecasts\"][:4]),\n            stock=stock_levels.get(f[\"reference\"], 0),\n            lt=lead_times.get(f[\"reference\"], 14)\n        ) for f in forecasts\n    )\n    prompt = (\n        \"Tu es un expert en supply chain et gestion des stocks B2B.\\n\"\n        \"Analyse ces previsions et niveaux de stock.\\n\\n\"\n        \"Previsions et stocks:\\n{data}\\n\\n\"\n        \"Contexte marche:\\n{context}\\n\\n\"\n        \"Pour chaque reference, recommande: quantite a commander, urgence, \"\n        \"risque de rupture, et raisonnement.\\n\"\n        \"Retourne une liste JSON de recommandations.\"\n    ).format(data=forecast_summary, context=market_context)\n    response = client.chat.completions.create(\n        model=\"gpt-4.1\",\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n        response_format={\"type\": \"json_object\"}\n    )\n    import json\n    parsed = json.loads(response.choices[0].message.content)\n    return [ReplenishmentRecommendation(**r) for r in parsed[\"recommendations\"]]",
            filename: "recommendation_agent.py",
          },
        ],
      },
      {
        title: "API, alertes et deploiement",
        content:
          "Exposez le systeme de prevision via une API REST qui alimente votre ERP et votre dashboard supply chain. L'API permet de lancer des previsions a la demande, de consulter les recommandations en cours, et de simuler des scenarios.\n\nConfigurez un systeme d'alertes proactif. Chaque matin, le systeme compare les niveaux de stock avec les previsions de demande et envoie un email aux responsables supply chain avec la liste des references en risque de rupture. Les alertes critiques declenchent une notification Slack immediate.\n\nDeployez le pipeline de prevision comme un job batch quotidien sur AWS Lambda. Les previsions sont recalculees chaque nuit avec les dernieres donnees de vente et stockees dans PostgreSQL. Le dashboard affiche les previsions, niveaux de stock et recommandations en temps reel.\n\nMettez en place un feedback loop : chaque semaine, comparez les previsions passees avec les ventes reelles pour mesurer et ameliorer la precision. Ce mecanisme d'apprentissage continu permet d'affiner les modeles et de detecter rapidement une degradation de la qualite.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, Depends\nfrom fastapi.security import HTTPBearer\n\napp = FastAPI(title=\"Demand Forecasting Agent\")\nsecurity = HTTPBearer()\n\n@app.post(\"/api/forecast/run\")\nasync def run_forecast(params: dict, token=Depends(security)):\n    references = params.get(\"references\", [])\n    horizon_weeks = params.get(\"horizon_weeks\", 12)\n    sales_data = load_sales_data(references)\n    forecasts = []\n    for ref in sales_data[\"reference\"].unique():\n        forecast = train_forecast_model(sales_data, ref, horizon_weeks)\n        forecasts.append(forecast)\n    stock_levels = load_current_stock(references)\n    lead_times = load_lead_times(references)\n    market_context = fetch_market_context()\n    recommendations = generate_recommendations(\n        forecasts, stock_levels, lead_times, market_context\n    )\n    await save_forecasts(forecasts)\n    await save_recommendations(recommendations)\n    critical = [r for r in recommendations if r.rupture_risk == \"critique\"]\n    if critical:\n        await send_critical_alerts(critical)\n    return {\n        \"forecasts_count\": len(forecasts),\n        \"recommendations_count\": len(recommendations),\n        \"critical_alerts\": len(critical)\n    }\n\n@app.post(\"/api/forecast/scenario\")\nasync def what_if_scenario(scenario: dict, token=Depends(security)):\n    return await simulate_scenario(scenario)",
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les donnees de prevision ne contiennent generalement pas de PII. Les donnees de vente peuvent inclure des noms de clients : anonymisez-les avant envoi au LLM. Les donnees commerciales (volumes, prix) sont des secrets d'affaires : utilisez un LLM avec clauses de non-retention des donnees. Stockage chiffre AES-256.",
      auditLog: "Tracabilite complete de chaque cycle de prevision : horodatage, references traitees, version du modele Prophet, parametres du LLM, previsions generees, recommandations emises, decisions prises par les operateurs, et ecart prevision vs realite. Conservation 5 ans pour l'analyse retrospective.",
      humanInTheLoop: "Les recommandations de reapprovisionnement superieure a un seuil de valeur (configurable, par exemple 50K EUR) necessitent une validation par le responsable supply chain. Les alertes de surstockage avec recommandation de destockage sont toujours soumises a validation humaine.",
      monitoring: "Dashboard de performance : MAPE par famille de produits, biais de prevision, taux de rupture reel vs predit, taux de surstockage, valeur du stock moyen, taux de couverture, et evolution de la precision dans le temps. Alertes si le MAPE depasse 30% sur une famille de produits.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (quotidien 6h) -> Node PostgreSQL (extraction donnees ventes) -> Node Code (preparation donnees + modele Prophet) -> Node HTTP Request (API LLM - analyse contextuelle) -> Node Code (generation recommandations) -> Node Switch (alertes critiques ?) -> Branch critique: Node Slack (alerte rupture) -> Node PostgreSQL (sauvegarde) -> Node Email (rapport quotidien supply chain).",
      nodes: ["Cron Trigger (6h)", "PostgreSQL (donnees ventes)", "Code (Prophet forecast)", "HTTP Request (LLM contexte)", "Code (recommandations)", "Switch (criticite)", "Slack (alerte rupture)", "PostgreSQL (sauvegarde)", "Email (rapport quotidien)"],
      triggerType: "Cron (quotidien a 6h du matin)",
    },
    estimatedTime: "12-18h",
    difficulty: "Expert",
    sectors: ["Distribution", "Industrie", "E-commerce", "Agroalimentaire", "Pharmacie"],
    metiers: ["Supply Chain", "Logistique", "Achats", "Direction des Operations"],
    functions: ["Supply Chain"],
    metaTitle: "Agent IA de Prevision de Demande et Optimisation des Stocks -- Guide Expert",
    metaDescription:
      "Optimisez vos stocks avec un agent IA de prevision de demande. Modele Prophet, analyse contextuelle LLM et recommandations de reapprovisionnement automatiques. Tutoriel complet.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-generation-contrats",
    title: "Agent IA de Generation de Contrats",
    subtitle: "Generez automatiquement des contrats commerciaux personnalises a partir de modeles et de donnees CRM",
    problem:
      "Les entreprises perdent des heures a rediger manuellement des contrats commerciaux a partir de modeles Word ou PDF. Le processus implique du copier-coller de donnees clients depuis le CRM, la selection manuelle des clauses appropriees selon le type de contrat, et des allers-retours interminables avec le service juridique pour validation. Les erreurs sont frequentes : mauvais tarifs, clauses manquantes, donnees client obsoletes, fautes de nommage. Le service juridique devient un goulot d'etranglement car chaque contrat doit etre relu integralement. En moyenne, la generation d'un contrat prend 2 a 4 heures et mobilise 3 personnes differentes.",
    value:
      "Un agent IA connecte au CRM et a la base de modeles de contrats automatise l'integralite du processus. L'agent extrait les donnees client a jour (raison sociale, SIRET, adresse, contacts, historique commercial), selectionne le modele de contrat adapte au type de deal (licence SaaS, prestation de service, contrat cadre), remplit automatiquement toutes les variables, ajoute les clauses pertinentes selon le secteur et le montant du contrat, et genere un PDF pret a etre relu. La relecture juridique se reduit a une simple verification car la structure et les clauses obligatoires sont toujours presentes. Temps de generation reduit de 2-4h a 10 minutes.",
    inputs: [
      "Donnees client depuis le CRM (raison sociale, SIRET, contacts, historique)",
      "Type de contrat demande (licence, prestation, contrat cadre, NDA)",
      "Parametres commerciaux (montant, duree, conditions de paiement)",
      "Bibliotheque de modeles de contrats (.docx, .pdf)",
      "Base de clauses juridiques par categorie et secteur",
    ],
    outputs: [
      "Contrat PDF genere automatiquement avec toutes les variables remplies",
      "Version Word editable pour modifications manuelles si necessaire",
      "Checklist de conformite validee automatiquement",
      "Resume du contrat avec points d'attention pour le relecteur",
      "Journal de generation avec traçabilite des sources de donnees",
    ],
    risks: [
      "Erreur de remplissage de variables critiques (montants, dates, raison sociale)",
      "Hallucination du LLM sur des clauses juridiques inexistantes ou incorrectes",
      "Non-conformite RGPD lors du traitement des donnees personnelles des contacts",
      "Mauvaise selection de clauses entrainant un risque juridique pour l'entreprise",
      "Dependance a la disponibilite du CRM et des APIs de generation PDF",
    ],
    roiIndicatif:
      "Reduction de 80% du temps de redaction des contrats (de 2-4h a 10-15 minutes). Diminution de 95% des erreurs de saisie et de clauses manquantes. Gain de 60% sur le temps de relecture juridique. ROI estime : 2 ETP economises par an pour une equipe commerciale de 10 personnes.",
    recommendedStack: [
      { name: "Claude 3.5 Sonnet", category: "LLM" },
      { name: "n8n", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
    ],
    architectureDiagram: "+-------------+     +----------------+     +-------------+\n|    CRM      |---->|  API Agent     |---->|  Agent LLM  |\n| (Donnees)   |     |  (FastAPI)     |     |  (Analyse)  |\n+-------------+     +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  Modeles       |<----|  Moteur de  |\n                    |  Contrats      |     |  Clauses    |\n                    +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  PDF/Word      |<----|  Generateur |\n                    |  (Sortie)      |     |  Documents  |\n                    +----------------+     +-------------+",
    tutorial: [
      {
        title: "Configuration des modeles de contrats et de la base de clauses",
        content:
          "La premiere etape consiste a structurer votre bibliotheque de modeles de contrats. Chaque modele doit etre converti en un format exploitable par l'agent : un template avec des variables balisees. Nous utilisons le format Jinja2 pour les variables dans les documents.\n\nCommencez par identifier vos types de contrats les plus frequents : contrat de licence SaaS, contrat de prestation de services, accord-cadre, NDA. Pour chaque type, creez un modele de reference avec des variables clairement identifiees (nom_client, siret, montant_ht, duree_contrat, etc.).\n\nLa base de clauses est un element central du systeme. Chaque clause est stockee dans PostgreSQL avec des metadonnees : categorie (responsabilite, confidentialite, resiliation, SLA), secteur d'application, caractere obligatoire ou optionnel, et conditions d'inclusion automatique. Par exemple, une clause de conformite bancaire est automatiquement ajoutee pour les clients du secteur financier.\n\nInstallez les dependances Python necessaires et configurez l'acces a votre base de donnees PostgreSQL. Le schema de base comprend trois tables principales : contract_templates, clauses, et generated_contracts pour la traçabilite.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install fastapi uvicorn anthropic python-docx jinja2 psycopg2-binary pydantic python-dotenv weasyprint langfuse",
            filename: "terminal",
          },
          {
            language: "sql",
            code: "CREATE TABLE contract_templates (\n  id SERIAL PRIMARY KEY,\n  name VARCHAR(255) NOT NULL,\n  type VARCHAR(50) NOT NULL, -- 'licence_saas', 'prestation', 'cadre', 'nda'\n  template_content TEXT NOT NULL,\n  variables JSONB NOT NULL, -- liste des variables attendues\n  is_active BOOLEAN DEFAULT true,\n  created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE TABLE clauses (\n  id SERIAL PRIMARY KEY,\n  title VARCHAR(255) NOT NULL,\n  content TEXT NOT NULL,\n  category VARCHAR(100) NOT NULL, -- 'responsabilite', 'confidentialite', 'sla', 'resiliation'\n  sectors TEXT[] DEFAULT '{}', -- secteurs applicables\n  is_mandatory BOOLEAN DEFAULT false,\n  min_contract_amount DECIMAL, -- seuil de montant pour inclusion auto\n  created_at TIMESTAMP DEFAULT NOW()\n);\n\nCREATE TABLE generated_contracts (\n  id SERIAL PRIMARY KEY,\n  client_id VARCHAR(100) NOT NULL,\n  template_id INTEGER REFERENCES contract_templates(id),\n  variables_used JSONB NOT NULL,\n  clauses_included INTEGER[] NOT NULL,\n  pdf_path VARCHAR(500),\n  status VARCHAR(50) DEFAULT 'draft', -- 'draft', 'review', 'approved', 'signed'\n  generated_by VARCHAR(100),\n  reviewed_by VARCHAR(100),\n  created_at TIMESTAMP DEFAULT NOW()\n);",
            filename: "schema.sql",
          },
          {
            language: "python",
            code: "from pydantic import BaseModel\nfrom typing import Optional\nfrom enum import Enum\n\nclass ContractType(str, Enum):\n    LICENCE_SAAS = \"licence_saas\"\n    PRESTATION = \"prestation\"\n    CADRE = \"cadre\"\n    NDA = \"nda\"\n\nclass ContractRequest(BaseModel):\n    client_id: str\n    contract_type: ContractType\n    montant_ht: float\n    duree_mois: int\n    conditions_paiement: str = \"30 jours fin de mois\"\n    clauses_supplementaires: Optional[list[str]] = None\n\nclass ClientData(BaseModel):\n    raison_sociale: str\n    siret: str\n    adresse: str\n    contact_nom: str\n    contact_email: str\n    contact_telephone: str\n    secteur: str\n    historique_ca: Optional[float] = None",
            filename: "models.py",
          },
        ],
      },
      {
        title: "Integration CRM et extraction des donnees client",
        content:
          "L'agent doit pouvoir interroger votre CRM pour recuperer les donnees client a jour. Nous implementons un connecteur generique qui supporte les CRM les plus courants (Salesforce, HubSpot, Pipedrive) via leurs APIs respectives. Le connecteur abstrait les differences entre les CRM derriere une interface unifiee.\n\nLorsqu'une demande de generation de contrat arrive, l'agent commence par extraire toutes les informations necessaires depuis le CRM : donnees legales de l'entreprise (raison sociale, SIRET, adresse du siege), coordonnees du contact signataire, historique commercial (CA cumule, nombre de contrats precedents), et le secteur d'activite du client. Ces informations sont validees avant d'etre injectees dans le contrat.\n\nL'agent LLM intervient ici pour analyser le contexte du deal et enrichir la requete. A partir de la description commerciale du deal dans le CRM, il identifie les besoins specifiques qui doivent se refleter dans le contrat : perimetre fonctionnel, niveaux de SLA attendus, conditions particulieres negociees.\n\nUn systeme de cache est mis en place pour eviter les appels CRM redondants. Les donnees client sont cachees pendant 1 heure avec invalidation automatique en cas de mise a jour dans le CRM via webhook.",
        codeSnippets: [
          {
            language: "python",
            code: "import httpx\nimport os\nfrom models import ClientData\nfrom functools import lru_cache\n\nclass CRMConnector:\n    def __init__(self):\n        self.base_url = os.getenv(\"CRM_API_URL\")\n        self.api_key = os.getenv(\"CRM_API_KEY\")\n        self.headers = {\"Authorization\": f\"Bearer {self.api_key}\"}\n\n    async def get_client_data(self, client_id: str) -> ClientData:\n        async with httpx.AsyncClient() as client:\n            response = await client.get(\n                f\"{self.base_url}/contacts/{client_id}\",\n                headers=self.headers,\n            )\n            response.raise_for_status()\n            data = response.json()\n\n            company_response = await client.get(\n                f\"{self.base_url}/companies/{data['company_id']}\",\n                headers=self.headers,\n            )\n            company = company_response.json()\n\n            return ClientData(\n                raison_sociale=company[\"name\"],\n                siret=company.get(\"siret\", \"\"),\n                adresse=company.get(\"address\", \"\"),\n                contact_nom=f\"{data['first_name']} {data['last_name']}\",\n                contact_email=data[\"email\"],\n                contact_telephone=data.get(\"phone\", \"\"),\n                secteur=company.get(\"industry\", \"Autres\"),\n                historique_ca=company.get(\"total_revenue\", 0),\n            )\n\n    async def get_deal_context(self, deal_id: str) -> dict:\n        async with httpx.AsyncClient() as client:\n            response = await client.get(\n                f\"{self.base_url}/deals/{deal_id}\",\n                headers=self.headers,\n            )\n            deal = response.json()\n            return {\n                \"description\": deal.get(\"description\", \"\"),\n                \"montant\": deal.get(\"amount\", 0),\n                \"produits\": deal.get(\"line_items\", []),\n                \"notes\": deal.get(\"notes\", \"\"),\n            }",
            filename: "crm_connector.py",
          },
          {
            language: "python",
            code: "from anthropic import Anthropic\nimport json\nimport os\n\nanthopic_client = Anthropic(api_key=os.getenv(\"ANTHROPIC_API_KEY\"))\n\nasync def analyze_deal_for_clauses(deal_context: dict, client_data: ClientData) -> dict:\n    prompt = f\"\"\"Analyse le contexte commercial suivant et identifie les elements\n    contractuels importants.\n\n    Client : {client_data.raison_sociale} (secteur : {client_data.secteur})\n    Description du deal : {deal_context['description']}\n    Montant : {deal_context['montant']} EUR HT\n    Produits : {json.dumps(deal_context['produits'], ensure_ascii=False)}\n\n    Reponds en JSON avec les champs suivants :\n    - clauses_recommandees: liste de categories de clauses a inclure\n    - sla_niveau: \"standard\" | \"premium\" | \"enterprise\"\n    - conditions_particulieres: liste de conditions specifiques identifiees\n    - risques_identifies: liste de points de vigilance juridique\"\"\"\n\n    response = anthopic_client.messages.create(\n        model=\"claude-3-5-sonnet-latest\",\n        max_tokens=1024,\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n    )\n    return json.loads(response.content[0].text)",
            filename: "deal_analyzer.py",
          },
        ],
      },
      {
        title: "Moteur de selection de clauses et assemblage du contrat",
        content:
          "Le moteur de clauses est le composant qui assemble intelligemment le contrat final. Il combine le modele de base, les variables remplies avec les donnees CRM, et les clauses selectionnees par l'agent LLM. La selection des clauses suit un algorithme en trois etapes.\n\nPremierement, les clauses obligatoires sont systematiquement incluses : elles correspondent aux mentions legales requises et aux clauses standard de votre entreprise. Deuxiemement, les clauses sectorielles sont ajoutees en fonction du secteur du client (conformite bancaire pour la finance, protection des donnees de sante pour le medical, etc.). Troisiemement, l'agent LLM recommande des clauses supplementaires basees sur l'analyse du deal.\n\nL'assemblage final utilise Jinja2 pour le remplissage des variables et python-docx pour la manipulation du document Word. Chaque variable est validee avant insertion : les montants sont formates en euros avec separateur de milliers, les dates suivent le format francais, les raisons sociales sont en majuscules conformement aux usages juridiques.\n\nUn systeme de validation pre-generation verifie la coherence du contrat : toutes les variables obligatoires sont remplies, les montants sont positifs, la duree est raisonnable, les clauses ne se contredisent pas. En cas d'incoherence, l'agent signale le probleme et propose une correction.",
        codeSnippets: [
          {
            language: "python",
            code: "import psycopg2\nimport psycopg2.extras\nfrom jinja2 import Template\nfrom datetime import datetime, timedelta\nfrom models import ContractRequest, ClientData\n\nclass ClauseEngine:\n    def __init__(self, db_connection):\n        self.conn = db_connection\n\n    def get_mandatory_clauses(self) -> list[dict]:\n        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:\n            cur.execute(\n                \"SELECT * FROM clauses WHERE is_mandatory = true ORDER BY category\"\n            )\n            return cur.fetchall()\n\n    def get_sector_clauses(self, secteur: str) -> list[dict]:\n        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:\n            cur.execute(\n                \"SELECT * FROM clauses WHERE %s = ANY(sectors) AND is_mandatory = false\",\n                (secteur,),\n            )\n            return cur.fetchall()\n\n    def get_amount_clauses(self, montant: float) -> list[dict]:\n        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:\n            cur.execute(\n                \"SELECT * FROM clauses WHERE min_contract_amount IS NOT NULL AND min_contract_amount <= %s AND is_mandatory = false\",\n                (montant,),\n            )\n            return cur.fetchall()\n\n    def select_clauses(\n        self, client_data: ClientData, request: ContractRequest, ai_recommendations: list[str]\n    ) -> list[dict]:\n        clauses = []\n        clauses.extend(self.get_mandatory_clauses())\n        clauses.extend(self.get_sector_clauses(client_data.secteur))\n        clauses.extend(self.get_amount_clauses(request.montant_ht))\n\n        if ai_recommendations:\n            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:\n                cur.execute(\n                    \"SELECT * FROM clauses WHERE category = ANY(%s)\",\n                    (ai_recommendations,),\n                )\n                clauses.extend(cur.fetchall())\n\n        seen_ids = set()\n        unique_clauses = []\n        for c in clauses:\n            if c[\"id\"] not in seen_ids:\n                seen_ids.add(c[\"id\"])\n                unique_clauses.append(c)\n        return unique_clauses",
            filename: "clause_engine.py",
          },
          {
            language: "python",
            code: "from jinja2 import Template\nfrom datetime import datetime, timedelta\nimport locale\n\nlocale.setlocale(locale.LC_ALL, \"fr_FR.UTF-8\")\n\ndef prepare_variables(client_data: ClientData, request: ContractRequest) -> dict:\n    date_debut = datetime.now()\n    date_fin = date_debut + timedelta(days=request.duree_mois * 30)\n\n    return {\n        \"raison_sociale\": client_data.raison_sociale.upper(),\n        \"siret\": client_data.siret,\n        \"adresse_siege\": client_data.adresse,\n        \"contact_nom\": client_data.contact_nom,\n        \"contact_email\": client_data.contact_email,\n        \"montant_ht\": locale.format_string(\"%.2f\", request.montant_ht, grouping=True),\n        \"montant_tva\": locale.format_string(\"%.2f\", request.montant_ht * 0.20, grouping=True),\n        \"montant_ttc\": locale.format_string(\"%.2f\", request.montant_ht * 1.20, grouping=True),\n        \"duree_mois\": str(request.duree_mois),\n        \"date_debut\": date_debut.strftime(\"%d/%m/%Y\"),\n        \"date_fin\": date_fin.strftime(\"%d/%m/%Y\"),\n        \"conditions_paiement\": request.conditions_paiement,\n        \"date_generation\": datetime.now().strftime(\"%d/%m/%Y a %H:%M\"),\n    }\n\ndef validate_variables(variables: dict, required_fields: list[str]) -> list[str]:\n    errors = []\n    for field in required_fields:\n        if field not in variables or not variables[field]:\n            errors.append(f\"Variable obligatoire manquante : {field}\")\n    if not variables.get(\"siret\") or len(variables[\"siret\"].replace(\" \", \"\")) != 14:\n        errors.append(\"Numero SIRET invalide\")\n    return errors",
            filename: "variable_engine.py",
          },
        ],
      },
      {
        title: "Generation PDF et deploiement de l'agent",
        content:
          "La derniere etape consiste a generer le document final en PDF a partir du contrat assemble. Nous utilisons python-docx pour creer le document Word intermediaire avec une mise en page professionnelle, puis WeasyPrint pour la conversion en PDF.\n\nLe document genere inclut automatiquement : une page de garde avec les logos des deux parties, un sommaire cliquable, les articles numerotes avec les clauses selectionnees, les annexes techniques le cas echeant, et un bloc de signatures en derniere page. La mise en forme respecte les standards juridiques francais.\n\nDeployez l'API sur Vercel avec les fonctions serverless. Creez un endpoint POST /api/generate-contract qui recoit une requete ContractRequest, orchestre l'ensemble du processus (extraction CRM, analyse LLM, selection clauses, assemblage, generation PDF) et retourne le document genere.\n\nIntegrez Langfuse pour le monitoring de la qualite. Chaque generation est tracee avec les metriques cles : temps de generation total, nombre de clauses selectionnees, score de confiance de l'analyse LLM, et statut de validation. Configurez des alertes si le temps de generation depasse 60 secondes ou si des erreurs de validation sont detectees.",
        codeSnippets: [
          {
            language: "python",
            code: "from docx import Document\nfrom docx.shared import Pt, Inches, Cm\nfrom docx.enum.text import WD_ALIGN_PARAGRAPH\nimport weasyprint\nimport tempfile\nimport os\n\nclass ContractGenerator:\n    def __init__(self, templates_dir: str):\n        self.templates_dir = templates_dir\n\n    def generate_docx(self, template_name: str, variables: dict, clauses: list[dict]) -> str:\n        template_path = os.path.join(self.templates_dir, f\"{template_name}.docx\")\n        doc = Document(template_path)\n\n        for paragraph in doc.paragraphs:\n            for key, value in variables.items():\n                placeholder = \"{{\" + key + \"}}\"\n                if placeholder in paragraph.text:\n                    paragraph.text = paragraph.text.replace(placeholder, str(value))\n\n        clauses_by_category = {}\n        for clause in clauses:\n            cat = clause[\"category\"]\n            if cat not in clauses_by_category:\n                clauses_by_category[cat] = []\n            clauses_by_category[cat].append(clause)\n\n        article_num = 1\n        for category, cat_clauses in clauses_by_category.items():\n            heading = doc.add_heading(f\"Article {article_num} - {category.title()}\", level=2)\n            for clause in cat_clauses:\n                doc.add_heading(clause[\"title\"], level=3)\n                doc.add_paragraph(clause[\"content\"])\n            article_num += 1\n\n        doc.add_page_break()\n        signatures = doc.add_paragraph()\n        signatures.alignment = WD_ALIGN_PARAGRAPH.CENTER\n        signatures.add_run(\"Fait en deux exemplaires originaux\\n\\n\").bold = True\n        signatures.add_run(f\"Pour le Prestataire\\n\\n\\n_________________________\\n\\n\")\n        signatures.add_run(f\"Pour le Client : {variables['raison_sociale']}\\n\\n\\n_________________________\")\n\n        output_path = tempfile.mktemp(suffix=\".docx\")\n        doc.save(output_path)\n        return output_path\n\n    def convert_to_pdf(self, docx_path: str) -> str:\n        pdf_path = docx_path.replace(\".docx\", \".pdf\")\n        html_content = self._docx_to_html(docx_path)\n        weasyprint.HTML(string=html_content).write_pdf(pdf_path)\n        return pdf_path",
            filename: "contract_generator.py",
          },
          {
            language: "python",
            code: "from fastapi import FastAPI, HTTPException\nfrom langfuse import Langfuse\nfrom models import ContractRequest\nfrom crm_connector import CRMConnector\nfrom deal_analyzer import analyze_deal_for_clauses\nfrom clause_engine import ClauseEngine\nfrom variable_engine import prepare_variables, validate_variables\nfrom contract_generator import ContractGenerator\nimport psycopg2\nimport os\nimport time\n\napp = FastAPI(title=\"Agent Generation Contrats\")\nlangfuse = Langfuse()\ncrm = CRMConnector()\n\n@app.post(\"/api/generate-contract\")\nasync def generate_contract(request: ContractRequest):\n    trace = langfuse.trace(name=\"contract-generation\", input=request.dict())\n    start_time = time.time()\n\n    try:\n        client_data = await crm.get_client_data(request.client_id)\n        trace.span(name=\"crm-extraction\", input={\"client_id\": request.client_id})\n\n        deal_context = await crm.get_deal_context(request.client_id)\n        analysis = await analyze_deal_for_clauses(deal_context, client_data)\n        trace.span(name=\"llm-analysis\", output=analysis)\n\n        conn = psycopg2.connect(os.getenv(\"DATABASE_URL\"))\n        engine = ClauseEngine(conn)\n        clauses = engine.select_clauses(\n            client_data, request, analysis.get(\"clauses_recommandees\", [])\n        )\n\n        variables = prepare_variables(client_data, request)\n        errors = validate_variables(variables, [\"raison_sociale\", \"siret\", \"montant_ht\"])\n        if errors:\n            raise HTTPException(400, detail={\"errors\": errors})\n\n        generator = ContractGenerator(templates_dir=\"./templates\")\n        docx_path = generator.generate_docx(\n            request.contract_type.value, variables, clauses\n        )\n        pdf_path = generator.convert_to_pdf(docx_path)\n\n        duration = time.time() - start_time\n        trace.span(name=\"generation-complete\", output={\n            \"duration_seconds\": duration,\n            \"clauses_count\": len(clauses),\n            \"pdf_path\": pdf_path,\n        })\n\n        return {\n            \"status\": \"success\",\n            \"pdf_url\": f\"/downloads/{os.path.basename(pdf_path)}\",\n            \"clauses_included\": len(clauses),\n            \"generation_time\": f\"{duration:.1f}s\",\n            \"review_checklist\": analysis.get(\"risques_identifies\", []),\n        }\n    except Exception as e:\n        trace.span(name=\"error\", output={\"error\": str(e)})\n        raise HTTPException(500, detail=str(e))",
            filename: "main.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les contrats contiennent des donnees personnelles sensibles (noms, adresses, SIRET, coordonnees). Les donnees sont chiffrees en transit (TLS 1.3) et au repos (AES-256). Les documents generes sont stockes dans un bucket S3 chiffre avec acces restreint par role IAM. Les donnees envoyees au LLM sont anonymisees : les SIRET et numeros de telephone sont pseudonymises avant l'appel API. Conformite RGPD assuree avec droit a l'effacement des contrats et donnees associees sur demande. Retention limitee a 5 ans conformement aux obligations legales.",
      auditLog: "Chaque generation de contrat est integralement tracee : identifiant unique de generation, horodatage, utilisateur demandeur, client concerne, modele utilise, variables injectees, clauses selectionnees (avec justification de selection), temps de generation, et statut de validation. Tous les appels au LLM sont logges dans Langfuse avec les prompts et reponses. Les modifications post-generation sont versionees. Retention des logs de 24 mois minimum.",
      humanInTheLoop: "Chaque contrat genere passe obligatoirement par une etape de validation humaine avant envoi au client. Les contrats depassant un seuil de montant configurable (par defaut 50 000 EUR) necessitent une double validation (commercial + juridique). Les clauses ajoutees par recommandation IA sont marquees visuellement dans le document pour attirer l'attention du relecteur. Un workflow d'approbation est integre avec notifications par email et Slack.",
      monitoring: "Dashboard temps reel dans Langfuse : nombre de contrats generes par jour, temps moyen de generation, taux d'erreurs de validation, repartition par type de contrat, top des clauses les plus utilisees. Alertes configurees sur : temps de generation superieur a 60 secondes, taux d'erreur superieur a 5%, echec de connexion CRM. Rapport hebdomadaire automatique envoye a l'equipe juridique.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (demande de contrat depuis le CRM) -> Node HTTP Request (extraction donnees client CRM) -> Node Code (preparation des variables) -> Node HTTP Request (analyse LLM Claude pour selection clauses) -> Node Postgres (recuperation clauses) -> Node Code (assemblage contrat et validation) -> Node HTTP Request (generation PDF) -> Node IF (validation OK ?) -> Branch OK : Node Email (envoi au relecteur) + Node Slack (notification) -> Branch Erreur : Node Slack (alerte equipe).",
      nodes: ["Webhook (demande CRM)", "HTTP Request (CRM API)", "Code (preparation variables)", "HTTP Request (Claude LLM)", "Postgres (clauses)", "Code (assemblage)", "HTTP Request (generation PDF)", "IF (validation)", "Email (envoi relecteur)", "Slack (notification)"],
      triggerType: "Webhook (demande depuis CRM ou formulaire interne)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["B2B SaaS", "Services", "Audit", "Assurance"],
    metiers: ["Conformite", "Commercial"],
    functions: ["Legal", "Sales"],
    metaTitle: "Agent IA de Generation de Contrats -- Guide Complet",
    metaDescription:
      "Automatisez la generation de contrats commerciaux avec un agent IA connecte a votre CRM. Modeles, clauses intelligentes et PDF en 10 minutes. Tutoriel complet.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-audit-securite-code",
    title: "Agent IA d'Audit de Securite de Code",
    subtitle: "Analysez automatiquement votre code source pour detecter les vulnerabilites OWASP et les failles de securite",
    problem:
      "Les audits de securite du code sont traditionnellement realises de maniere ponctuelle, souvent uniquement avant les mises en production majeures. Les revues manuelles sont lentes (plusieurs jours pour une application moyenne), couteuses (consultants specialises) et inconsistantes (dependantes de l'expertise individuelle du revieweur). Les developpeurs introduisent des vulnerabilites sans le savoir : injections SQL, failles XSS, gestion incorrecte des secrets, dependances obsoletes avec des CVE connues. Les outils SAST classiques generent trop de faux positifs et ne comprennent pas le contexte metier du code, ce qui conduit les equipes a ignorer les alertes.",
    value:
      "Un agent IA analyse chaque Pull Request en temps reel et identifie les vulnerabilites du Top 10 OWASP : injections SQL, XSS, authentification cassee, exposition de donnees sensibles, mauvaise configuration de securite. L'agent comprend le contexte du code grace a l'analyse AST (Abstract Syntax Tree) et fournit non seulement l'alerte mais aussi une explication detaillee de la faille, un exemple d'exploitation, et un snippet de code corrige. Les dependances sont verifiees contre les bases CVE. Le taux de faux positifs est reduit de 80% par rapport aux outils SAST classiques grace a la comprehension contextuelle du LLM. La securite devient continue plutot que ponctuelle.",
    inputs: [
      "Code source des Pull Requests (diff et fichiers complets)",
      "Historique des vulnerabilites detectees et corrigees",
      "Configuration des regles de securite specifiques au projet",
      "Base de donnees CVE pour les dependances (NVD, GitHub Advisory)",
      "Fichiers de configuration (Dockerfile, CI/CD, env) pour analyse de la surface d'attaque",
    ],
    outputs: [
      "Rapport de securite detaille par Pull Request avec niveau de severite",
      "Commentaires inline sur les lignes de code vulnerables dans la PR",
      "Suggestions de correction avec snippets de code prets a l'emploi",
      "Score de securite global du repository avec evolution temporelle",
      "Tableau de bord des vulnerabilites par categorie OWASP et par equipe",
    ],
    risks: [
      "Faux negatifs : vulnerabilites critiques non detectees par le LLM",
      "Faux positifs excessifs entrainant une fatigue d'alerte chez les developpeurs",
      "Exposition du code source proprietaire au fournisseur LLM cloud",
      "Latence d'analyse bloquant le pipeline CI/CD",
      "Hallucination du LLM sur des vulnerabilites inexistantes ou des corrections incorrectes",
    ],
    roiIndicatif:
      "Detection de 85% des vulnerabilites supplementaires avant mise en production. Reduction de 60% du temps de revue de securite manuelle. Diminution de 90% des incidents de securite en production. ROI estime : prevention de 3 a 5 incidents de securite majeurs par an, representant chacun un cout moyen de 50 000 a 200 000 EUR.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "PostgreSQL", category: "Database" },
      { name: "AWS Lambda", category: "Hosting" },
      { name: "Datadog", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + CodeLlama", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-------------+     +----------------+     +-------------+\n|  GitHub     |---->|  Webhook API   |---->|  Agent LLM  |\n|  (PR Event) |     |  (Lambda)      |     |  (Analyse)  |\n+-------------+     +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  AST Parser    |<----|  Moteur de  |\n                    |  + CVE DB      |     |  Detection  |\n                    +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  GitHub API    |<----|  Rapporteur |\n                    |  (Commentaires)|     |  (Resultats)|\n                    +----------------+     +-------------+",
    tutorial: [
      {
        title: "Integration Git et reception des Pull Requests",
        content:
          "La premiere etape consiste a configurer un webhook GitHub (ou GitLab) pour recevoir les evenements de Pull Request. Chaque fois qu'une PR est ouverte ou mise a jour, votre agent recoit le diff du code modifie et peut demarrer l'analyse de securite automatiquement.\n\nCreez une application GitHub ou un webhook de repository qui envoie les evenements 'pull_request' a votre API. Le webhook inclut les metadonnees de la PR (auteur, branche, description) et un lien vers le diff. Vous devez ensuite utiliser l'API GitHub pour recuperer le contenu complet des fichiers modifies, car le diff seul ne suffit pas pour comprendre le contexte.\n\nConfigurez l'authentification avec un token GitHub App pour acceder aux repositories prives. Le token doit avoir les permissions 'contents:read' et 'pull_requests:write' pour lire le code et poster des commentaires de revue.\n\nInstallez les dependances Python. Nous utilisons tree-sitter pour l'analyse AST multi-langages (Python, JavaScript, TypeScript, Java, Go), LangChain pour l'orchestration de l'agent, et les clients GitHub et OpenAI.",
        codeSnippets: [
          {
            language: "bash",
            code: "pip install fastapi uvicorn langchain openai httpx python-dotenv pydantic tree-sitter tree-sitter-python tree-sitter-javascript psycopg2-binary datadog-api-client PyGithub",
            filename: "terminal",
          },
          {
            language: "python",
            code: "from fastapi import FastAPI, Request, HTTPException\nfrom github import Github, GithubIntegration\nimport hmac\nimport hashlib\nimport os\nimport json\n\napp = FastAPI(title=\"Agent Audit Securite Code\")\n\ndef verify_github_signature(payload: bytes, signature: str) -> bool:\n    secret = os.getenv(\"GITHUB_WEBHOOK_SECRET\").encode()\n    expected = \"sha256=\" + hmac.new(secret, payload, hashlib.sha256).hexdigest()\n    return hmac.compare_digest(expected, signature)\n\n@app.post(\"/webhook/github\")\nasync def github_webhook(request: Request):\n    payload = await request.body()\n    signature = request.headers.get(\"X-Hub-Signature-256\", \"\")\n\n    if not verify_github_signature(payload, signature):\n        raise HTTPException(403, \"Signature invalide\")\n\n    event = request.headers.get(\"X-GitHub-Event\")\n    data = json.loads(payload)\n\n    if event == \"pull_request\" and data[\"action\"] in [\"opened\", \"synchronize\"]:\n        pr_number = data[\"pull_request\"][\"number\"]\n        repo_name = data[\"repository\"][\"full_name\"]\n        base_sha = data[\"pull_request\"][\"base\"][\"sha\"]\n        head_sha = data[\"pull_request\"][\"head\"][\"sha\"]\n\n        await analyze_pull_request(repo_name, pr_number, base_sha, head_sha)\n\n    return {\"status\": \"ok\"}\n\nasync def get_pr_files(repo_name: str, pr_number: int) -> list[dict]:\n    g = Github(os.getenv(\"GITHUB_TOKEN\"))\n    repo = g.get_repo(repo_name)\n    pr = repo.get_pull(pr_number)\n    files = []\n    for f in pr.get_files():\n        if f.status != \"removed\" and f.filename.endswith(\n            (\".py\", \".js\", \".ts\", \".java\", \".go\", \".rb\", \".php\")\n        ):\n            content = repo.get_contents(f.filename, ref=pr.head.sha)\n            files.append({\n                \"filename\": f.filename,\n                \"patch\": f.patch,\n                \"content\": content.decoded_content.decode(\"utf-8\"),\n                \"additions\": f.additions,\n                \"language\": f.filename.rsplit(\".\", 1)[-1],\n            })\n    return files",
            filename: "webhook_handler.py",
          },
        ],
      },
      {
        title: "Analyse AST et detection des patterns de vulnerabilites",
        content:
          "L'analyse AST (Abstract Syntax Tree) est la cle pour reduire les faux positifs. Plutot que d'analyser le code comme du texte brut, nous le parsons en arbre syntaxique pour comprendre la structure : quelles fonctions sont appelees, comment les donnees circulent des entrees utilisateur vers les requetes base de donnees, ou sont geres les secrets.\n\nNous utilisons tree-sitter, un parseur incremental multi-langages, pour generer l'AST de chaque fichier. Le parseur identifie les noeuds critiques : appels de fonctions de base de donnees, manipulation de HTML, lecture de variables d'environnement, imports de bibliotheques cryptographiques, gestion des entrees utilisateur.\n\nLe moteur de detection combine deux approches : une analyse statique basee sur des patterns connus (regles codees en dur pour les injections SQL, XSS, etc.) et une analyse contextuelle par le LLM qui comprend la logique metier du code. L'analyse statique est rapide et precise pour les patterns simples, tandis que le LLM excelle sur les vulnerabilites subtiles qui necessitent de comprendre le flux de donnees.\n\nChaque vulnerabilite detectee est classee selon le framework OWASP Top 10 avec un niveau de severite (critique, haute, moyenne, basse, informationnelle). Le contexte complet est fourni : ligne de code concernee, explication de la faille, scenario d'exploitation, et proposition de correction.",
        codeSnippets: [
          {
            language: "python",
            code: "import tree_sitter_python as tspython\nimport tree_sitter_javascript as tsjavascript\nfrom tree_sitter import Language, Parser\nfrom dataclasses import dataclass\nfrom enum import Enum\n\nclass Severity(str, Enum):\n    CRITICAL = \"critique\"\n    HIGH = \"haute\"\n    MEDIUM = \"moyenne\"\n    LOW = \"basse\"\n    INFO = \"informationnelle\"\n\nclass OWASPCategory(str, Enum):\n    INJECTION = \"A03:2021 - Injection\"\n    BROKEN_AUTH = \"A07:2021 - Identification et authentification\"\n    SENSITIVE_DATA = \"A02:2021 - Defaillances cryptographiques\"\n    XSS = \"A03:2021 - Injection (XSS)\"\n    MISCONFIG = \"A05:2021 - Mauvaise configuration de securite\"\n    VULNERABLE_DEPS = \"A06:2021 - Composants vulnerables et obsoletes\"\n    BROKEN_ACCESS = \"A01:2021 - Controle d'acces defaillant\"\n\n@dataclass\nclass Vulnerability:\n    file: str\n    line: int\n    severity: Severity\n    category: OWASPCategory\n    title: str\n    description: str\n    exploit_scenario: str\n    fix_suggestion: str\n    code_snippet: str\n    fixed_code: str\n\nclass ASTAnalyzer:\n    def __init__(self):\n        self.py_language = Language(tspython.language())\n        self.js_language = Language(tsjavascript.language())\n        self.parser = Parser()\n\n    def analyze_python(self, code: str, filename: str) -> list[dict]:\n        self.parser.language = self.py_language\n        tree = self.parser.parse(bytes(code, \"utf-8\"))\n        findings = []\n\n        self._check_sql_injection(tree.root_node, code, filename, findings)\n        self._check_hardcoded_secrets(tree.root_node, code, filename, findings)\n        self._check_unsafe_deserialization(tree.root_node, code, filename, findings)\n        self._check_command_injection(tree.root_node, code, filename, findings)\n\n        return findings\n\n    def _check_sql_injection(self, node, code: str, filename: str, findings: list):\n        if node.type == \"call\":\n            func_text = code[node.start_byte:node.end_byte]\n            if any(kw in func_text.lower() for kw in [\"execute(\", \"raw(\", \"rawquery(\"]):\n                if \"f\\\"\" in func_text or \"format(\" in func_text or \"%s\" not in func_text:\n                    if \".format(\" in func_text or \"f\\\"\" in func_text or \"f'\" in func_text:\n                        findings.append({\n                            \"type\": \"sql_injection\",\n                            \"line\": node.start_point[0] + 1,\n                            \"file\": filename,\n                            \"code\": func_text,\n                            \"severity\": \"critique\",\n                        })\n        for child in node.children:\n            self._check_sql_injection(child, code, filename, findings)",
            filename: "ast_analyzer.py",
          },
          {
            language: "python",
            code: "import re\nfrom ast_analyzer import Vulnerability, Severity, OWASPCategory\n\nSECRET_PATTERNS = [\n    (r'(?i)(password|passwd|pwd|secret|token|api_key|apikey)\\s*=\\s*[\"\\'][^\"\\']{8,}[\"\\']', \"Secret code en dur\"),\n    (r'(?i)(aws_access_key_id|aws_secret_access_key)\\s*=\\s*[\"\\'][A-Za-z0-9/+=]{20,}[\"\\']', \"Cle AWS en dur\"),\n    (r'(?i)bearer\\s+[A-Za-z0-9\\-._~+/]+=*', \"Token Bearer en dur\"),\n    (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', \"Cle privee dans le code\"),\n]\n\nUNSAFE_FUNCTIONS = {\n    \"python\": {\n        \"eval(\": (\"Execution de code arbitraire\", Severity.CRITICAL),\n        \"exec(\": (\"Execution de code arbitraire\", Severity.CRITICAL),\n        \"pickle.loads(\": (\"Deserialisation non securisee\", Severity.HIGH),\n        \"yaml.load(\": (\"Deserialisation YAML non securisee\", Severity.HIGH),\n        \"subprocess.call(shell=True\": (\"Injection de commande OS\", Severity.CRITICAL),\n        \"os.system(\": (\"Injection de commande OS\", Severity.CRITICAL),\n        \"__import__(\": (\"Import dynamique non securise\", Severity.MEDIUM),\n    },\n    \"javascript\": {\n        \"eval(\": (\"Execution de code arbitraire\", Severity.CRITICAL),\n        \"innerHTML\": (\"Risque de XSS\", Severity.HIGH),\n        \"document.write(\": (\"Risque de XSS\", Severity.HIGH),\n        \"dangerouslySetInnerHTML\": (\"Risque de XSS dans React\", Severity.HIGH),\n        \"child_process.exec(\": (\"Injection de commande OS\", Severity.CRITICAL),\n    },\n}\n\ndef scan_for_secrets(code: str, filename: str) -> list[Vulnerability]:\n    vulnerabilities = []\n    lines = code.split(\"\\n\")\n    for i, line in enumerate(lines):\n        for pattern, description in SECRET_PATTERNS:\n            if re.search(pattern, line):\n                vulnerabilities.append(Vulnerability(\n                    file=filename,\n                    line=i + 1,\n                    severity=Severity.CRITICAL,\n                    category=OWASPCategory.SENSITIVE_DATA,\n                    title=description,\n                    description=f\"Un secret semble etre code en dur dans le fichier. \"\n                                f\"Les secrets doivent etre stockes dans des variables d'environnement \"\n                                f\"ou un gestionnaire de secrets (Vault, AWS Secrets Manager).\",\n                    exploit_scenario=f\"Un attaquant ayant acces au code source (fuite, depot public) \"\n                                     f\"peut extraire le secret et l'utiliser pour acceder aux systemes proteges.\",\n                    fix_suggestion=f\"Deplacez le secret dans une variable d'environnement \"\n                                   f\"et utilisez os.getenv() pour le lire.\",\n                    code_snippet=line.strip(),\n                    fixed_code=\"# Utiliser une variable d'environnement\\nimport os\\nvalue = os.getenv('SECRET_NAME')\",\n                ))\n    return vulnerabilities",
            filename: "pattern_scanner.py",
          },
        ],
      },
      {
        title: "Analyse contextuelle par LLM et generation des rapports",
        content:
          "L'analyse contextuelle par LLM est le differenciateur principal de cet agent par rapport aux outils SAST classiques. Le LLM comprend la logique metier du code et peut identifier des vulnerabilites subtiles que les regles statiques ne detectent pas : validation insuffisante des roles dans un middleware d'autorisation, fuite de donnees sensibles via des logs trop verbeux, conditions de course dans la gestion de sessions.\n\nL'agent envoie au LLM le code complet des fichiers modifies avec le contexte des fichiers adjacents (imports, classes parentes, middleware). Le prompt est structure pour guider l'analyse selon les categories OWASP et exiger des reponses structurees en JSON avec tous les champs necessaires au rapport.\n\nPour chaque vulnerabilite detectee, le LLM genere une explication pedagogique destinee au developpeur : pourquoi c'est dangereux, comment un attaquant pourrait l'exploiter, et un snippet de code corrige pret a copier-coller. Cette approche educative ameliore la securite a long terme en formant les developpeurs.\n\nLe rapport de securite est publie directement comme commentaire de revue sur la Pull Request GitHub. Les vulnerabilites critiques et hautes bloquent automatiquement le merge via un check status. Les vulnerabilites moyennes et basses sont des avertissements informatifs.",
        codeSnippets: [
          {
            language: "python",
            code: "from langchain.chat_models import ChatOpenAI\nfrom langchain.prompts import ChatPromptTemplate\nimport json\nfrom ast_analyzer import Vulnerability, Severity, OWASPCategory\n\nSECURITY_PROMPT = ChatPromptTemplate.from_messages([\n    (\"system\", \"\"\"Tu es un expert en securite applicative specialise dans l'audit de code.\nAnalyse le code source fourni et identifie les vulnerabilites de securite.\n\nPour chaque vulnerabilite trouvee, reponds en JSON avec ce schema :\n{{\n  \"vulnerabilities\": [\n    {{\n      \"line\": <numero de ligne>,\n      \"severity\": \"critique\" | \"haute\" | \"moyenne\" | \"basse\",\n      \"category\": \"<categorie OWASP>\",\n      \"title\": \"<titre court>\",\n      \"description\": \"<explication detaillee en francais>\",\n      \"exploit_scenario\": \"<scenario d'exploitation>\",\n      \"fix_suggestion\": \"<explication de la correction>\",\n      \"fixed_code\": \"<code corrige>\"\n    }}\n  ],\n  \"security_score\": <score de 0 a 100>,\n  \"summary\": \"<resume en francais>\"\n}}\n\nConcentre-toi sur : injections SQL/NoSQL, XSS, authentification/autorisation,\nexposition de donnees sensibles, configuration de securite, composants vulnerables.\nNe signale que les vrais problemes, evite les faux positifs.\"\"\"),\n    (\"human\", \"\"\"Fichier : {filename} (langage : {language})\n\nDiff de la PR :\n```\n{patch}\n```\n\nContenu complet du fichier :\n```{language}\n{content}\n```\n\nContexte du projet : {project_context}\n\nAnalyse les modifications et le fichier complet pour detecter les vulnerabilites.\"\"\")\n])\n\nclass LLMSecurityAnalyzer:\n    def __init__(self):\n        self.llm = ChatOpenAI(model=\"gpt-4.1\", temperature=0, max_tokens=4096)\n        self.chain = SECURITY_PROMPT | self.llm\n\n    async def analyze_file(self, file_data: dict, project_context: str) -> dict:\n        result = await self.chain.ainvoke({\n            \"filename\": file_data[\"filename\"],\n            \"language\": file_data[\"language\"],\n            \"patch\": file_data[\"patch\"],\n            \"content\": file_data[\"content\"],\n            \"project_context\": project_context,\n        })\n        return json.loads(result.content)\n\n    async def analyze_pr(self, files: list[dict], project_context: str) -> list[dict]:\n        all_vulns = []\n        total_score = 0\n        for f in files:\n            analysis = await self.analyze_file(f, project_context)\n            for vuln in analysis.get(\"vulnerabilities\", []):\n                vuln[\"file\"] = f[\"filename\"]\n                all_vulns.append(vuln)\n            total_score += analysis.get(\"security_score\", 100)\n\n        avg_score = total_score // max(len(files), 1)\n        return {\n            \"vulnerabilities\": sorted(all_vulns, key=lambda v: [\"critique\", \"haute\", \"moyenne\", \"basse\"].index(v[\"severity\"])),\n            \"security_score\": avg_score,\n            \"files_analyzed\": len(files),\n            \"total_vulnerabilities\": len(all_vulns),\n        }",
            filename: "llm_analyzer.py",
          },
          {
            language: "python",
            code: "from github import Github\nimport os\n\nclass GitHubReporter:\n    def __init__(self):\n        self.g = Github(os.getenv(\"GITHUB_TOKEN\"))\n\n    def post_review(self, repo_name: str, pr_number: int, analysis: dict):\n        repo = self.g.get_repo(repo_name)\n        pr = repo.get_pull(pr_number)\n\n        score = analysis[\"security_score\"]\n        vulns = analysis[\"vulnerabilities\"]\n        critiques = [v for v in vulns if v[\"severity\"] == \"critique\"]\n        hautes = [v for v in vulns if v[\"severity\"] == \"haute\"]\n\n        status = \"APPROVE\" if not critiques and not hautes else \"REQUEST_CHANGES\"\n        icon = \"\\u2705\" if status == \"APPROVE\" else \"\\u274c\"\n\n        body = f\"## {icon} Rapport d'Audit de Securite\\n\\n\"\n        body += f\"**Score de securite : {score}/100**\\n\\n\"\n        body += f\"| Severite | Nombre |\\n|---|---|\\n\"\n        body += f\"| Critique | {len(critiques)} |\\n\"\n        body += f\"| Haute | {len(hautes)} |\\n\"\n        body += f\"| Moyenne | {len([v for v in vulns if v['severity'] == 'moyenne'])} |\\n\"\n        body += f\"| Basse | {len([v for v in vulns if v['severity'] == 'basse'])} |\\n\\n\"\n\n        for vuln in vulns:\n            severity_badge = {\"critique\": \"\\U0001f534\", \"haute\": \"\\U0001f7e0\", \"moyenne\": \"\\U0001f7e1\", \"basse\": \"\\U0001f535\"}\n            badge = severity_badge.get(vuln[\"severity\"], \"\")\n            body += f\"### {badge} {vuln['title']}\\n\"\n            body += f\"**Fichier :** `{vuln['file']}` ligne {vuln['line']}\\n\"\n            body += f\"**Categorie :** {vuln['category']}\\n\\n\"\n            body += f\"{vuln['description']}\\n\\n\"\n            body += f\"**Scenario d'exploitation :** {vuln['exploit_scenario']}\\n\\n\"\n            body += f\"**Correction suggeree :**\\n```\\n{vuln['fixed_code']}\\n```\\n\\n---\\n\\n\"\n\n        comments = []\n        for vuln in vulns:\n            if vuln.get(\"line\"):\n                comments.append({\n                    \"path\": vuln[\"file\"],\n                    \"line\": vuln[\"line\"],\n                    \"body\": f\"**{vuln['severity'].upper()}** - {vuln['title']}\\n\\n{vuln['description']}\\n\\nCorrection :\\n```\\n{vuln['fixed_code']}\\n```\",\n                })\n\n        pr.create_review(body=body, event=status, comments=comments)\n\n        commit = repo.get_commit(pr.head.sha)\n        state = \"failure\" if critiques or hautes else \"success\"\n        commit.create_status(\n            state=state,\n            target_url=f\"https://votre-dashboard.com/pr/{pr_number}\",\n            description=f\"Score: {score}/100 - {len(vulns)} vulnerabilite(s) detectee(s)\",\n            context=\"security-audit/ai-agent\",\n        )",
            filename: "github_reporter.py",
          },
        ],
      },
      {
        title: "Integration CI/CD et deploiement en production",
        content:
          "L'integration dans votre pipeline CI/CD est essentielle pour que l'audit de securite soit systematique et non contournable. Configurez l'agent comme un check obligatoire sur vos branches protegees : aucune PR ne peut etre mergee si des vulnerabilites critiques ou hautes sont detectees.\n\nDeployez l'API sur AWS Lambda pour beneficier du scaling automatique. Les analyses de securite peuvent etre couteuses en temps (30 secondes a 2 minutes par PR selon la taille), mais Lambda gere les executions paralleles si plusieurs PR sont ouvertes simultanement. Configurez un timeout de 5 minutes et 512 Mo de memoire.\n\nLa verification des dependances est automatisee via l'analyse des fichiers requirements.txt, package.json, go.mod et pom.xml. L'agent interroge la base NVD (National Vulnerability Database) et GitHub Advisory pour identifier les CVE connues et recommander les mises a jour.\n\nMettez en place le monitoring avec Datadog : tracez chaque analyse (duree, nombre de vulnerabilites, score), creez des dashboards d'evolution de la securite par repository et par equipe, et configurez des alertes si le score moyen de securite descend sous un seuil configurable. Un rapport hebdomadaire est genere automatiquement pour le RSSI.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI\nfrom webhook_handler import github_webhook, get_pr_files\nfrom ast_analyzer import ASTAnalyzer\nfrom pattern_scanner import scan_for_secrets, UNSAFE_FUNCTIONS\nfrom llm_analyzer import LLMSecurityAnalyzer\nfrom github_reporter import GitHubReporter\nfrom datadog_api_client import Configuration, ApiClient\nfrom datadog_api_client.v2.api.metrics_api import MetricsApi\nimport time\nimport os\n\napp = FastAPI(title=\"Agent Audit Securite Code\")\nast_analyzer = ASTAnalyzer()\nllm_analyzer = LLMSecurityAnalyzer()\nreporter = GitHubReporter()\n\nasync def analyze_pull_request(repo_name: str, pr_number: int, base_sha: str, head_sha: str):\n    start_time = time.time()\n\n    files = await get_pr_files(repo_name, pr_number)\n    if not files:\n        return\n\n    all_static_findings = []\n    for f in files:\n        secrets = scan_for_secrets(f[\"content\"], f[\"filename\"])\n        all_static_findings.extend(secrets)\n\n        if f[\"language\"] == \"py\":\n            ast_findings = ast_analyzer.analyze_python(f[\"content\"], f[\"filename\"])\n            all_static_findings.extend(ast_findings)\n\n    project_context = f\"Repository: {repo_name}, Langages: {set(f['language'] for f in files)}\"\n    llm_analysis = await llm_analyzer.analyze_pr(files, project_context)\n\n    combined_vulns = llm_analysis[\"vulnerabilities\"]\n    for finding in all_static_findings:\n        if hasattr(finding, \"__dict__\"):\n            combined_vulns.append(vars(finding))\n        else:\n            combined_vulns.append(finding)\n\n    seen = set()\n    unique_vulns = []\n    for v in combined_vulns:\n        key = (v.get(\"file\", \"\"), v.get(\"line\", 0), v.get(\"title\", \"\"))\n        if key not in seen:\n            seen.add(key)\n            unique_vulns.append(v)\n\n    final_analysis = {\n        \"vulnerabilities\": unique_vulns,\n        \"security_score\": llm_analysis[\"security_score\"],\n        \"files_analyzed\": len(files),\n        \"total_vulnerabilities\": len(unique_vulns),\n    }\n\n    reporter.post_review(repo_name, pr_number, final_analysis)\n\n    duration = time.time() - start_time\n    send_metrics(repo_name, pr_number, final_analysis, duration)\n\ndef send_metrics(repo_name: str, pr_number: int, analysis: dict, duration: float):\n    configuration = Configuration()\n    with ApiClient(configuration) as api_client:\n        api = MetricsApi(api_client)\n        # Metriques : score, nombre de vulnerabilites, duree d'analyse\n        print(f\"[Metrics] Repo={repo_name} PR={pr_number} Score={analysis['security_score']} \"\n              f\"Vulns={analysis['total_vulnerabilities']} Duration={duration:.1f}s\")",
            filename: "main.py",
          },
          {
            language: "yaml",
            code: "# Configuration GitHub Actions pour integrer l'audit de securite\nname: Security Audit\n\non:\n  pull_request:\n    types: [opened, synchronize]\n\njobs:\n  security-scan:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n        with:\n          fetch-depth: 0\n\n      - name: Declencher l'audit de securite IA\n        run: |\n          curl -X POST ${{ secrets.SECURITY_AGENT_URL }}/webhook/github \\\n            -H \"Content-Type: application/json\" \\\n            -H \"X-GitHub-Event: pull_request\" \\\n            -H \"X-Hub-Signature-256: $(echo -n '${{ toJSON(github.event) }}' | openssl dgst -sha256 -hmac ${{ secrets.WEBHOOK_SECRET }} | cut -d' ' -f2)\" \\\n            -d '${{ toJSON(github.event) }}'\n\n      - name: Attendre le resultat de l'audit\n        run: |\n          for i in $(seq 1 30); do\n            STATUS=$(gh api repos/${{ github.repository }}/commits/${{ github.event.pull_request.head.sha }}/statuses | jq -r '.[] | select(.context==\"security-audit/ai-agent\") | .state' | head -1)\n            if [ \"$STATUS\" = \"success\" ] || [ \"$STATUS\" = \"failure\" ]; then\n              echo \"Audit termine avec statut: $STATUS\"\n              [ \"$STATUS\" = \"success\" ] && exit 0 || exit 1\n            fi\n            echo \"Audit en cours... tentative $i/30\"\n            sleep 10\n          done\n          echo \"Timeout de l'audit de securite\"\n          exit 1\n        env:\n          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}",
            filename: ".github/workflows/security-audit.yml",
          },
          {
            language: "python",
            code: "import httpx\nfrom dataclasses import dataclass\n\n@dataclass\nclass CVEResult:\n    cve_id: str\n    severity: str\n    package: str\n    affected_versions: str\n    fixed_version: str\n    description: str\n\nasync def check_python_deps(requirements_content: str) -> list[CVEResult]:\n    vulnerabilities = []\n    lines = requirements_content.strip().split(\"\\n\")\n\n    for line in lines:\n        line = line.strip()\n        if not line or line.startswith(\"#\"):\n            continue\n        parts = line.split(\"==\")\n        if len(parts) != 2:\n            continue\n        package, version = parts[0].strip(), parts[1].strip()\n\n        async with httpx.AsyncClient() as client:\n            response = await client.get(\n                f\"https://api.github.com/advisories\",\n                params={\"ecosystem\": \"pip\", \"package\": package},\n                headers={\"Accept\": \"application/vnd.github+json\"},\n            )\n            if response.status_code == 200:\n                advisories = response.json()\n                for adv in advisories:\n                    for vuln in adv.get(\"vulnerabilities\", []):\n                        if is_version_affected(version, vuln.get(\"vulnerable_version_range\", \"\")):\n                            vulnerabilities.append(CVEResult(\n                                cve_id=adv.get(\"cve_id\", \"N/A\"),\n                                severity=adv.get(\"severity\", \"unknown\"),\n                                package=package,\n                                affected_versions=vuln.get(\"vulnerable_version_range\", \"\"),\n                                fixed_version=vuln.get(\"first_patched_version\", {}).get(\"identifier\", \"N/A\"),\n                                description=adv.get(\"summary\", \"\"),\n                            ))\n    return vulnerabilities",
            filename: "dependency_checker.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Le code source est une propriete intellectuelle sensible. Pour les deployments on-premise, utilisez Ollama avec CodeLlama pour que le code ne quitte jamais votre infrastructure. En mode cloud, les appels au LLM sont effectues via des endpoints conformes SOC 2 avec chiffrement TLS 1.3. Aucun code n'est stocke cote LLM (zero data retention). Les rapports de vulnerabilites sont stockes dans PostgreSQL avec chiffrement AES-256 et acces restreint par role RBAC. Retention des rapports limitee a 24 mois avec purge automatique.",
      auditLog: "Chaque analyse de PR est integralement tracee : horodatage, repository, numero de PR, auteur, fichiers analyses, vulnerabilites detectees (avec ligne et severite), score de securite, decision (approve/block), duree d'analyse, version du modele LLM utilise. Les logs sont stockes dans PostgreSQL avec export possible vers un SIEM (Splunk, ELK). Retention de 36 mois pour conformite ISO 27001.",
      humanInTheLoop: "Les vulnerabilites critiques declenchent une notification immediate au RSSI et au lead developpeur via Slack et email. Un processus d'exception permet a un responsable securite habilite de forcer le merge d'une PR bloquee avec justification obligatoire (documentee dans l'audit log). Les faux positifs peuvent etre marques comme tels par le developpeur, creant une regle d'exclusion soumise a validation du RSSI.",
      monitoring: "Dashboard Datadog temps reel : score de securite moyen par repository et par equipe, evolution temporelle des vulnerabilites, top 10 des categories OWASP les plus frequentes, temps moyen d'analyse par PR, taux de faux positifs rapportes. Alertes configurees : score de securite moyen sous 70/100, vulnerabilite critique non corrigee depuis plus de 48h, echec de l'agent sur une PR. Rapport hebdomadaire automatique pour le RSSI avec tendances et recommandations.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (evenement GitHub PR) -> Node Code (extraction des fichiers modifies via GitHub API) -> Node Code (analyse AST et scan de patterns) -> Node HTTP Request (analyse LLM contextuelle) -> Node Merge (combinaison des resultats statiques et LLM) -> Node IF (vulnerabilites critiques ?) -> Branch critique : Node GitHub API (commenter PR + bloquer merge) + Node Slack (alerte RSSI) -> Branch OK : Node GitHub API (approuver PR) -> Node Postgres (sauvegarder rapport).",
      nodes: ["Webhook (GitHub PR)", "Code (extraction fichiers)", "Code (analyse AST)", "HTTP Request (LLM analyse)", "Merge (resultats)", "IF (severite critique)", "GitHub API (commentaire PR)", "Slack (alerte RSSI)", "GitHub API (approbation)", "Postgres (sauvegarde rapport)"],
      triggerType: "Webhook (evenement pull_request GitHub)",
    },
    estimatedTime: "14-20h",
    difficulty: "Expert",
    sectors: ["B2B SaaS", "Banque", "Tous secteurs"],
    metiers: ["IT", "DevOps"],
    functions: ["IT"],
    metaTitle: "Agent IA d'Audit de Securite de Code -- Guide Complet",
    metaDescription:
      "Deployez un agent IA pour auditer automatiquement la securite de votre code sur chaque Pull Request. Detection OWASP, analyse AST et corrections automatiques.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-resume-documents",
    title: "Agent IA de Resume de Documents",
    subtitle: "Resumez automatiquement des documents longs (rapports, contrats, etudes) en quelques secondes",
    problem:
      "Les professionnels de la banque, de l'assurance et de l'audit passent en moyenne 3 a 4 heures par jour a lire et synthetiser des documents longs : rapports financiers de 100+ pages, contrats juridiques complexes, etudes reglementaires, notes de conformite. Les informations cles sont noyees dans des paragraphes denses, ce qui retarde les prises de decision et augmente le risque d'omission d'une clause critique ou d'un risque cache. Les outils de recherche classiques (Ctrl+F) ne suffisent pas car ils ne comprennent pas le contexte semantique du document.",
    value:
      "Un agent IA ingere automatiquement les documents (PDF, Word, scans via OCR), les decoupe en sections logiques, extrait les points cles, et genere des resumes structures avec des bullet points hierarchises. Pour chaque document, l'agent identifie les risques, obligations, dates limites et montants importants. Il peut comparer plusieurs documents entre eux et produire des tableaux de synthese comparative. Les analystes economisent 3 a 4 heures par jour et la qualite des syntheses est homogene et auditable.",
    inputs: [
      "Documents PDF (rapports financiers, contrats, etudes)",
      "Documents Word (.docx, .doc)",
      "Documents scannes (images, PDF scannes via OCR)",
      "Modeles de resume personnalises (templates)",
      "Criteres d'extraction specifiques (risques, obligations, montants)",
    ],
    outputs: [
      "Resume structure avec bullet points hierarchises",
      "Liste des risques et obligations identifies",
      "Tableau comparatif multi-documents",
      "Extraction des dates limites et montants cles",
      "Document de synthese exportable (PDF, Word, Markdown)",
    ],
    risks: [
      "Omission d'une clause critique dans le resume",
      "Hallucination du LLM inventant des informations absentes du document",
      "Mauvaise interpretation de termes juridiques ou financiers techniques",
      "Qualite degradee sur les documents scannes de mauvaise qualite (OCR)",
      "Non-conformite RGPD si les documents contiennent des donnees personnelles",
    ],
    roiIndicatif:
      "Economie de 3 a 4 heures par analyste par jour. Reduction de 60% des erreurs d'omission dans les revues documentaires. Acceleration de 5x du temps de traitement des dossiers de conformite.",
    recommendedStack: [
      { name: "Claude 3.5 Sonnet", category: "LLM" },
      { name: "n8n", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
    ],
    architectureDiagram: "+-------------+     +----------------+     +-------------+\n|  Documents  |---->|  Ingestion     |---->|  OCR +       |\n|  (PDF/Word/ |     |  Pipeline      |     |  Extraction  |\n|   Scans)    |     |  (n8n)         |     |  (Tesseract) |\n+-------------+     +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  Supabase      |<----|  Chunking   |\n                    |  (Stockage +   |     |  Semantique |\n                    |   Embeddings)  |     +------+------+\n                    +----------------+            |\n                                           +------v------+\n                                           |  Agent LLM  |\n                                           |  (Resume)   |\n                                           +------+------+\n                                                  |\n                                           +------v------+\n                                           |  Export     |\n                                           |  (PDF/Word) |\n                                           +-------------+",
    tutorial: [
      {
        title: "Ingestion de documents et OCR",
        content:
          "La premiere etape consiste a mettre en place un pipeline d'ingestion capable de traiter differents formats de documents. Le systeme doit accepter des PDF natifs, des documents Word, et des scans necessitant un traitement OCR.\n\nPour les PDF natifs, utilisez la bibliotheque PyMuPDF (fitz) qui extrait le texte avec une excellente preservation de la structure (titres, paragraphes, tableaux). Pour les documents Word, python-docx permet d'extraire le contenu en preservant la hierarchie des titres.\n\nLes documents scannes necessitent un pre-traitement OCR. Tesseract, combine avec un pre-traitement d'image via Pillow, offre de bons resultats pour les documents en francais. Pour les scans de mauvaise qualite, ajoutez une etape de nettoyage d'image (binarisation, desinclinaison, suppression du bruit).\n\nConfigurez un endpoint d'upload dans votre API FastAPI qui detecte automatiquement le type de document et applique le pipeline de traitement adapte. Stockez les documents originaux et le texte extrait dans Supabase pour un acces ulterieur.",
        codeSnippets: [
          {
            language: "python",
            code: "import fitz  # PyMuPDF\nfrom docx import Document\nimport pytesseract\nfrom PIL import Image\nimport io\nfrom pathlib import Path\n\ndef extract_text_from_pdf(file_bytes: bytes) -> dict:\n    \"\"\"Extrait le texte d'un PDF natif avec structure.\"\"\"\n    doc = fitz.open(stream=file_bytes, filetype=\"pdf\")\n    pages = []\n    for page_num, page in enumerate(doc):\n        text = page.get_text(\"text\")\n        if len(text.strip()) < 50:\n            # Page probablement scannee, appliquer OCR\n            pix = page.get_pixmap(dpi=300)\n            img = Image.open(io.BytesIO(pix.tobytes(\"png\")))\n            text = pytesseract.image_to_string(img, lang=\"fra\")\n        pages.append({\"page\": page_num + 1, \"content\": text})\n    return {\"total_pages\": len(pages), \"pages\": pages}\n\ndef extract_text_from_docx(file_bytes: bytes) -> dict:\n    \"\"\"Extrait le texte d'un document Word avec hierarchie.\"\"\"\n    doc = Document(io.BytesIO(file_bytes))\n    sections = []\n    current_section = {\"title\": \"Introduction\", \"content\": []}\n    for para in doc.paragraphs:\n        if para.style.name.startswith(\"Heading\"):\n            if current_section[\"content\"]:\n                sections.append(current_section)\n            current_section = {\"title\": para.text, \"content\": []}\n        else:\n            if para.text.strip():\n                current_section[\"content\"].append(para.text)\n    sections.append(current_section)\n    return {\"sections\": sections}",
            filename: "document_ingestion.py",
          },
        ],
      },
      {
        title: "Strategie de decoupage (chunking) semantique",
        content:
          "Le decoupage des documents en morceaux (chunks) est une etape critique pour la qualite du resume. Un mauvais decoupage peut couper une idee en deux, perdre le contexte, ou generer des resumes incoherents. La strategie optimale combine decoupage structurel et semantique.\n\nLe decoupage structurel utilise les titres, sous-titres et sauts de section du document pour creer des chunks naturels. Chaque section conserve son contexte hierarchique (titre du chapitre, titre de la section) comme metadonnee.\n\nPour les documents sans structure claire, utilisez un decoupage par fenetre glissante avec chevauchement. Une taille de chunk de 1500-2000 tokens avec un chevauchement de 200 tokens offre un bon equilibre entre contexte et precision.\n\nStockez chaque chunk dans Supabase avec ses metadonnees (position dans le document, titre de section, page) et un embedding vectoriel pour permettre la recherche semantique ulterieure. L'embedding est genere via le modele all-MiniLM-L6-v2 de sentence-transformers, performant et rapide.\n\nImplementez un systeme de poids par section : les introductions, conclusions et sections executives recoivent un poids plus eleve car elles contiennent generalement les informations les plus importantes du document.",
        codeSnippets: [
          {
            language: "python",
            code: "from dataclasses import dataclass\nfrom sentence_transformers import SentenceTransformer\nimport re\n\n@dataclass\nclass DocumentChunk:\n    content: str\n    section_title: str\n    page_number: int\n    chunk_index: int\n    weight: float\n    embedding: list[float] = None\n\nembedding_model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n\nSECTION_WEIGHTS = {\n    \"resume\": 1.5, \"synthese\": 1.5, \"executive\": 1.5,\n    \"conclusion\": 1.3, \"introduction\": 1.2, \"recommandation\": 1.3,\n    \"risque\": 1.4, \"obligation\": 1.4,\n}\n\ndef compute_weight(section_title: str) -> float:\n    title_lower = section_title.lower()\n    for keyword, weight in SECTION_WEIGHTS.items():\n        if keyword in title_lower:\n            return weight\n    return 1.0\n\ndef chunk_document(pages: list[dict], max_tokens: int = 1500, overlap: int = 200) -> list[DocumentChunk]:\n    chunks = []\n    current_text = \"\"\n    current_title = \"Document\"\n    chunk_idx = 0\n    for page in pages:\n        paragraphs = page[\"content\"].split(\"\\n\\n\")\n        for para in paragraphs:\n            # Detecter les titres de section\n            if re.match(r\"^[A-Z0-9][.)]?\\s+[A-Z]\", para) and len(para) < 200:\n                if current_text.strip():\n                    chunk = DocumentChunk(\n                        content=current_text.strip(),\n                        section_title=current_title,\n                        page_number=page[\"page\"],\n                        chunk_index=chunk_idx,\n                        weight=compute_weight(current_title),\n                    )\n                    chunk.embedding = embedding_model.encode(chunk.content).tolist()\n                    chunks.append(chunk)\n                    chunk_idx += 1\n                current_title = para.strip()\n                current_text = \"\"\n            else:\n                current_text += para + \"\\n\\n\"\n    if current_text.strip():\n        chunk = DocumentChunk(\n            content=current_text.strip(),\n            section_title=current_title,\n            page_number=pages[-1][\"page\"],\n            chunk_index=chunk_idx,\n            weight=compute_weight(current_title),\n        )\n        chunk.embedding = embedding_model.encode(chunk.content).tolist()\n        chunks.append(chunk)\n    return chunks",
            filename: "chunking.py",
          },
        ],
      },
      {
        title: "Prompts de resume et extraction d'informations cles",
        content:
          "La qualite du resume depend directement de la qualite des prompts envoyes au LLM. Utilisez une approche en deux passes : d'abord un resume par chunk avec extraction des informations cles, puis une synthese globale qui consolide tous les resumes partiels.\n\nLe prompt de premiere passe demande au LLM de resumer chaque chunk en identifiant : les faits principaux, les chiffres cles, les risques mentionnes, les obligations ou engagements, et les dates limites. Le format de sortie est structure en JSON pour faciliter l'agregation.\n\nLa deuxieme passe recoit tous les resumes partiels et produit un resume global coherent. Le prompt insiste sur la deduplication (un meme fait peut apparaitre dans plusieurs chunks) et la hierarchisation par importance. Le resume final est structure avec un executive summary, des bullet points par theme, et une section risques/alertes.\n\nPour la comparaison multi-documents, un troisieme prompt analyse les resumes de plusieurs documents cote a cote et produit un tableau comparatif. Cette fonctionnalite est particulierement utile pour comparer des offres commerciales, des contrats concurrents, ou des versions successives d'un meme document.\n\nCalibrez la temperature du LLM a 0.1 pour maximiser la fidelite au document source. Toute information du resume doit etre tracable a un passage precis du document original.",
        codeSnippets: [
          {
            language: "python",
            code: "import anthropic\nimport json\nfrom typing import Optional\n\nclient = anthropic.Anthropic()\n\nCHUNK_SUMMARY_PROMPT = \"\"\"Tu es un analyste expert en synthese de documents professionnels.\nResume le passage suivant en extrayant les informations cles.\n\nSection: {section_title}\nContenu:\n---\n{content}\n---\n\nRetourne un JSON avec cette structure exacte:\n{{\n  \"resume\": \"Resume en 2-3 phrases\",\n  \"faits_cles\": [\"fait 1\", \"fait 2\"],\n  \"chiffres\": [\"montant ou statistique 1\"],\n  \"risques\": [\"risque identifie 1\"],\n  \"obligations\": [\"obligation ou engagement 1\"],\n  \"dates_limites\": [\"date et contexte 1\"],\n  \"importance\": \"haute/moyenne/basse\"\n}}\"\"\"\n\nGLOBAL_SUMMARY_PROMPT = \"\"\"Tu es un analyste senior. A partir des resumes partiels ci-dessous,\nproduis un resume global structure du document.\n\nResumes partiels:\n{partial_summaries}\n\nProduis un resume structure avec:\n1. **Synthese executive** (3-5 phrases)\n2. **Points cles** (bullet points hierarchises)\n3. **Chiffres importants** (tableau)\n4. **Risques et alertes** (liste priorisee)\n5. **Obligations et echeances** (liste chronologique)\n6. **Recommandations** (si applicables)\n\nReste strictement fidele au contenu des documents. Ne genere aucune information non presente dans les resumes partiels.\"\"\"\n\nasync def summarize_chunk(chunk: dict) -> dict:\n    prompt = CHUNK_SUMMARY_PROMPT.format(\n        section_title=chunk[\"section_title\"],\n        content=chunk[\"content\"][:3000],\n    )\n    message = client.messages.create(\n        model=\"claude-3-5-sonnet-20241022\",\n        max_tokens=1024,\n        temperature=0.1,\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n    )\n    return json.loads(message.content[0].text)\n\nasync def generate_global_summary(partial_summaries: list[dict]) -> str:\n    formatted = json.dumps(partial_summaries, ensure_ascii=False, indent=2)\n    prompt = GLOBAL_SUMMARY_PROMPT.format(partial_summaries=formatted)\n    message = client.messages.create(\n        model=\"claude-3-5-sonnet-20241022\",\n        max_tokens=4096,\n        temperature=0.1,\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n    )\n    return message.content[0].text",
            filename: "summarization.py",
          },
        ],
      },
      {
        title: "Formatage des sorties et deploiement",
        content:
          "Le resume genere doit etre exporte dans des formats exploitables par les utilisateurs finaux. Implementez trois formats de sortie : PDF professionnel avec mise en page soignee, document Word editable, et Markdown pour integration dans des outils collaboratifs.\n\nPour la generation PDF, utilisez la bibliotheque WeasyPrint qui convertit du HTML/CSS en PDF. Creez un template HTML avec le branding de l'entreprise, une table des matieres automatique, et une mise en page adaptee a l'impression. Les tableaux comparatifs sont particulierement importants a bien formater.\n\nDeployez l'API sur Vercel avec un endpoint d'upload et un endpoint de telechargement. L'upload declenche le pipeline complet (ingestion, chunking, resume, export) et retourne un identifiant de job. Le client poll l'endpoint de statut jusqu'a completion.\n\nMettez en place le monitoring avec Langfuse pour tracer chaque etape du pipeline : temps d'extraction, nombre de chunks, cout LLM par document, qualite estimee du resume. Configurez des alertes si le temps de traitement depasse un seuil ou si le taux d'erreur OCR est anormalement eleve.\n\nPour les tests, constituez un jeu de 20 documents annotes manuellement et mesurez le taux de couverture des informations cles (rappel) et la precision du resume (absence d'hallucinations). Visez un rappel superieur a 90% et une precision de 95%.",
        codeSnippets: [
          {
            language: "python",
            code: "from fastapi import FastAPI, UploadFile, BackgroundTasks\nfrom fastapi.responses import FileResponse\nimport uuid\nfrom supabase import create_client\nimport os\n\napp = FastAPI()\nsupabase = create_client(os.getenv(\"SUPABASE_URL\"), os.getenv(\"SUPABASE_KEY\"))\n\njobs: dict = {}\n\n@app.post(\"/api/summarize\")\nasync def upload_document(file: UploadFile, background_tasks: BackgroundTasks):\n    job_id = str(uuid.uuid4())\n    file_bytes = await file.read()\n    jobs[job_id] = {\"status\": \"processing\", \"filename\": file.filename}\n    # Stocker le document original dans Supabase\n    supabase.storage.from_(\"documents\").upload(\n        path=\"{}/{}\".format(job_id, file.filename),\n        file=file_bytes,\n    )\n    background_tasks.add_task(process_document, job_id, file_bytes, file.filename)\n    return {\"job_id\": job_id, \"status\": \"processing\"}\n\nasync def process_document(job_id: str, file_bytes: bytes, filename: str):\n    try:\n        # 1. Extraction du texte\n        if filename.endswith(\".pdf\"):\n            extracted = extract_text_from_pdf(file_bytes)\n        elif filename.endswith(\".docx\"):\n            extracted = extract_text_from_docx(file_bytes)\n        else:\n            raise ValueError(\"Format non supporte\")\n        # 2. Chunking\n        chunks = chunk_document(extracted[\"pages\"])\n        # 3. Resume par chunk\n        partial_summaries = []\n        for chunk in chunks:\n            summary = await summarize_chunk(chunk.__dict__)\n            partial_summaries.append(summary)\n        # 4. Resume global\n        global_summary = await generate_global_summary(partial_summaries)\n        # 5. Sauvegarde\n        supabase.table(\"summaries\").insert({\n            \"job_id\": job_id,\n            \"filename\": filename,\n            \"summary\": global_summary,\n            \"partial_summaries\": partial_summaries,\n            \"chunk_count\": len(chunks),\n        }).execute()\n        jobs[job_id] = {\"status\": \"completed\", \"summary\": global_summary}\n    except Exception as e:\n        jobs[job_id] = {\"status\": \"error\", \"error\": str(e)}\n\n@app.get(\"/api/summarize/{job_id}\")\nasync def get_summary_status(job_id: str):\n    if job_id not in jobs:\n        return {\"status\": \"not_found\"}\n    return jobs[job_id]",
            filename: "api_server.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les documents peuvent contenir des donnees personnelles sensibles (noms, adresses, numeros de comptes, RIB). Avant envoi au LLM, un module de detection PII (base sur Presidio de Microsoft) anonymise les donnees personnelles en les remplacant par des tokens generiques. Les documents originaux sont chiffres en AES-256 au repos dans Supabase. Politique de retention configurable par client avec purge automatique. Conformite RGPD avec registre de traitement.",
      auditLog: "Chaque traitement de document est trace de bout en bout : horodatage d'upload, hash SHA-256 du document original, nombre de pages et chunks, temps de traitement par etape, cout LLM consomme, identifiant de l'analyste, resume genere (versionne). Logs immutables stockes dans une table d'audit Supabase avec retention de 24 mois.",
      humanInTheLoop: "Pour les documents critiques (contrats > 1M EUR, rapports reglementaires), le resume genere est soumis a validation par un analyste senior avant diffusion. Un workflow de validation avec commentaires permet de corriger et enrichir le resume. Le mode revision compare le resume IA avec les annotations humaines pour ameliorer le modele en continu.",
      monitoring: "Dashboard Langfuse : temps de traitement moyen par type de document, cout LLM par resume, taux d'erreur OCR, taux de validation humaine, score de couverture des informations cles, volume de documents traites par jour, alertes sur les anomalies de qualite et les depassements de couts.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (upload document) -> Node Code (detection format PDF/Word/scan) -> Node Switch (format) -> Branch PDF: Node Code (extraction PyMuPDF) -> Branch Word: Node Code (extraction python-docx) -> Branch Scan: Node HTTP Request (OCR Tesseract) -> Node Code (chunking semantique) -> Node Loop (pour chaque chunk) -> Node HTTP Request (API Claude - resume chunk) -> Node Code (agregation resumes partiels) -> Node HTTP Request (API Claude - resume global) -> Node Code (formatage export) -> Node Supabase (sauvegarde) -> Node Email (notification analyste).",
      nodes: ["Webhook (upload)", "Code (detection format)", "Switch (PDF/Word/Scan)", "Code (extraction texte)", "HTTP Request (OCR)", "Code (chunking)", "Loop (chunks)", "HTTP Request (Claude resume)", "Code (agregation)", "HTTP Request (Claude synthese)", "Code (export PDF/Word)", "Supabase (sauvegarde)", "Email (notification)"],
      triggerType: "Webhook (upload de document)",
    },
    estimatedTime: "4-6h",
    difficulty: "Facile",
    sectors: ["Banque", "Assurance", "Audit", "Services"],
    metiers: ["Finance", "Conformite", "Risk Management"],
    functions: ["Finance"],
    metaTitle: "Agent IA de Resume de Documents -- Guide Complet",
    metaDescription:
      "Resumez automatiquement des documents longs (rapports, contrats, etudes) grace a un agent IA. Pipeline OCR, chunking semantique et resume structure. Tutoriel complet avec code.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-creation-fiches-produit",
    title: "Agent IA de Creation de Fiches Produit",
    subtitle: "Generez des fiches produit SEO-optimisees automatiquement a partir de donnees brutes fournisseur",
    problem:
      "Les equipes e-commerce doivent creer des fiches produit uniques et SEO-optimisees pour des centaines, voire des milliers de references (SKUs). Les donnees fournisseurs arrivent sous forme de fichiers CSV bruts avec des descriptions techniques minimalistes, souvent en anglais. Le copier-coller de ces donnees genere du contenu duplique penalise par Google, des descriptions generiques qui ne convertissent pas, et une incapacite a maintenir un ton de marque coherent sur l'ensemble du catalogue. Un redacteur produit manuellement 4 a 5 fiches par heure, creant un goulot d'etranglement majeur lors des lancements de collections ou de l'integration de nouveaux fournisseurs.",
    value:
      "Un agent IA ingere les donnees brutes fournisseur (CSV, flux API), les images produit et les descriptions concurrentes, puis genere automatiquement des fiches produit completes et uniques. Chaque fiche comprend un titre SEO-optimise, une description marketing engageante, des bullet points techniques, des balises meta (title, description), et des donnees structurees Schema.org. L'agent maintient le ton de voix de la marque de maniere coherente sur l'ensemble du catalogue. Le pipeline permet de generer plus de 100 fiches produit par heure contre 5 manuellement, avec un score SEO superieur de 40% en moyenne.",
    inputs: [
      "Fichiers CSV de donnees fournisseur (references, specs techniques)",
      "Images produit (pour analyse visuelle optionnelle)",
      "Guide de ton de marque et exemples de fiches existantes",
      "Mots-cles SEO cibles par categorie de produit",
      "Descriptions concurrentes (pour differentiation)",
    ],
    outputs: [
      "Titre produit SEO-optimise (H1)",
      "Description marketing engageante (200-400 mots)",
      "Bullet points techniques structures",
      "Balises meta (meta title, meta description)",
      "Donnees structurees Schema.org (JSON-LD)",
      "Suggestions de mots-cles secondaires et liens internes",
    ],
    risks: [
      "Descriptions generiques ou repetitives entre produits similaires",
      "Informations techniques incorrectes inventees par le LLM",
      "Ton de marque inconsistant sur un large volume de fiches",
      "Sur-optimisation SEO (keyword stuffing) penalisee par Google",
      "Non-conformite reglementaire sur les allegations produit (cosmetique, alimentaire)",
    ],
    roiIndicatif:
      "Generation de 100+ fiches produit par heure contre 5 manuellement (gain de productivite 20x). Amelioration de 40% du score SEO moyen des pages produit. Augmentation de 25% du taux de conversion grace a des descriptions plus engageantes.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "n8n", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "SQLite", category: "Database", isFree: true },
      { name: "n8n self-hosted", category: "Orchestration", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: "+-------------+     +----------------+     +-------------+\n|  Donnees    |---->|  Pipeline      |---->|  Enrichiss. |\n|  Fournisseur|     |  Ingestion     |     |  Donnees    |\n|  (CSV/API)  |     |  (n8n)         |     |  (SEO+Conc) |\n+-------------+     +----------------+     +------+------+\n                                                  |\n                    +----------------+     +------v------+\n                    |  Supabase      |<----|  Agent LLM  |\n                    |  (Catalogue +  |     |  (GPT-4.1)  |\n                    |   Fiches)      |     +------+------+\n                    +----------------+            |\n                                           +------v------+\n                                           |  Validation |\n                                           |  SEO Score  |\n                                           +------+------+\n                                                  |\n                                           +------v------+\n                                           |  Export     |\n                                           |  (CMS/API)  |\n                                           +-------------+",
    tutorial: [
      {
        title: "Ingestion et normalisation des donnees fournisseur",
        content:
          "La premiere etape du pipeline consiste a ingerer et normaliser les donnees brutes des fournisseurs. Ces donnees arrivent sous differents formats (CSV, Excel, API) avec des structures heterogenes : chaque fournisseur utilise ses propres noms de colonnes, unites de mesure et conventions.\n\nCreez un module d'ingestion flexible qui mappe automatiquement les colonnes fournisseur vers votre schema produit interne. Utilisez un fichier de configuration YAML par fournisseur pour definir les correspondances de colonnes. Pour les nouveaux fournisseurs, le LLM peut suggerer les mappings automatiquement.\n\nNormalisez les donnees : convertissez les unites (pouces vers cm, oz vers g), standardisez les noms de couleurs, nettoyez les caracteres speciaux, et traduisez les descriptions anglaises en francais si necessaire. Stockez les donnees brutes et normalisees dans Supabase pour tracabilite.\n\nImplementez une validation automatique : verifiez que les champs obligatoires sont presents (nom, prix, categorie), que les valeurs numeriques sont coherentes (poids positif, prix > 0), et que les images existent aux URLs indiquees. Les produits invalides sont mis en quarantaine pour correction manuelle.",
        codeSnippets: [
          {
            language: "python",
            code: "import csv\nimport io\nimport yaml\nfrom pydantic import BaseModel, Field, validator\nfrom typing import Optional\n\nclass RawProduct(BaseModel):\n    sku: str\n    name: str\n    description: str = \"\"\n    category: str\n    price: float = Field(gt=0)\n    brand: str = \"\"\n    features: list[str] = []\n    images: list[str] = []\n    weight: Optional[float] = None\n    dimensions: Optional[str] = None\n    material: Optional[str] = None\n    color: Optional[str] = None\n\n    @validator(\"name\")\n    def name_not_empty(cls, v):\n        if not v.strip():\n            raise ValueError(\"Le nom du produit ne peut pas etre vide\")\n        return v.strip()\n\ndef load_supplier_mapping(supplier_id: str) -> dict:\n    \"\"\"Charge le mapping de colonnes pour un fournisseur.\"\"\"\n    with open(\"mappings/{}.yaml\".format(supplier_id)) as f:\n        return yaml.safe_load(f)\n\ndef ingest_csv(file_bytes: bytes, supplier_id: str) -> list[RawProduct]:\n    \"\"\"Ingere un CSV fournisseur et retourne des produits normalises.\"\"\"\n    mapping = load_supplier_mapping(supplier_id)\n    reader = csv.DictReader(io.StringIO(file_bytes.decode(\"utf-8-sig\")))\n    products = []\n    errors = []\n    for row_num, row in enumerate(reader, start=2):\n        try:\n            mapped = {}\n            for our_field, supplier_field in mapping[\"columns\"].items():\n                mapped[our_field] = row.get(supplier_field, \"\")\n            # Normaliser le prix\n            if \"price\" in mapped:\n                mapped[\"price\"] = float(str(mapped[\"price\"]).replace(\",\", \".\").replace(\" \", \"\"))\n            # Normaliser les features\n            if \"features\" in mapped and isinstance(mapped[\"features\"], str):\n                mapped[\"features\"] = [f.strip() for f in mapped[\"features\"].split(\"|\") if f.strip()]\n            product = RawProduct(**mapped)\n            products.append(product)\n        except Exception as e:\n            errors.append({\"row\": row_num, \"error\": str(e)})\n    return products",
            filename: "ingestion.py",
          },
        ],
      },
      {
        title: "Generation de descriptions produit avec le LLM",
        content:
          "Le coeur du systeme est le prompt de generation de fiches produit. Le prompt doit produire un contenu unique, engageant et fidele aux donnees techniques du produit. La cle est de fournir au LLM un contexte riche : donnees produit, guide de ton de marque, exemples de fiches existantes, et mots-cles SEO cibles.\n\nStructurez le prompt en sections claires : d'abord les donnees brutes du produit, puis les directives de ton de marque (formel, decontracte, technique), les mots-cles SEO a integrer naturellement, et enfin le format de sortie attendu en JSON strict.\n\nPour eviter les descriptions repetitives entre produits similaires, incluez dans le prompt les descriptions deja generees pour les produits de la meme categorie. Le LLM utilisera cette information pour varier le vocabulaire et les angles d'accroche.\n\nImplementez un systeme de templates par categorie de produit. Un vetement ne se decrit pas comme un composant electronique. Chaque template definit les attributs a mettre en avant, le vocabulaire sectoriel, et les structures de phrases adaptees.\n\nLe format de sortie JSON garantit une integration facile avec n'importe quel CMS (Shopify, WooCommerce, PrestaShop, Magento). Chaque champ est valide par un schema Pydantic avant export.",
        codeSnippets: [
          {
            language: "python",
            code: "import openai\nimport json\nfrom pydantic import BaseModel\n\nclient = openai.OpenAI()\n\nclass ProductSheet(BaseModel):\n    seo_title: str\n    meta_title: str\n    meta_description: str\n    short_description: str\n    long_description: str\n    bullet_points: list[str]\n    keywords: list[str]\n    schema_org: dict\n\nGENERATION_PROMPT = \"\"\"Tu es un redacteur e-commerce expert en SEO francophone.\nGenere une fiche produit complete et unique a partir des donnees ci-dessous.\n\n## Donnees produit\n- Nom: {name}\n- Categorie: {category}\n- Marque: {brand}\n- Prix: {price} EUR\n- Caracteristiques: {features}\n- Materiau: {material}\n- Couleur: {color}\n- Dimensions: {dimensions}\n\n## Directives de marque\n{brand_guidelines}\n\n## Mots-cles SEO a integrer\n{seo_keywords}\n\n## Fiches existantes dans la meme categorie (pour varier le style)\n{existing_descriptions}\n\n## Consignes\n1. Le titre SEO doit contenir le mot-cle principal et faire moins de 60 caracteres\n2. La meta description doit faire entre 120 et 155 caracteres\n3. La description courte doit faire 1-2 phrases accrocheuses\n4. La description longue doit faire 200-400 mots, structuree en paragraphes\n5. Genere 5-8 bullet points techniques\n6. Integre les mots-cles naturellement, sans keyword stuffing\n7. Ne jamais inventer de caracteristiques non presentes dans les donnees\n8. Utiliser le vouvoiement\n\nRetourne un JSON avec: seo_title, meta_title, meta_description,\nshort_description, long_description, bullet_points, keywords, schema_org\"\"\"\n\ndef generate_product_sheet(\n    product: dict,\n    brand_guidelines: str,\n    seo_keywords: list[str],\n    existing_descriptions: list[str] = None,\n) -> ProductSheet:\n    existing = \"\\n---\\n\".join(existing_descriptions[:3]) if existing_descriptions else \"Aucune\"\n    prompt = GENERATION_PROMPT.format(\n        name=product[\"name\"],\n        category=product[\"category\"],\n        brand=product.get(\"brand\", \"N/A\"),\n        price=product[\"price\"],\n        features=\", \".join(product.get(\"features\", [])),\n        material=product.get(\"material\", \"N/A\"),\n        color=product.get(\"color\", \"N/A\"),\n        dimensions=product.get(\"dimensions\", \"N/A\"),\n        brand_guidelines=brand_guidelines,\n        seo_keywords=\", \".join(seo_keywords),\n        existing_descriptions=existing,\n    )\n    response = client.chat.completions.create(\n        model=\"gpt-4.1\",\n        temperature=0.7,\n        response_format={\"type\": \"json_object\"},\n        messages=[{\"role\": \"user\", \"content\": prompt}],\n    )\n    data = json.loads(response.choices[0].message.content)\n    return ProductSheet(**data)",
            filename: "product_generator.py",
          },
        ],
      },
      {
        title: "Optimisation SEO et validation qualite",
        content:
          "Une fois les fiches generees, un pipeline de validation automatique verifie la qualite SEO et la coherence du contenu. Ce pipeline agit comme un redacteur en chef automatise qui accepte, rejette ou demande une revision de chaque fiche.\n\nLe score SEO est calcule sur plusieurs criteres : presence du mot-cle principal dans le titre (H1), la meta description et le premier paragraphe, densite de mots-cles entre 1% et 3%, longueur de la meta description (120-155 caracteres), unicite du contenu (comparaison avec les fiches existantes via similarite cosinus).\n\nLa detection de contenu duplique est critique pour eviter les penalites Google. Calculez un embedding de chaque description generee et comparez-le avec les descriptions existantes dans votre catalogue. Si la similarite cosinus depasse 0.85, la fiche est rejetee et regeneree avec des directives de diversification renforcees.\n\nVerifiez egalement la fidelite aux donnees source : aucune caracteristique technique du resume ne doit etre absente des donnees fournisseur originales. Un module de fact-checking compare les bullet points generes avec les features brutes du produit.\n\nPour les secteurs reglementes (cosmetique, alimentaire, sante), ajoutez une couche de verification des allegations. Certaines formulations sont interdites sans certification (bio, hypoallergenique, therapeutique). Une liste noire de termes par categorie est appliquee automatiquement.",
        codeSnippets: [
          {
            language: "python",
            code: "from sentence_transformers import SentenceTransformer\nimport numpy as np\nfrom dataclasses import dataclass\n\nembedding_model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n\n@dataclass\nclass SEOScore:\n    total: float\n    keyword_in_title: bool\n    keyword_in_meta: bool\n    meta_length_ok: bool\n    keyword_density: float\n    uniqueness: float\n    issues: list[str]\n\ndef compute_seo_score(\n    sheet: dict,\n    target_keyword: str,\n    existing_embeddings: list[list[float]],\n) -> SEOScore:\n    issues = []\n    score = 0.0\n    # 1. Mot-cle dans le titre\n    kw_in_title = target_keyword.lower() in sheet[\"seo_title\"].lower()\n    if kw_in_title:\n        score += 20\n    else:\n        issues.append(\"Mot-cle principal absent du titre SEO\")\n    # 2. Mot-cle dans la meta description\n    kw_in_meta = target_keyword.lower() in sheet[\"meta_description\"].lower()\n    if kw_in_meta:\n        score += 15\n    else:\n        issues.append(\"Mot-cle principal absent de la meta description\")\n    # 3. Longueur meta description\n    meta_len = len(sheet[\"meta_description\"])\n    meta_ok = 120 <= meta_len <= 155\n    if meta_ok:\n        score += 15\n    else:\n        issues.append(\"Meta description: {} chars (attendu: 120-155)\".format(meta_len))\n    # 4. Densite de mots-cles\n    full_text = sheet[\"long_description\"].lower()\n    word_count = len(full_text.split())\n    kw_count = full_text.count(target_keyword.lower())\n    density = (kw_count / max(word_count, 1)) * 100\n    if 1.0 <= density <= 3.0:\n        score += 20\n    else:\n        issues.append(\"Densite mot-cle: {:.1f}% (attendu: 1-3%)\".format(density))\n    # 5. Unicite (similarite cosinus avec descriptions existantes)\n    desc_embedding = embedding_model.encode(sheet[\"long_description\"])\n    if existing_embeddings:\n        similarities = [\n            float(np.dot(desc_embedding, e) / (np.linalg.norm(desc_embedding) * np.linalg.norm(e)))\n            for e in existing_embeddings\n        ]\n        max_sim = max(similarities)\n        uniqueness = 1.0 - max_sim\n    else:\n        uniqueness = 1.0\n    if uniqueness > 0.15:\n        score += 30\n    else:\n        issues.append(\"Contenu trop similaire a une fiche existante (sim: {:.2f})\".format(1 - uniqueness))\n    return SEOScore(\n        total=score,\n        keyword_in_title=kw_in_title,\n        keyword_in_meta=kw_in_meta,\n        meta_length_ok=meta_ok,\n        keyword_density=density,\n        uniqueness=uniqueness,\n        issues=issues,\n    )",
            filename: "seo_validator.py",
          },
        ],
      },
      {
        title: "Pipeline de traitement en masse et deploiement",
        content:
          "Le pipeline de traitement en masse permet de generer des centaines de fiches produit en un seul batch. Le systeme gere la file d'attente, le rate limiting des API, la reprise sur erreur, et l'export vers le CMS cible.\n\nUtilisez un systeme de file d'attente base sur Supabase (table de jobs avec statut) pour gerer les batches. Chaque produit est un job independant qui peut etre retraite en cas d'echec. Le worker traite les produits en parallele (5-10 simultanes) tout en respectant les limites de taux de l'API OpenAI.\n\nL'export vers le CMS est automatise via les API natives. Pour Shopify, utilisez l'API GraphQL Admin pour creer ou mettre a jour les produits. Pour WooCommerce, utilisez l'API REST. Pour PrestaShop, le webservice XML. Le module d'export est pluggable pour supporter n'importe quel CMS.\n\nDeployez le dashboard de gestion sur Vercel. L'interface permet de lancer un batch, suivre la progression en temps reel, previsualiser les fiches generees, approuver ou rejeter individuellement, et declencher la publication vers le CMS.\n\nMettez en place le monitoring Langfuse pour suivre les metriques cles : nombre de fiches generees par jour, score SEO moyen, taux de rejet au premier passage, cout moyen par fiche (tokens LLM), et temps de traitement par batch. Configurez des alertes si le score SEO moyen descend sous 70/100.",
        codeSnippets: [
          {
            language: "python",
            code: "import asyncio\nfrom supabase import create_client\nimport os\nfrom datetime import datetime\n\nsupabase = create_client(os.getenv(\"SUPABASE_URL\"), os.getenv(\"SUPABASE_KEY\"))\n\nasync def process_batch(batch_id: str, products: list[dict], config: dict):\n    \"\"\"Traite un batch de produits en parallele.\"\"\"\n    semaphore = asyncio.Semaphore(5)  # Max 5 produits simultanes\n    brand_guidelines = config[\"brand_guidelines\"]\n    existing_descriptions = []\n    existing_embeddings = []\n    results = {\"success\": 0, \"failed\": 0, \"rejected_seo\": 0}\n\n    async def process_single(product: dict):\n        async with semaphore:\n            try:\n                # Recuperer les mots-cles SEO pour la categorie\n                seo_data = supabase.table(\"seo_keywords\").select(\"*\").eq(\n                    \"category\", product[\"category\"]\n                ).execute()\n                keywords = [k[\"keyword\"] for k in seo_data.data] if seo_data.data else []\n                # Generer la fiche produit\n                sheet = generate_product_sheet(\n                    product, brand_guidelines, keywords, existing_descriptions[-5:]\n                )\n                # Valider le score SEO\n                seo_score = compute_seo_score(\n                    sheet.model_dump(), keywords[0] if keywords else product[\"name\"],\n                    existing_embeddings,\n                )\n                if seo_score.total < 60:\n                    # Regenerer avec directives renforcees\n                    sheet = generate_product_sheet(\n                        product, brand_guidelines + \"\\nATTENTION: \" + \"; \".join(seo_score.issues),\n                        keywords, existing_descriptions[-5:]\n                    )\n                    seo_score = compute_seo_score(\n                        sheet.model_dump(), keywords[0] if keywords else product[\"name\"],\n                        existing_embeddings,\n                    )\n                # Sauvegarder\n                supabase.table(\"product_sheets\").insert({\n                    \"batch_id\": batch_id,\n                    \"sku\": product[\"sku\"],\n                    \"sheet\": sheet.model_dump(),\n                    \"seo_score\": seo_score.total,\n                    \"seo_issues\": seo_score.issues,\n                    \"status\": \"generated\",\n                    \"created_at\": datetime.utcnow().isoformat(),\n                }).execute()\n                existing_descriptions.append(sheet.long_description)\n                results[\"success\"] += 1\n            except Exception as e:\n                supabase.table(\"product_sheets\").insert({\n                    \"batch_id\": batch_id,\n                    \"sku\": product[\"sku\"],\n                    \"status\": \"error\",\n                    \"error\": str(e),\n                }).execute()\n                results[\"failed\"] += 1\n\n    tasks = [process_single(p) for p in products]\n    await asyncio.gather(*tasks)\n    # Mettre a jour le statut du batch\n    supabase.table(\"batches\").update({\n        \"status\": \"completed\",\n        \"results\": results,\n        \"completed_at\": datetime.utcnow().isoformat(),\n    }).eq(\"id\", batch_id).execute()\n    return results",
            filename: "batch_processor.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les donnees fournisseurs peuvent contenir des informations commerciales confidentielles (prix d'achat, marges, conditions negociees). Ces donnees sont stockees chiffrees dans Supabase et ne sont jamais envoyees au LLM. Seules les informations publiques du produit (nom, description, specs techniques) sont transmises a l'API. Les cles API sont stockees dans un vault securise (Supabase Vault). Conformite RGPD si les donnees contiennent des informations de contacts fournisseurs.",
      auditLog: "Chaque fiche produit generee est versionnee avec : horodatage de generation, donnees source utilisees (hash), modele LLM et version, prompt utilise (versionne), score SEO avant et apres validation, identifiant de l'operateur ayant approuve la publication, et horodatage de publication vers le CMS. Retention de 24 mois avec export CSV automatique.",
      humanInTheLoop: "Les fiches dont le score SEO est inferieur a 70/100 ou dont le contenu est signale comme potentiellement inexact sont routees vers une file de validation humaine. Un redacteur peut editer, approuver ou rejeter chaque fiche via le dashboard. Les fiches des categories reglementees (cosmetique, alimentaire) necessitent systematiquement une validation humaine avant publication.",
      monitoring: "Dashboard Langfuse et Supabase : volume de fiches generees par jour, score SEO moyen par categorie, taux de rejet au premier passage, taux de validation humaine, cout LLM moyen par fiche, temps de traitement par batch, comparaison de performance entre modeles LLM, alertes sur les degradations de qualite et les anomalies de cout.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (upload CSV fournisseur) -> Node Code (parsing et normalisation CSV) -> Node Supabase (sauvegarde produits bruts) -> Node Loop (pour chaque produit) -> Node HTTP Request (API GPT-4.1 - generation fiche) -> Node Code (validation SEO et unicite) -> Node Switch (score SEO >= 70 ?) -> Branch OK: Node Supabase (sauvegarde fiche validee) -> Branch KO: Node HTTP Request (regeneration avec directives) -> Node Supabase (sauvegarde finale) -> Node HTTP Request (export vers CMS Shopify/WooCommerce) -> Node Email (rapport de batch).",
      nodes: ["Webhook (upload CSV)", "Code (parsing CSV)", "Supabase (produits bruts)", "Loop (produits)", "HTTP Request (GPT-4.1 generation)", "Code (validation SEO)", "Switch (score SEO)", "HTTP Request (regeneration)", "Supabase (fiches validees)", "HTTP Request (export CMS)", "Email (rapport batch)"],
      triggerType: "Webhook (upload fichier fournisseur)",
    },
    estimatedTime: "4-8h",
    difficulty: "Facile",
    sectors: ["E-commerce", "Retail", "Distribution"],
    metiers: ["Marketing Digital", "Marketing Strategique"],
    functions: ["Marketing"],
    metaTitle: "Agent IA de Creation de Fiches Produit SEO -- Guide Complet",
    metaDescription:
      "Generez automatiquement des fiches produit SEO-optimisees a partir de donnees fournisseurs. Pipeline de generation en masse, validation qualite et export CMS. Tutoriel complet avec code.",
    createdAt: "2026-02-07",
    updatedAt: "2026-02-07",
  },
  {
    slug: "agent-veille-concurrentielle-automatisee",
    title: "Agent de Veille Concurrentielle Automatisée",
    subtitle: "Orchestrez une surveillance multi-sources en continu de vos concurrents avec alertes intelligentes et rapports stratégiques",
    problem:
      "Les équipes marketing et stratégie n'ont pas les moyens de surveiller en continu l'ensemble des mouvements concurrentiels : lancements produits, changements de prix, campagnes publicitaires, recrutements clés, brevets déposés et partenariats annoncés. La veille manuelle est fragmentée, réactive et ne couvre qu'une fraction des sources pertinentes. Les décideurs reçoivent des rapports obsolètes qui ne permettent pas d'anticiper les mouvements du marché. Les signaux faibles sont systématiquement manqués car noyés dans le bruit informationnel.",
    value:
      "Un agent IA orchestre un réseau de collecteurs automatisés qui scrapent en continu les sites concurrents, flux RSS, réseaux sociaux, bases de brevets, offres d'emploi et communiqués de presse. Un pipeline NLP analyse chaque source, détecte les changements significatifs, les classifie par type (prix, produit, stratégie, RH) et niveau d'impact, puis génère des alertes en temps réel et des rapports de synthèse hebdomadaires avec recommandations stratégiques actionnables.",
    inputs: [
      "Liste des concurrents avec URLs de sites web, pages produits et réseaux sociaux",
      "Flux RSS et newsletters sectorielles",
      "Bases de brevets (INPI, EPO, USPTO)",
      "Sites d'offres d'emploi (LinkedIn, Indeed, Welcome to the Jungle)",
      "Critères de surveillance pondérés par priorité stratégique",
      "Historique de veille et rapports précédents",
    ],
    outputs: [
      "Alertes temps réel sur changements critiques (prix, lancements, partenariats)",
      "Rapport de synthèse hebdomadaire avec scoring d'impact",
      "Tableau comparatif des positionnements prix actualisé",
      "Cartographie des mouvements RH clés (recrutements, départs)",
      "Analyse des tendances brevets et innovation par concurrent",
      "Recommandations stratégiques contextualisées",
    ],
    risks: [
      "Violation des CGU lors du scraping de sites concurrents",
      "Faux positifs sur la détection de changements mineurs interprétés comme stratégiques",
      "Dépendance à des sources web instables (changements de structure HTML)",
      "Biais de confirmation dans l'interprétation LLM des signaux faibles",
      "Surcharge informationnelle si les seuils d'alerte sont mal calibrés",
    ],
    roiIndicatif:
      "Réduction de 75% du temps analyste consacré à la veille manuelle. Détection des mouvements concurrentiels 3x plus rapide. Couverture de sources multipliée par 10.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "Firecrawl", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral Large", category: "LLM", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "SQLite + ChromaDB", category: "Database", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
      { name: "Scrapy", category: "Other", isFree: true },
    ],
    architectureDiagram: `┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Scraper     │  │  RSS/API     │  │  Réseaux     │
│  Web Sites   │  │  Collector   │  │  Sociaux     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────┬───────┴─────────┬───────┘
                 │                 │
          ┌──────▼───────┐  ┌─────▼────────┐
          │  Pipeline    │  │  Vector DB   │
          │  NLP/LLM     │  │  (Historique)│
          └──────┬───────┘  └──────────────┘
                 │
          ┌──────▼───────┐
          │  Alertes +   │
          │  Rapports    │
          └──────────────┘`,
    tutorial: [
      {
        title: "Configuration de l'infrastructure de collecte",
        content:
          "Mettez en place le système de collecte multi-sources. Configurez Firecrawl pour le scraping web structuré, les connecteurs RSS, et les API de réseaux sociaux. Chaque source est normalisée dans un format commun avant analyse.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain anthropic supabase firecrawl-py feedparser tweepy python-dotenv schedule`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `# .env
ANTHROPIC_API_KEY=sk-ant-...
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
FIRECRAWL_API_KEY=fc-...
TWITTER_BEARER_TOKEN=...`,
            filename: ".env",
          },
          {
            language: "python",
            code: `import feedparser
from firecrawl import FirecrawlApp
from dataclasses import dataclass
from datetime import datetime

@dataclass
class SourceItem:
    source_type: str  # web, rss, social, patent
    competitor: str
    url: str
    title: str
    content: str
    collected_at: str
    raw_metadata: dict

class MultiSourceCollector:
    def __init__(self):
        self.firecrawl = FirecrawlApp()
        self.sources_config = {}

    def collect_web_pages(self, competitor: str, urls: list[str]) -> list[SourceItem]:
        """Scrape les pages web des concurrents via Firecrawl."""
        items = []
        for url in urls:
            result = self.firecrawl.scrape_url(url, params={"formats": ["markdown"]})
            items.append(SourceItem(
                source_type="web",
                competitor=competitor,
                url=url,
                title=result.get("metadata", {}).get("title", ""),
                content=result.get("markdown", ""),
                collected_at=datetime.utcnow().isoformat(),
                raw_metadata=result.get("metadata", {}),
            ))
        return items

    def collect_rss_feeds(self, competitor: str, feed_urls: list[str]) -> list[SourceItem]:
        """Collecte les derniers articles via flux RSS."""
        items = []
        for feed_url in feed_urls:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                items.append(SourceItem(
                    source_type="rss",
                    competitor=competitor,
                    url=entry.get("link", ""),
                    title=entry.get("title", ""),
                    content=entry.get("summary", ""),
                    collected_at=datetime.utcnow().isoformat(),
                    raw_metadata={"published": entry.get("published", "")},
                ))
        return items`,
            filename: "collector.py",
          },
        ],
      },
      {
        title: "Détection de changements et analyse NLP",
        content:
          "Implémentez le moteur de détection de changements qui compare les collectes successives et identifie les modifications significatives. Le LLM analyse le contexte de chaque changement pour le classifier et évaluer son impact stratégique.",
        codeSnippets: [
          {
            language: "python",
            code: `from anthropic import Anthropic
from supabase import create_client
import json
import hashlib
import os

client = Anthropic()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

ANALYSIS_PROMPT = """Tu es un analyste en intelligence concurrentielle pour le marché français.

Analyse le changement détecté ci-dessous et produis un rapport structuré.

## Concurrent: {competitor}
## Source: {source_type}
## Contenu précédent:
{previous_content}

## Contenu actuel:
{current_content}

## Consignes:
1. Identifie la nature du changement (prix, produit, stratégie, RH, communication, partenariat)
2. Évalue l'impact stratégique de 1 (mineur) à 5 (critique)
3. Explique les implications pour notre entreprise
4. Suggère des actions concrètes à envisager

Retourne un JSON avec: change_type, impact_score, summary, implications, recommended_actions"""

def detect_changes(item: dict) -> dict | None:
    """Compare avec la version précédente et détecte les changements."""
    content_hash = hashlib.sha256(item["content"].encode()).hexdigest()
    previous = supabase.table("veille_snapshots").select("*").eq(
        "url", item["url"]
    ).order("collected_at", desc=True).limit(1).execute()

    if previous.data and previous.data[0]["content_hash"] == content_hash:
        return None  # Pas de changement

    # Sauvegarder le nouveau snapshot
    supabase.table("veille_snapshots").insert({
        "url": item["url"],
        "competitor": item["competitor"],
        "content": item["content"],
        "content_hash": content_hash,
        "collected_at": item["collected_at"],
    }).execute()

    if not previous.data:
        return None  # Premier snapshot, pas de comparaison possible

    # Analyser le changement avec le LLM
    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": ANALYSIS_PROMPT.format(
                competitor=item["competitor"],
                source_type=item["source_type"],
                previous_content=previous.data[0]["content"][:3000],
                current_content=item["content"][:3000],
            ),
        }],
    )
    return json.loads(response.content[0].text)`,
            filename: "change_detector.py",
          },
        ],
      },
      {
        title: "Système d'alertes et notifications",
        content:
          "Configurez le système d'alertes intelligentes qui notifie les bonnes personnes selon le type et l'impact du changement détecté. Les alertes critiques (impact >= 4) déclenchent une notification immédiate. Les changements mineurs sont agrégés dans le rapport hebdomadaire.",
        codeSnippets: [
          {
            language: "python",
            code: `import requests
from datetime import datetime

class AlertManager:
    def __init__(self, slack_webhook_url: str, email_config: dict):
        self.slack_webhook = slack_webhook_url
        self.email_config = email_config
        self.alert_rules = {
            "prix": {"threshold": 3, "channels": ["slack", "email"], "mentions": ["@pricing-team"]},
            "produit": {"threshold": 3, "channels": ["slack"], "mentions": ["@product-team"]},
            "stratégie": {"threshold": 4, "channels": ["slack", "email"], "mentions": ["@direction"]},
            "RH": {"threshold": 2, "channels": ["slack"], "mentions": ["@rh-veille"]},
            "partenariat": {"threshold": 3, "channels": ["slack", "email"], "mentions": ["@bizdev"]},
        }

    def process_alert(self, change: dict, competitor: str):
        """Traite un changement détecté et envoie les alertes appropriées."""
        change_type = change["change_type"]
        impact = change["impact_score"]
        rule = self.alert_rules.get(change_type, {"threshold": 4, "channels": ["slack"], "mentions": []})

        if impact < rule["threshold"]:
            # Stocker pour le rapport hebdomadaire
            supabase.table("veille_weekly_buffer").insert({
                "competitor": competitor,
                "change": change,
                "created_at": datetime.utcnow().isoformat(),
            }).execute()
            return

        # Alerte immédiate
        message = self._format_alert(change, competitor)
        if "slack" in rule["channels"]:
            self._send_slack(message, rule["mentions"])
        if "email" in rule["channels"]:
            self._send_email(message, change_type)

    def _format_alert(self, change: dict, competitor: str) -> str:
        impact_emoji = "🔴" if change["impact_score"] >= 4 else "🟡"
        return (
            f"{impact_emoji} *Alerte Veille Concurrentielle*\\n"
            f"*Concurrent:* {competitor}\\n"
            f"*Type:* {change['change_type']} | *Impact:* {change['impact_score']}/5\\n"
            f"*Résumé:* {change['summary']}\\n"
            f"*Actions recommandées:*\\n"
            + "\\n".join(f"• {a}" for a in change["recommended_actions"])
        )

    def _send_slack(self, message: str, mentions: list[str]):
        mention_str = " ".join(mentions)
        requests.post(self.slack_webhook, json={
            "text": f"{mention_str}\\n{message}",
        })`,
            filename: "alert_manager.py",
          },
        ],
      },
      {
        title: "Génération de rapports stratégiques hebdomadaires",
        content:
          "Chaque semaine, l'agent génère un rapport de synthèse consolidant tous les changements détectés, les tendances identifiées et les recommandations stratégiques. Le rapport est structuré par concurrent et par thématique, avec un scoring d'importance.",
        codeSnippets: [
          {
            language: "python",
            code: `WEEKLY_REPORT_PROMPT = """Tu es un directeur de veille stratégique dans un cabinet de conseil français.

Génère un rapport de veille concurrentielle hebdomadaire à partir des changements détectés.

## Changements de la semaine:
{changes_json}

## Consignes:
1. Structure le rapport par concurrent puis par thématique
2. Identifie les 3 tendances principales de la semaine
3. Mets en avant les signaux faibles qui méritent une surveillance renforcée
4. Propose 5 recommandations stratégiques actionnables et priorisées
5. Attribue un score de menace global (1-10) pour chaque concurrent
6. Rédige en français professionnel, ton analytique et factuel

Retourne le rapport en Markdown structuré."""

def generate_weekly_report() -> str:
    """Génère le rapport hebdomadaire de veille concurrentielle."""
    from datetime import datetime, timedelta

    # Récupérer les changements de la semaine
    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
    changes = supabase.table("veille_weekly_buffer").select("*").gte(
        "created_at", week_ago
    ).execute()

    # Récupérer aussi les alertes immédiates de la semaine
    alerts = supabase.table("veille_alerts").select("*").gte(
        "created_at", week_ago
    ).execute()

    all_changes = [c["change"] for c in changes.data] + [a["change"] for a in alerts.data]

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": WEEKLY_REPORT_PROMPT.format(
                changes_json=json.dumps(all_changes, ensure_ascii=False, indent=2)
            ),
        }],
    )

    report_md = response.content[0].text

    # Sauvegarder le rapport
    supabase.table("veille_reports").insert({
        "report_date": datetime.utcnow().date().isoformat(),
        "content_md": report_md,
        "changes_count": len(all_changes),
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    return report_md`,
            filename: "weekly_report.py",
          },
        ],
      },
      {
        title: "Orchestration et scheduling",
        content:
          "Mettez en place le scheduler qui orchestre les collectes à intervalles réguliers. Les pages web sont scrapées quotidiennement, les flux RSS toutes les 4 heures, et le rapport hebdomadaire est généré chaque lundi matin. Déployez l'ensemble sur Railway ou Vercel avec des cron jobs.",
        codeSnippets: [
          {
            language: "python",
            code: `import schedule
import time
from collector import MultiSourceCollector
from change_detector import detect_changes
from alert_manager import AlertManager
from weekly_report import generate_weekly_report

collector = MultiSourceCollector()
alert_mgr = AlertManager(
    slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
    email_config={"smtp_host": "smtp.gmail.com", "port": 587},
)

COMPETITORS = {
    "Concurrent A": {
        "web_urls": ["https://concurrent-a.fr/produits", "https://concurrent-a.fr/tarifs"],
        "rss_feeds": ["https://concurrent-a.fr/blog/feed"],
    },
    "Concurrent B": {
        "web_urls": ["https://concurrent-b.fr/offres"],
        "rss_feeds": ["https://concurrent-b.fr/actualites/rss"],
    },
}

def run_collection_cycle():
    """Exécute un cycle complet de collecte et analyse."""
    for competitor, sources in COMPETITORS.items():
        # Collecter les pages web
        items = collector.collect_web_pages(competitor, sources["web_urls"])
        items += collector.collect_rss_feeds(competitor, sources["rss_feeds"])

        # Détecter et analyser les changements
        for item in items:
            change = detect_changes(item.__dict__)
            if change:
                alert_mgr.process_alert(change, competitor)
                print(f"[CHANGE] {competitor}: {change['summary']}")

# Programmer les collectes
schedule.every(4).hours.do(run_collection_cycle)
schedule.every().monday.at("08:00").do(generate_weekly_report)

if __name__ == "__main__":
    print("Démarrage de l'agent de veille concurrentielle automatisée...")
    run_collection_cycle()  # Première exécution immédiate
    while True:
        schedule.run_pending()
        time.sleep(60)`,
            filename: "main.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données collectées peuvent contenir des informations personnelles (noms de dirigeants, profils LinkedIn). Appliquer un filtre de pseudonymisation avant stockage long terme. Les données brutes de scraping sont purgées après extraction des insights. Conformité RGPD assurée car seules des données publiquement accessibles sont collectées.",
      auditLog: "Chaque collecte est loguée avec : timestamp, source URL, concurrent, hash du contenu, changements détectés, score d'impact attribué, alertes déclenchées. Rétention 12 mois avec archivage automatique des rapports hebdomadaires.",
      humanInTheLoop: "Les changements classifiés avec un impact >= 4 nécessitent une validation humaine avant diffusion au comité de direction. Les analystes peuvent corriger la classification et le scoring via un dashboard de modération. Les recommandations stratégiques du rapport hebdomadaire sont relues par le responsable veille avant envoi.",
      monitoring: "Dashboard Langfuse : nombre de sources collectées par cycle, taux de changements détectés, distribution des scores d'impact, temps de traitement par source, coût LLM par cycle de collecte. Alertes si le taux de succès du scraping descend sous 90% ou si aucun changement n'est détecté pendant 72h.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Cron Trigger (toutes les 4h) -> Node HTTP Request (Firecrawl scraping) -> Node RSS Feed Reader -> Node Code (normalisation multi-sources) -> Node Supabase (comparaison avec snapshots précédents) -> Node IF (changement détecté ?) -> Node HTTP Request (API Claude - analyse d'impact) -> Node Switch (impact >= 4 ?) -> Branch haute priorité : Node Slack (alerte immédiate) + Node Email -> Branch basse priorité : Node Supabase (buffer hebdomadaire) -> Cron hebdomadaire : Node HTTP Request (génération rapport) -> Node Email (envoi rapport).",
      nodes: ["Cron Trigger (4h)", "HTTP Request (Firecrawl)", "RSS Feed Reader", "Code (normalisation)", "Supabase (snapshots)", "IF (changement détecté)", "HTTP Request (Claude API)", "Switch (impact score)", "Slack (alerte)", "Email (notification)", "Supabase (buffer)", "Cron Monday (rapport)"],
      triggerType: "Cron (toutes les 4 heures + hebdomadaire lundi 8h)",
    },
    estimatedTime: "6-10h",
    difficulty: "Moyen",
    sectors: ["Services", "E-commerce", "Industrie", "Tech"],
    metiers: ["Marketing Strategique", "Direction Générale"],
    functions: ["Marketing", "Stratégie"],
    metaTitle: "Agent IA de Veille Concurrentielle Automatisée — Guide Complet",
    metaDescription:
      "Déployez un agent IA de surveillance concurrentielle multi-sources avec détection de changements, alertes intelligentes et rapports stratégiques hebdomadaires. Tutoriel complet.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-notes-frais-ocr-conformite",
    title: "Agent de Gestion des Notes de Frais avec OCR et Conformité",
    subtitle: "Automatisez l'extraction OCR, la vérification de conformité et l'imputation comptable des notes de frais en temps réel",
    problem:
      "Les services comptabilité des ETI et grands groupes français traitent des milliers de notes de frais mensuellement. La saisie manuelle des justificatifs génère en moyenne 15% d'erreurs de montant ou de catégorisation. Les contrôles de conformité à la politique de dépenses sont inconsistants : certains dépassements passent inaperçus tandis que des dépenses légitimes sont bloquées. Les délais de remboursement dépassant 3 semaines impactent la satisfaction des collaborateurs. La réconciliation avec les relevés de cartes bancaires corporate mobilise 2 ETP à temps plein dans les structures de plus de 500 salariés.",
    value:
      "Un agent IA équipé d'OCR avancé extrait automatiquement les données de chaque justificatif (montant, date, fournisseur, TVA), les croise avec le relevé de carte corporate, vérifie la conformité en temps réel avec la politique de dépenses paramétrable, détecte les anomalies et doublons, impute comptablement chaque ligne, et soumet le dossier complet pour validation managériale. Le délai de traitement chute de 21 jours à 48 heures.",
    inputs: [
      "Photos ou scans de justificatifs (tickets, factures, reçus de restaurant)",
      "Relevés de cartes bancaires corporate (CSV ou API bancaire)",
      "Politique de dépenses paramétrable (plafonds par catégorie, par grade, par zone)",
      "Plan comptable avec axes analytiques (centre de coût, projet, activité)",
      "Données collaborateur (service, grade, lieu d'affectation, supérieur hiérarchique)",
      "Historique des notes de frais précédentes (détection patterns)",
    ],
    outputs: [
      "Extraction structurée de chaque justificatif (montant HT/TTC, TVA, date, fournisseur)",
      "Rapport de conformité avec alertes sur les dépassements",
      "Imputation comptable automatique (compte, centre de coût, axe analytique)",
      "Score de risque fraude par note de frais (0-100)",
      "Dossier complet réconcilié prêt pour validation managériale",
      "Tableau de bord consolidé des dépenses par service/projet",
    ],
    risks: [
      "Erreurs OCR sur les justificatifs de mauvaise qualité (tickets thermiques effacés)",
      "Faux positifs de fraude créant de la friction avec les collaborateurs",
      "Mauvaise imputation comptable impactant les clôtures mensuelles",
      "Non-conformité fiscale si la TVA récupérable est mal calculée",
      "Dépendance au LLM pour des décisions financières auditables",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de traitement comptable. Taux d'erreur de saisie divisé par 10. Délai de remboursement réduit de 21 jours à 48h. Détection de 95% des anomalies et doublons.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "Azure Document Intelligence", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "PostgreSQL", category: "Database", isFree: true },
      { name: "Tesseract OCR", category: "Other", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Upload      │────▶│  OCR Engine  │────▶│  Agent LLM   │
│  Justificatif│     │  (Extraction)│     │  (Conformité)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
┌──────────────┐     ┌──────────────┐     ┌──────▼───────┐
│  Relevé CB   │────▶│  Réconcil.   │────▶│  Imputation  │
│  Corporate   │     │  Automatique │     │  Comptable   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           ┌──────▼───────┐
                                           │  Validation  │
                                           │  Manager     │
                                           └──────────────┘`,
    tutorial: [
      {
        title: "Configuration OCR et extraction de justificatifs",
        content:
          "Mettez en place le pipeline OCR qui transforme les photos de justificatifs en données structurées. Azure Document Intelligence offre une extraction de haute qualité pour les tickets et factures français, avec reconnaissance automatique des champs clés (montant, TVA, date, fournisseur).",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain openai supabase azure-ai-documentintelligence python-dotenv pillow`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from dataclasses import dataclass
from decimal import Decimal
import os

@dataclass
class ExtractedReceipt:
    merchant_name: str
    date: str
    total_ttc: Decimal
    total_ht: Decimal | None
    tva_amount: Decimal | None
    tva_rate: str | None
    currency: str
    items: list[dict]
    confidence_score: float
    raw_text: str

class ReceiptOCR:
    def __init__(self):
        self.client = DocumentIntelligenceClient(
            endpoint=os.getenv("AZURE_DOC_ENDPOINT"),
            credential=AzureKeyCredential(os.getenv("AZURE_DOC_KEY")),
        )

    def extract_receipt(self, image_bytes: bytes) -> ExtractedReceipt:
        """Extrait les données structurées d'un justificatif."""
        poller = self.client.begin_analyze_document(
            "prebuilt-receipt",
            body=image_bytes,
            content_type="application/octet-stream",
        )
        result = poller.result()
        receipt = result.documents[0] if result.documents else None

        if not receipt:
            raise ValueError("Aucun justificatif détecté dans l'image")

        fields = receipt.fields
        return ExtractedReceipt(
            merchant_name=self._get_field(fields, "MerchantName", "Inconnu"),
            date=self._get_field(fields, "TransactionDate", ""),
            total_ttc=Decimal(str(self._get_field(fields, "Total", 0))),
            total_ht=self._calc_ht(fields),
            tva_amount=self._get_tva(fields),
            tva_rate=self._detect_tva_rate(fields),
            currency=self._get_field(fields, "Currency", "EUR"),
            items=self._extract_items(fields),
            confidence_score=receipt.confidence,
            raw_text=result.content or "",
        )

    def _get_field(self, fields, name, default):
        f = fields.get(name)
        return f.value if f else default

    def _calc_ht(self, fields):
        total = fields.get("Total")
        tax = fields.get("TotalTax")
        if total and tax:
            return Decimal(str(total.value)) - Decimal(str(tax.value))
        return None

    def _get_tva(self, fields):
        tax = fields.get("TotalTax")
        return Decimal(str(tax.value)) if tax else None

    def _detect_tva_rate(self, fields):
        ht = self._calc_ht(fields)
        tva = self._get_tva(fields)
        if ht and tva and ht > 0:
            rate = (tva / ht * 100).quantize(Decimal("0.1"))
            if rate >= 19 and rate <= 21:
                return "20%"
            elif rate >= 9 and rate <= 11:
                return "10%"
            elif rate >= 4.5 and rate <= 6:
                return "5.5%"
        return None

    def _extract_items(self, fields):
        items_field = fields.get("Items")
        if not items_field:
            return []
        return [{"description": i.value.get("Description", {}).get("value", ""),
                 "amount": str(i.value.get("TotalPrice", {}).get("value", ""))}
                for i in items_field.value]`,
            filename: "receipt_ocr.py",
          },
        ],
      },
      {
        title: "Moteur de conformité et politique de dépenses",
        content:
          "Implémentez le moteur de règles qui vérifie chaque note de frais contre la politique de dépenses de l'entreprise. Les règles sont paramétrables par catégorie, grade du collaborateur et zone géographique. Le LLM intervient pour les cas ambigus que les règles déterministes ne couvrent pas.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
from dataclasses import dataclass
from decimal import Decimal
import json

client = OpenAI()

@dataclass
class ComplianceResult:
    is_compliant: bool
    violations: list[str]
    warnings: list[str]
    fraud_score: int  # 0-100
    auto_approved: bool
    needs_manager_review: bool

# Politique de dépenses paramétrable
EXPENSE_POLICY = {
    "restaurant": {
        "plafond_par_personne": {"standard": 25, "manager": 40, "directeur": 60},
        "requires_guest_list": True,
        "max_alcohol_pct": 20,
    },
    "transport": {
        "taxi_max_km": 50,
        "train_class": {"standard": 2, "manager": 1, "directeur": 1},
        "avion_requires_approval": True,
    },
    "hotel": {
        "plafond_nuit": {"paris": 180, "province": 130, "etranger": 200},
        "max_consecutive_nights": 5,
    },
    "fournitures": {"plafond_mensuel": 100},
}

def check_compliance(receipt: dict, employee: dict, category: str) -> ComplianceResult:
    """Vérifie la conformité d'une dépense avec la politique."""
    violations = []
    warnings = []
    fraud_score = 0
    grade = employee.get("grade", "standard")

    policy = EXPENSE_POLICY.get(category, {})

    # Vérification des plafonds
    if category == "restaurant":
        plafond = policy["plafond_par_personne"].get(grade, 25)
        nb_convives = receipt.get("nb_convives", 1)
        max_amount = Decimal(str(plafond * nb_convives))
        if receipt["total_ttc"] > max_amount:
            violations.append(
                f"Dépassement plafond restaurant: {receipt['total_ttc']}EUR > "
                f"{max_amount}EUR ({nb_convives} convive(s) x {plafond}EUR)"
            )
        if policy["requires_guest_list"] and not receipt.get("guest_list"):
            warnings.append("Liste des convives manquante pour le repas d'affaires")

    elif category == "hotel":
        zone = employee.get("zone", "province")
        plafond = Decimal(str(policy["plafond_nuit"].get(zone, 130)))
        if receipt["total_ttc"] > plafond:
            violations.append(
                f"Dépassement plafond hôtel ({zone}): {receipt['total_ttc']}EUR > {plafond}EUR/nuit"
            )

    # Détection doublons et anomalies
    fraud_indicators = detect_fraud_indicators(receipt, employee)
    fraud_score = fraud_indicators["score"]
    if fraud_score > 60:
        warnings.append(f"Score de risque fraude élevé: {fraud_score}/100")

    is_compliant = len(violations) == 0
    auto_approved = is_compliant and fraud_score < 30
    needs_review = not is_compliant or fraud_score >= 40

    return ComplianceResult(
        is_compliant=is_compliant,
        violations=violations,
        warnings=warnings,
        fraud_score=fraud_score,
        auto_approved=auto_approved,
        needs_manager_review=needs_review,
    )

def detect_fraud_indicators(receipt: dict, employee: dict) -> dict:
    """Détecte les indicateurs de fraude potentielle via LLM."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "system",
            "content": "Tu es un auditeur financier spécialisé en détection de fraude sur notes de frais.",
        }, {
            "role": "user",
            "content": f"""Analyse cette dépense pour détecter des indicateurs de fraude:
- Montant: {receipt['total_ttc']}EUR
- Fournisseur: {receipt.get('merchant_name', 'N/A')}
- Date: {receipt.get('date', 'N/A')} (jour: {receipt.get('day_of_week', 'N/A')})
- Catégorie: {receipt.get('category', 'N/A')}
- Collaborateur grade: {employee.get('grade', 'N/A')}
- Montant arrondi: {'oui' if float(receipt['total_ttc']) % 1 == 0 else 'non'}

Retourne un JSON: score (0-100), indicators (liste de strings), explanation (string)""",
        }],
    )
    return json.loads(response.choices[0].message.content)`,
            filename: "compliance_engine.py",
          },
        ],
      },
      {
        title: "Réconciliation bancaire automatique",
        content:
          "Le module de réconciliation croise automatiquement les justificatifs soumis avec les transactions du relevé de carte corporate. Il détecte les dépenses non justifiées et les justificatifs orphelins, en utilisant un matching intelligent par montant, date et fournisseur.",
        codeSnippets: [
          {
            language: "python",
            code: `import csv
from datetime import datetime, timedelta
from decimal import Decimal
from dataclasses import dataclass

@dataclass
class ReconciliationResult:
    matched: list[dict]       # Justificatif <-> transaction appariés
    unmatched_receipts: list[dict]  # Justificatifs sans transaction
    unmatched_transactions: list[dict]  # Transactions sans justificatif
    match_confidence: dict    # ID -> score de confiance

class BankReconciler:
    def __init__(self, tolerance_amount: Decimal = Decimal("0.50"),
                 tolerance_days: int = 3):
        self.tolerance_amount = tolerance_amount
        self.tolerance_days = tolerance_days

    def load_bank_statement(self, csv_path: str) -> list[dict]:
        """Charge un relevé bancaire au format CSV."""
        transactions = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                transactions.append({
                    "date": datetime.strptime(row["Date"], "%d/%m/%Y"),
                    "amount": Decimal(row["Montant"].replace(",", ".")),
                    "merchant": row.get("Libellé", ""),
                    "reference": row.get("Référence", ""),
                })
        return transactions

    def reconcile(self, receipts: list[dict],
                  transactions: list[dict]) -> ReconciliationResult:
        """Réconcilie les justificatifs avec les transactions bancaires."""
        matched = []
        used_tx_ids = set()
        match_confidence = {}

        for receipt in receipts:
            best_match = None
            best_score = 0

            for i, tx in enumerate(transactions):
                if i in used_tx_ids:
                    continue
                score = self._compute_match_score(receipt, tx)
                if score > best_score:
                    best_score = score
                    best_match = i

            if best_match is not None and best_score >= 0.6:
                matched.append({
                    "receipt": receipt,
                    "transaction": transactions[best_match],
                    "confidence": best_score,
                })
                used_tx_ids.add(best_match)
                match_confidence[receipt.get("id", "")] = best_score

        unmatched_receipts = [r for r in receipts
                              if r.get("id") not in match_confidence]
        unmatched_transactions = [t for i, t in enumerate(transactions)
                                   if i not in used_tx_ids]

        return ReconciliationResult(
            matched=matched,
            unmatched_receipts=unmatched_receipts,
            unmatched_transactions=unmatched_transactions,
            match_confidence=match_confidence,
        )

    def _compute_match_score(self, receipt: dict, tx: dict) -> float:
        score = 0.0
        # Matching montant (40% du score)
        r_amount = Decimal(str(receipt.get("total_ttc", 0)))
        t_amount = abs(tx["amount"])
        if abs(r_amount - t_amount) <= self.tolerance_amount:
            score += 0.4
        # Matching date (30% du score)
        r_date = datetime.strptime(receipt["date"], "%Y-%m-%d") if isinstance(receipt["date"], str) else receipt["date"]
        if abs((r_date - tx["date"]).days) <= self.tolerance_days:
            score += 0.3
        # Matching fournisseur (30% du score)
        r_merchant = receipt.get("merchant_name", "").lower()
        t_merchant = tx.get("merchant", "").lower()
        if r_merchant and t_merchant and (r_merchant in t_merchant or t_merchant in r_merchant):
            score += 0.3
        return score`,
            filename: "bank_reconciler.py",
          },
        ],
      },
      {
        title: "Imputation comptable automatique",
        content:
          "L'agent classifie chaque dépense selon le plan comptable de l'entreprise et attribue les axes analytiques (centre de coût, projet, activité). Il utilise l'historique des imputations précédentes et les règles métier pour proposer une imputation fiable.",
        codeSnippets: [
          {
            language: "python",
            code: `ACCOUNTING_PROMPT = """Tu es un comptable expert en plan comptable français (PCG).

Attribue l'imputation comptable pour la dépense suivante:
- Catégorie: {category}
- Fournisseur: {merchant}
- Montant HT: {amount_ht} EUR
- TVA: {tva_amount} EUR (taux: {tva_rate})
- Collaborateur service: {department}
- Projet: {project}

## Plan comptable disponible:
625100 - Déplacements, missions et réceptions
625600 - Missions (transport)
625700 - Réceptions
606100 - Fournitures non stockables
611000 - Sous-traitance générale
613200 - Locations mobilières
616000 - Assurances
618100 - Documentation générale
623400 - Cadeaux à la clientèle
635100 - Impôts directs

## Comptes de TVA:
445660 - TVA déductible sur ABS (20%)
445662 - TVA déductible sur ABS (10%)
445664 - TVA déductible sur ABS (5.5%)

## Règles:
1. Retourne le compte de charge principal
2. Retourne le compte de TVA si la TVA est récupérable
3. Indique le centre de coût basé sur le service
4. Justifie brièvement ton choix

Retourne un JSON: compte_charge, compte_tva, centre_cout, axe_projet, justification"""

def compute_accounting_entry(receipt: dict, employee: dict) -> dict:
    """Calcule l'imputation comptable automatique."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": ACCOUNTING_PROMPT.format(
                category=receipt.get("category", ""),
                merchant=receipt.get("merchant_name", ""),
                amount_ht=receipt.get("total_ht", receipt.get("total_ttc", 0)),
                tva_amount=receipt.get("tva_amount", 0),
                tva_rate=receipt.get("tva_rate", "N/A"),
                department=employee.get("department", ""),
                project=employee.get("current_project", "N/A"),
            ),
        }],
    )
    return json.loads(response.choices[0].message.content)`,
            filename: "accounting_engine.py",
          },
        ],
      },
      {
        title: "API REST et workflow de validation",
        content:
          "Exposez l'ensemble du pipeline via une API REST. Le collaborateur soumet ses justificatifs, l'agent traite tout automatiquement, et le manager reçoit un dossier complet prêt à valider en un clic. Déployez sur Vercel ou Railway.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="Agent Notes de Frais")

@app.post("/api/expense-reports/submit")
async def submit_expense(
    files: list[UploadFile] = File(...),
    employee_id: str = "",
):
    """Soumet une note de frais avec justificatifs."""
    ocr = ReceiptOCR()
    results = []

    # 1. Extraction OCR de chaque justificatif
    for file in files:
        image_bytes = await file.read()
        receipt = ocr.extract_receipt(image_bytes)
        receipt_dict = receipt.__dict__

        # 2. Récupérer les données collaborateur
        employee = supabase.table("employees").select("*").eq(
            "id", employee_id
        ).single().execute().data

        # 3. Classification automatique de la catégorie
        category = classify_expense_category(receipt_dict)
        receipt_dict["category"] = category

        # 4. Vérification de conformité
        compliance = check_compliance(receipt_dict, employee, category)

        # 5. Imputation comptable
        accounting = compute_accounting_entry(receipt_dict, employee)

        # 6. Sauvegarder
        entry = {
            "employee_id": employee_id,
            "receipt_data": receipt_dict,
            "category": category,
            "compliance": compliance.__dict__,
            "accounting": accounting,
            "status": "auto_approved" if compliance.auto_approved else "pending_review",
        }
        saved = supabase.table("expense_entries").insert(entry).execute()
        results.append(saved.data[0])

    # Notifier le manager si validation requise
    pending = [r for r in results if r["status"] == "pending_review"]
    if pending:
        notify_manager(employee, pending)

    return {"entries": results, "auto_approved": len(results) - len(pending), "pending_review": len(pending)}`,
            filename: "api.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les justificatifs contiennent des données personnelles (noms, numéros de carte, lieux fréquentés). Les images brutes sont chiffrées AES-256 au repos dans Supabase Storage. Les numéros de carte sont masqués après extraction OCR (seuls les 4 derniers chiffres sont conservés). Conformité RGPD : droit à l'effacement des justificatifs après le délai légal de conservation comptable (10 ans).",
      auditLog: "Chaque note de frais est tracée intégralement : horodatage de soumission, résultat OCR brut, score de confiance extraction, catégorie attribuée, résultat du contrôle de conformité, score de fraude, imputation comptable proposée, identifiant du valideur, horodatage de validation/rejet, motif de rejet le cas échéant. Piste d'audit complète exportable pour les commissaires aux comptes.",
      humanInTheLoop: "Les notes de frais avec un score de fraude supérieur à 40 ou un dépassement de plafond sont systématiquement routées vers le manager pour validation. Les dépenses dépassant 500 EUR nécessitent une double validation (manager + contrôleur de gestion). Les imputations comptables avec un score de confiance inférieur à 0.8 sont vérifiées par un comptable.",
      monitoring: "Dashboard Langfuse et Supabase : volume de notes traitées par jour, taux d'extraction OCR réussi, taux de conformité automatique, délai moyen de remboursement, distribution des scores de fraude, taux de correction d'imputation comptable, coût LLM moyen par note de frais. Alertes si le taux d'erreur OCR dépasse 5% ou si le délai moyen de validation dépasse 72h.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (soumission justificatif) -> Node HTTP Request (Azure Document Intelligence OCR) -> Node Code (structuration données extraites) -> Node Supabase (récupération profil employé + politique) -> Node HTTP Request (GPT-4.1 classification + conformité) -> Node Switch (auto-approuvé ?) -> Branch OK : Node Supabase (sauvegarde + statut approuvé) -> Branch KO : Node Slack (notification manager) -> Node Wait (approbation manager) -> Node Supabase (mise à jour statut) -> Node HTTP Request (imputation comptable) -> Node HTTP Request (export ERP/SAP).",
      nodes: ["Webhook (soumission)", "HTTP Request (OCR Azure)", "Code (structuration)", "Supabase (profil employé)", "HTTP Request (GPT-4.1 conformité)", "Switch (auto-approuvé)", "Supabase (sauvegarde)", "Slack (notification manager)", "Wait (approbation)", "HTTP Request (imputation)", "HTTP Request (export ERP)"],
      triggerType: "Webhook (soumission de note de frais via app mobile ou web)",
    },
    estimatedTime: "6-8h",
    difficulty: "Moyen",
    sectors: ["Services", "Industrie", "Banque", "Conseil"],
    metiers: ["Comptabilité", "Finance", "Contrôle de Gestion"],
    functions: ["Finance", "Comptabilité"],
    metaTitle: "Agent IA de Gestion des Notes de Frais avec OCR — Guide Complet",
    metaDescription:
      "Automatisez le traitement des notes de frais avec OCR intelligent, contrôle de conformité en temps réel et imputation comptable automatique. Tutoriel pas-à-pas avec code Python.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-planification-reunions-intelligente",
    title: "Agent de Planification de Réunions Intelligente",
    subtitle: "Optimisez automatiquement la planification des réunions en tenant compte des disponibilités, des fuseaux horaires et de la charge cognitive des participants",
    problem:
      "La planification de réunions dans les organisations de plus de 100 collaborateurs est un cauchemar logistique. Les assistants de direction passent en moyenne 5 heures par semaine à coordonner les agendas. Les conflits de créneaux génèrent des chaînes d'emails interminables. Les réunions s'accumulent sans tenir compte de la charge cognitive des participants : pas de temps de respiration entre deux meetings, réunions placées sur les plages de travail profond, fuseaux horaires ignorés pour les équipes internationales. Résultat : 67% des cadres estiment que les réunions les empêchent de travailler efficacement.",
    value:
      "Un agent IA analyse les agendas de tous les participants, identifie les créneaux optimaux en tenant compte des préférences individuelles (travail profond le matin, pas de réunion le vendredi après-midi), des fuseaux horaires, de la charge de réunions quotidienne, et de la priorité du sujet. Il propose automatiquement les 3 meilleurs créneaux, gère les relances, réserve les salles et génère un ordre du jour structuré.",
    inputs: [
      "Agendas Google Calendar ou Microsoft Outlook de tous les participants",
      "Préférences individuelles de planning (plages protégées, jours sans réunion)",
      "Fuseaux horaires des participants distants",
      "Priorité et durée estimée de la réunion",
      "Sujet et objectifs de la réunion",
      "Disponibilité des salles de réunion (intégration room booking)",
    ],
    outputs: [
      "Top 3 des créneaux optimaux avec score de pertinence",
      "Invitation calendrier envoyée automatiquement avec salle réservée",
      "Ordre du jour structuré généré à partir du sujet",
      "Rappels intelligents avec documents préparatoires",
      "Rapport hebdomadaire de charge réunion par équipe",
      "Suggestions de réunions à annuler ou fusionner",
    ],
    risks: [
      "Accès en lecture aux agendas personnels soulevant des questions de vie privée",
      "Créneaux imposés sans consentement réel des participants",
      "Sur-optimisation rendant les agendas trop rigides",
      "Erreurs de fuseau horaire pour les équipes multi-sites",
      "Dépendance aux API de calendrier tierces (rate limiting, pannes)",
    ],
    roiIndicatif:
      "Réduction de 80% du temps de coordination des réunions. Diminution de 30% du nombre de réunions grâce aux suggestions de fusion. Amélioration de 25% du score de satisfaction planning des collaborateurs.",
    recommendedStack: [
      { name: "OpenAI GPT-4.1", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral Large", category: "LLM", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "PostgreSQL", category: "Database", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
    ],
    architectureDiagram: `┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Google      │  │  Outlook     │  │  Room        │
│  Calendar    │  │  Calendar    │  │  Booking     │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────┬───────┴─────────┬───────┘
                 │                 │
          ┌──────▼───────┐  ┌─────▼────────┐
          │  Agent LLM   │  │  Préférences │
          │  (Optimizer)  │  │  DB          │
          └──────┬───────┘  └──────────────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
┌──────▼──┐ ┌───▼────┐ ┌──▼───────┐
│ Invite  │ │ Salle  │ │ Ordre du │
│ Calendar│ │ Réserv.│ │ jour     │
└─────────┘ └────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Connexion aux API de calendrier",
        content:
          "Configurez les connecteurs Google Calendar et Microsoft Graph pour accéder en lecture/écriture aux agendas des participants. Utilisez OAuth 2.0 pour l'authentification sécurisée. Le service account Google permet un accès délégué à l'échelle de l'organisation.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain openai supabase google-auth google-api-python-client msal python-dotenv pytz`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `from google.oauth2 import service_account
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly",
           "https://www.googleapis.com/auth/calendar.events"]

class GoogleCalendarConnector:
    def __init__(self, service_account_file: str):
        self.credentials = service_account.Credentials.from_service_account_file(
            service_account_file, scopes=SCOPES
        )

    def get_busy_slots(self, email: str, start: datetime,
                       end: datetime) -> list[dict]:
        """Récupère les créneaux occupés d'un utilisateur."""
        service = build("calendar", "v3",
                       credentials=self.credentials.with_subject(email))
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "timeZone": "Europe/Paris",
            "items": [{"id": email}],
        }
        result = service.freebusy().query(body=body).execute()
        busy = result["calendars"][email]["busy"]
        return [{"start": b["start"], "end": b["end"]} for b in busy]

    def get_events_detail(self, email: str, start: datetime,
                          end: datetime) -> list[dict]:
        """Récupère les détails des événements (pour analyse de charge)."""
        service = build("calendar", "v3",
                       credentials=self.credentials.with_subject(email))
        events = service.events().list(
            calendarId=email,
            timeMin=start.isoformat(),
            timeMax=end.isoformat(),
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        return [{
            "summary": e.get("summary", "Sans titre"),
            "start": e["start"].get("dateTime", e["start"].get("date")),
            "end": e["end"].get("dateTime", e["end"].get("date")),
            "attendees_count": len(e.get("attendees", [])),
            "is_recurring": "recurringEventId" in e,
        } for e in events.get("items", [])]

    def create_event(self, organizer_email: str, event: dict) -> str:
        """Crée un événement dans le calendrier de l'organisateur."""
        service = build("calendar", "v3",
                       credentials=self.credentials.with_subject(organizer_email))
        created = service.events().insert(
            calendarId=organizer_email,
            body=event,
            sendUpdates="all",
        ).execute()
        return created["htmlLink"]`,
            filename: "calendar_connector.py",
          },
        ],
      },
      {
        title: "Moteur d'optimisation de créneaux",
        content:
          "Implémentez l'algorithme de scoring qui évalue chaque créneau possible en tenant compte de multiples critères pondérés : disponibilité de tous les participants, respect des préférences individuelles, charge cognitive quotidienne, proximité avec d'autres réunions, et fuseaux horaires. Le LLM intervient pour les arbitrages complexes.",
        codeSnippets: [
          {
            language: "python",
            code: `from dataclasses import dataclass
from datetime import datetime, timedelta, time
import pytz

@dataclass
class SlotScore:
    start: datetime
    end: datetime
    total_score: float
    availability_score: float
    preference_score: float
    cognitive_load_score: float
    timezone_score: float
    details: dict

@dataclass
class UserPreferences:
    email: str
    timezone: str
    protected_hours: list[dict]  # [{"day": "monday", "start": "09:00", "end": "12:00", "reason": "deep work"}]
    no_meeting_days: list[str]   # ["friday"]
    max_meetings_per_day: int
    min_break_minutes: int       # Pause minimale entre 2 réunions
    preferred_hours: dict        # {"start": "10:00", "end": "17:00"}

class SlotOptimizer:
    def __init__(self, calendar_connector):
        self.calendar = calendar_connector
        self.weights = {
            "availability": 0.35,
            "preference": 0.25,
            "cognitive_load": 0.25,
            "timezone": 0.15,
        }

    def find_optimal_slots(
        self, participants: list[dict], duration_minutes: int,
        search_window_days: int = 5, priority: str = "normal"
    ) -> list[SlotScore]:
        """Trouve les 3 meilleurs créneaux pour une réunion."""
        now = datetime.now(pytz.timezone("Europe/Paris"))
        search_end = now + timedelta(days=search_window_days)

        # Collecter les disponibilités de tous les participants
        all_busy = {}
        all_events = {}
        all_prefs = {}
        for p in participants:
            email = p["email"]
            tz = pytz.timezone(p.get("timezone", "Europe/Paris"))
            all_busy[email] = self.calendar.get_busy_slots(email, now, search_end)
            all_events[email] = self.calendar.get_events_detail(email, now, search_end)
            all_prefs[email] = self._load_preferences(email)

        # Générer tous les créneaux possibles (par tranches de 30 min)
        candidates = self._generate_candidates(now, search_end, duration_minutes)

        # Scorer chaque créneau
        scored = []
        for slot_start, slot_end in candidates:
            score = self._score_slot(
                slot_start, slot_end, participants,
                all_busy, all_events, all_prefs, priority
            )
            if score.availability_score > 0:  # Au moins un créneau dispo
                scored.append(score)

        # Retourner le top 3
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored[:3]

    def _score_slot(self, start, end, participants, all_busy,
                    all_events, all_prefs, priority) -> SlotScore:
        avail_scores = []
        pref_scores = []
        cognitive_scores = []
        tz_scores = []

        for p in participants:
            email = p["email"]
            prefs = all_prefs.get(email)

            # Disponibilité (0 ou 1)
            is_free = not any(
                self._overlaps(start, end, b["start"], b["end"])
                for b in all_busy.get(email, [])
            )
            avail_scores.append(1.0 if is_free else 0.0)

            # Préférences (0 à 1)
            pref_score = self._score_preferences(start, end, prefs)
            pref_scores.append(pref_score)

            # Charge cognitive (0 à 1)
            day_events = [e for e in all_events.get(email, [])
                          if self._same_day(start, e["start"])]
            max_meetings = prefs.max_meetings_per_day if prefs else 6
            load = 1.0 - (len(day_events) / max(max_meetings, 1))
            cognitive_scores.append(max(load, 0.0))

            # Fuseau horaire (0 à 1) - heure locale acceptable ?
            tz = pytz.timezone(p.get("timezone", "Europe/Paris"))
            local_hour = start.astimezone(tz).hour
            tz_score = 1.0 if 9 <= local_hour <= 17 else (0.5 if 8 <= local_hour <= 18 else 0.0)
            tz_scores.append(tz_score)

        availability = sum(avail_scores) / len(avail_scores)
        preference = sum(pref_scores) / len(pref_scores)
        cognitive = sum(cognitive_scores) / len(cognitive_scores)
        timezone = sum(tz_scores) / len(tz_scores)

        total = (
            availability * self.weights["availability"]
            + preference * self.weights["preference"]
            + cognitive * self.weights["cognitive_load"]
            + timezone * self.weights["timezone"]
        )

        return SlotScore(
            start=start, end=end, total_score=total,
            availability_score=availability, preference_score=preference,
            cognitive_load_score=cognitive, timezone_score=timezone,
            details={"participants_free": sum(avail_scores), "total_participants": len(participants)},
        )

    def _generate_candidates(self, start, end, duration):
        candidates = []
        current = start.replace(hour=8, minute=0, second=0, microsecond=0)
        if current < start:
            current += timedelta(days=1)
        while current < end:
            if current.weekday() < 5:  # Lundi-Vendredi
                slot_start = current.replace(hour=8, minute=0)
                day_end = current.replace(hour=19, minute=0)
                while slot_start + timedelta(minutes=duration) <= day_end:
                    candidates.append((slot_start, slot_start + timedelta(minutes=duration)))
                    slot_start += timedelta(minutes=30)
            current += timedelta(days=1)
        return candidates

    def _overlaps(self, s1, e1, s2, e2):
        if isinstance(s2, str):
            s2 = datetime.fromisoformat(s2)
            e2 = datetime.fromisoformat(e2)
        return s1 < e2 and s2 < e1

    def _same_day(self, dt, dt_str):
        if isinstance(dt_str, str):
            other = datetime.fromisoformat(dt_str)
        else:
            other = dt_str
        return dt.date() == other.date()

    def _score_preferences(self, start, end, prefs):
        if not prefs:
            return 0.5
        score = 1.0
        day_name = start.strftime("%A").lower()
        if day_name in [d.lower() for d in prefs.no_meeting_days]:
            score -= 0.8
        for protected in prefs.protected_hours:
            if protected["day"].lower() == day_name:
                p_start = time.fromisoformat(protected["start"])
                p_end = time.fromisoformat(protected["end"])
                if start.time() < p_end and end.time() > p_start:
                    score -= 0.6
        return max(score, 0.0)

    def _load_preferences(self, email):
        result = supabase.table("user_preferences").select("*").eq("email", email).execute()
        if result.data:
            d = result.data[0]
            return UserPreferences(**d)
        return None`,
            filename: "slot_optimizer.py",
          },
        ],
      },
      {
        title: "Génération d'ordre du jour intelligent",
        content:
          "L'agent génère automatiquement un ordre du jour structuré à partir du sujet de la réunion, de l'historique des réunions précédentes sur le même sujet, et des documents pertinents. Il estime la durée de chaque point et propose un minutage réaliste.",
        codeSnippets: [
          {
            language: "python",
            code: `from openai import OpenAI
import json

client = OpenAI()

AGENDA_PROMPT = """Tu es un facilitateur de réunion professionnel dans une entreprise française.

Génère un ordre du jour structuré pour la réunion suivante:

## Sujet: {subject}
## Objectifs: {objectives}
## Participants: {participants}
## Durée totale: {duration} minutes
## Contexte: {context}
## Notes de la dernière réunion sur ce sujet: {previous_notes}

## Consignes:
1. Structure l'ordre du jour en 4-6 points maximum
2. Attribue une durée réaliste à chaque point (total = durée de la réunion)
3. Identifie le responsable de chaque point parmi les participants
4. Commence par un tour de table rapide (5 min max)
5. Termine par les décisions à prendre et prochaines étapes
6. Indique les documents à préparer avant la réunion

Retourne un JSON avec: points (array de {{title, duration_min, owner, description}}),
preparation_docs (array de strings), expected_outcomes (array de strings)"""

def generate_agenda(meeting: dict) -> dict:
    """Génère un ordre du jour intelligent pour une réunion."""
    # Rechercher les réunions précédentes sur le même sujet
    previous = supabase.table("meeting_notes").select("*").ilike(
        "subject", f"%{meeting['subject']}%"
    ).order("date", desc=True).limit(3).execute()

    previous_notes = "\\n---\\n".join(
        [f"{p['date']}: {p['summary']}" for p in previous.data]
    ) if previous.data else "Aucune réunion précédente trouvée."

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": AGENDA_PROMPT.format(
                subject=meeting["subject"],
                objectives=meeting.get("objectives", "À définir"),
                participants=", ".join([p["name"] for p in meeting["participants"]]),
                duration=meeting["duration_minutes"],
                context=meeting.get("context", ""),
                previous_notes=previous_notes,
            ),
        }],
    )
    return json.loads(response.choices[0].message.content)`,
            filename: "agenda_generator.py",
          },
        ],
      },
      {
        title: "Analyse de charge réunion et suggestions d'optimisation",
        content:
          "L'agent analyse la charge de réunions de chaque collaborateur et identifie les opportunités d'optimisation : réunions récurrentes qui pourraient être remplacées par un message asynchrone, réunions trop longues, participants non essentiels. Il génère un rapport hebdomadaire avec des recommandations.",
        codeSnippets: [
          {
            language: "python",
            code: `MEETING_AUDIT_PROMPT = """Tu es un consultant en productivité organisationnelle.

Analyse le planning de réunions de cette semaine pour l'équipe et identifie les optimisations:

## Réunions de la semaine:
{meetings_json}

## Statistiques:
- Nombre total de réunions: {total_meetings}
- Temps total en réunion: {total_hours}h
- Moyenne par personne: {avg_per_person}h
- Collaborateur le plus chargé: {busiest_person} ({busiest_hours}h)

## Analyse demandée:
1. Identifie les réunions qui pourraient être un email/Slack
2. Repère les réunions récurrentes avec trop de participants
3. Détecte les créneaux surchargés
4. Propose des fusions de réunions sur des sujets proches
5. Calcule le coût estimé en heures-personne

Retourne un JSON: recommendations (array), savings_hours, meetings_to_cancel (array), meetings_to_merge (array de pairs)"""

def weekly_meeting_audit(team_emails: list[str]) -> dict:
    """Audit hebdomadaire de la charge de réunions."""
    now = datetime.now(pytz.timezone("Europe/Paris"))
    week_start = now - timedelta(days=now.weekday())
    week_end = week_start + timedelta(days=5)

    all_meetings = []
    per_person = {}
    for email in team_emails:
        events = calendar.get_events_detail(email, week_start, week_end)
        per_person[email] = {
            "count": len(events),
            "hours": sum(calculate_duration(e) for e in events) / 60,
        }
        all_meetings.extend(events)

    total_hours = sum(p["hours"] for p in per_person.values())
    busiest = max(per_person.items(), key=lambda x: x[1]["hours"])

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": MEETING_AUDIT_PROMPT.format(
                meetings_json=json.dumps(all_meetings[:50], ensure_ascii=False, default=str),
                total_meetings=len(all_meetings),
                total_hours=round(total_hours, 1),
                avg_per_person=round(total_hours / max(len(team_emails), 1), 1),
                busiest_person=busiest[0],
                busiest_hours=round(busiest[1]["hours"], 1),
            ),
        }],
    )
    return json.loads(response.choices[0].message.content)

def calculate_duration(event: dict) -> float:
    """Calcule la durée d'un événement en minutes."""
    start = datetime.fromisoformat(event["start"])
    end = datetime.fromisoformat(event["end"])
    return (end - start).total_seconds() / 60`,
            filename: "meeting_audit.py",
          },
        ],
      },
      {
        title: "API et bot Slack d'interaction",
        content:
          "Exposez l'agent via une API REST et un bot Slack pour permettre aux collaborateurs de planifier des réunions en langage naturel. Le bot comprend des requêtes comme 'Planifie un point hebdo de 30 min avec l'équipe produit' et gère tout automatiquement.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI, Request
from openai import OpenAI
import json

app = FastAPI(title="Agent Planification Réunions")
client = OpenAI()

INTENT_PROMPT = """Tu es un assistant de planification de réunions.
Extrais les informations de la demande utilisateur:

Demande: {user_message}

Retourne un JSON:
- action: "schedule" | "reschedule" | "cancel" | "audit"
- subject: string (sujet de la réunion)
- participants: array de strings (emails ou noms)
- duration_minutes: int
- priority: "low" | "normal" | "high"
- constraints: string (contraintes spécifiques mentionnées)
- recurring: boolean
- recurring_pattern: string | null ("weekly", "biweekly", "monthly")"""

@app.post("/api/slack/events")
async def handle_slack_event(request: Request):
    """Gère les messages Slack pour planification en langage naturel."""
    body = await request.json()

    if body.get("type") == "url_verification":
        return {"challenge": body["challenge"]}

    event = body.get("event", {})
    if event.get("type") != "app_mention":
        return {"ok": True}

    user_message = event["text"]
    channel = event["channel"]

    # Extraire l'intention
    intent_response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": INTENT_PROMPT.format(user_message=user_message)}],
    )
    intent = json.loads(intent_response.choices[0].message.content)

    if intent["action"] == "schedule":
        # Résoudre les participants
        participants = resolve_participants(intent["participants"])

        # Trouver les créneaux optimaux
        optimizer = SlotOptimizer(calendar)
        slots = optimizer.find_optimal_slots(
            participants=participants,
            duration_minutes=intent["duration_minutes"],
            priority=intent["priority"],
        )

        if not slots:
            post_slack_message(channel, "Aucun créneau disponible trouvé dans les 5 prochains jours.")
            return {"ok": True}

        # Proposer les 3 meilleurs créneaux
        message = f"*Planification: {intent['subject']}*\\n"
        message += f"Durée: {intent['duration_minutes']} min | Participants: {len(participants)}\\n\\n"
        for i, slot in enumerate(slots):
            local_time = slot.start.strftime("%A %d/%m à %Hh%M")
            message += f"{i+1}. {local_time} (score: {slot.total_score:.0%})\\n"
        message += "\\nRépondez avec le numéro de votre choix."

        post_slack_message(channel, message)

    elif intent["action"] == "audit":
        team_emails = resolve_team_emails(intent.get("participants", []))
        audit = weekly_meeting_audit(team_emails)
        post_slack_message(channel, format_audit_report(audit))

    return {"ok": True}`,
            filename: "slack_bot.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "L'accès aux calendriers expose les rendez-vous personnels (médecin, entretiens). Configurer le connecteur pour ne remonter que les informations de disponibilité (free/busy) sans les détails des événements personnels. Les événements marqués 'privé' dans Google/Outlook sont traités comme des créneaux occupés sans exposer le contenu. Conformité RGPD via consentement explicite de chaque collaborateur.",
      auditLog: "Chaque planification est tracée : demandeur, participants, créneaux proposés, créneau choisi, salle réservée, ordre du jour généré. Les accès aux calendriers sont loggués avec timestamp et périmètre. Rétention 6 mois. Export CSV pour audit RH sur la charge de réunions.",
      humanInTheLoop: "L'organisateur valide toujours le créneau final parmi les propositions de l'agent. Les réunions impliquant plus de 10 personnes ou des membres du comité de direction nécessitent une confirmation explicite. Les suggestions d'annulation de réunions récurrentes sont soumises à l'organisateur original.",
      monitoring: "Dashboard Langfuse : nombre de réunions planifiées par jour, score moyen de satisfaction des créneaux, taux d'acceptation des propositions, temps moyen de planification, nombre de conflits résolus, charge de réunion moyenne par équipe. Alertes si le taux d'acceptation descend sous 70% ou si le temps de réponse de l'API calendrier dépasse 5 secondes.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Slack Trigger (mention du bot) -> Node Code (extraction intention via GPT-4.1) -> Node Switch (action: schedule/reschedule/audit) -> Branch schedule : Node Google Calendar (get free/busy) -> Node Code (algorithme scoring créneaux) -> Node Slack (proposition 3 créneaux) -> Node Wait (choix utilisateur) -> Node Google Calendar (création événement) -> Node HTTP Request (génération ordre du jour) -> Node Slack (confirmation + agenda). Branch audit : Node Google Calendar (get events semaine) -> Node HTTP Request (GPT-4.1 analyse) -> Node Slack (rapport).",
      nodes: ["Slack Trigger (mention)", "Code (extraction intention)", "Switch (action)", "Google Calendar (free/busy)", "Code (scoring créneaux)", "Slack (proposition)", "Wait (choix)", "Google Calendar (créer événement)", "HTTP Request (ordre du jour)", "Slack (confirmation)"],
      triggerType: "Slack mention du bot + Cron hebdomadaire (audit lundi 7h)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["Services", "Tech", "Conseil", "Banque"],
    metiers: ["IT", "Management", "Assistanat de Direction"],
    functions: ["IT", "Organisation", "Productivité"],
    metaTitle: "Agent IA de Planification de Réunions Intelligente — Guide Complet",
    metaDescription:
      "Automatisez la planification de réunions avec un agent IA qui optimise les créneaux selon les disponibilités, fuseaux horaires et charge cognitive. Bot Slack inclus.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-fraude-transactionnelle-temps-reel",
    title: "Agent de Détection de Fraude Transactionnelle en Temps Réel",
    subtitle: "Combinez scoring ML et analyse contextuelle LLM pour détecter les fraudes bancaires avec un taux de faux positifs réduit de 70%",
    problem:
      "Les banques et fintechs françaises font face à une explosion des fraudes transactionnelles (+30% par an). Les systèmes traditionnels basés sur des règles statiques (montant > seuil, pays à risque) génèrent jusqu'à 95% de faux positifs, mobilisant des dizaines d'analystes sur des alertes non pertinentes. Parallèlement, les fraudeurs sophistiqués contournent ces règles en fragmentant les montants et en utilisant des schémas comportementaux inédits. Le coût de la fraude non détectée et du traitement des faux positifs dépasse 2% du chiffre d'affaires pour les acteurs du paiement. Les exigences de la DSP2 imposent une authentification forte tout en maintenant une expérience client fluide.",
    value:
      "Un agent IA combine un modèle de scoring ML en temps réel (< 100ms par transaction) avec un LLM pour l'analyse contextuelle des transactions suspectes. Le ML filtre 99% des transactions légitimes. Les 1% restants sont analysés par le LLM qui examine le contexte comportemental complet du client, génère une explication en langage naturel, et recommande une action (bloquer, authentifier, laisser passer). Les faux positifs sont réduits de 70% et les fraudes non détectées de 40%.",
    inputs: [
      "Flux de transactions en temps réel (montant, devise, marchand, géolocalisation, device)",
      "Profil comportemental historique du porteur (habitudes de dépenses, lieux fréquents)",
      "Données de device fingerprinting (IP, user agent, empreinte navigateur)",
      "Signaux de vélocité (nombre de transactions dans les dernières heures)",
      "Base de marchands à risque et BIN blacklistés",
      "Historique des fraudes confirmées pour entraînement du modèle",
    ],
    outputs: [
      "Score de fraude en temps réel (0-1000) avec seuils configurables",
      "Explication LLM en langage naturel de la décision",
      "Action recommandée (approuver, challenger 3DS, bloquer, escalader)",
      "Graphe de liens entre transactions suspectes (détection de réseaux)",
      "Rapport quotidien des patterns de fraude émergents",
      "Métriques de performance : taux de détection, faux positifs, latence",
    ],
    risks: [
      "Latence excessive bloquant l'expérience de paiement (SLA < 200ms)",
      "Faux positifs restants générant de la friction client et des pertes commerciales",
      "Biais du modèle ML discriminant certains profils démographiques",
      "Attaques adversariales contre le modèle de scoring",
      "Non-conformité réglementaire si les décisions ne sont pas explicables (IA Act EU)",
    ],
    roiIndicatif:
      "Réduction de 70% des faux positifs. Augmentation de 40% du taux de détection des fraudes. Économie estimée : 2-5M EUR/an pour une banque traitant 10M transactions/mois. Réduction de 60% de la charge des analystes fraude.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "scikit-learn", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Llama 3", category: "LLM", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "PostgreSQL + Redis", category: "Database", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
      { name: "XGBoost", category: "Other", isFree: true },
    ],
    architectureDiagram: `┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Transaction │────▶│  Feature     │────▶│  ML Scoring  │
│  Stream      │     │  Engineering │     │  (< 50ms)    │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                           ┌──────▼───────┐
                                           │  Seuil ML    │
                                           │  Score > 500 ?│
                                           └──────┬───────┘
                                          ┌───────┴───────┐
                                     NON  │               │ OUI
                                ┌─────────▼──┐    ┌───────▼──────┐
                                │  Approuver │    │  Agent LLM   │
                                │  (auto)    │    │  (Contexte)  │
                                └────────────┘    └──────┬───────┘
                                                         │
                                                  ┌──────▼───────┐
                                                  │  Décision    │
                                                  │  + Explication│
                                                  └──────────────┘`,
    tutorial: [
      {
        title: "Feature engineering pour le scoring transactionnel",
        content:
          "Construisez le pipeline de feature engineering qui transforme chaque transaction brute en un vecteur de features exploitables par le modèle ML. Les features incluent des signaux de vélocité, d'écart au comportement habituel, de géolocalisation et de device.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain anthropic supabase scikit-learn xgboost pandas numpy redis python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

@dataclass
class TransactionFeatures:
    # Features brutes
    amount: float
    is_international: bool
    is_online: bool
    hour_of_day: int
    day_of_week: int
    # Features de vélocité
    tx_count_1h: int
    tx_count_24h: int
    total_amount_1h: float
    total_amount_24h: float
    # Features comportementales
    amount_vs_avg_ratio: float  # montant / moyenne habituelle
    amount_vs_max_ratio: float  # montant / max historique
    new_merchant: bool
    new_country: bool
    new_device: bool
    # Features de distance
    distance_from_last_tx_km: float
    time_since_last_tx_minutes: float
    # Features agrégées
    distinct_merchants_24h: int
    distinct_countries_24h: int
    declined_count_24h: int

class FeatureEngine:
    def __init__(self):
        self.redis = r

    def compute_features(self, tx: dict, card_id: str) -> TransactionFeatures:
        """Calcule les features en temps réel pour une transaction."""
        now = datetime.utcnow()
        profile = self._get_cardholder_profile(card_id)
        recent_txs = self._get_recent_transactions(card_id, hours=24)
        recent_1h = [t for t in recent_txs
                     if (now - datetime.fromisoformat(t["timestamp"])).seconds < 3600]

        return TransactionFeatures(
            amount=tx["amount"],
            is_international=tx.get("country", "FR") != "FR",
            is_online=tx.get("channel") == "ecommerce",
            hour_of_day=now.hour,
            day_of_week=now.weekday(),
            tx_count_1h=len(recent_1h),
            tx_count_24h=len(recent_txs),
            total_amount_1h=sum(t["amount"] for t in recent_1h),
            total_amount_24h=sum(t["amount"] for t in recent_txs),
            amount_vs_avg_ratio=tx["amount"] / max(profile.get("avg_amount", 50), 1),
            amount_vs_max_ratio=tx["amount"] / max(profile.get("max_amount", 100), 1),
            new_merchant=tx.get("merchant_id") not in profile.get("known_merchants", []),
            new_country=tx.get("country") not in profile.get("known_countries", ["FR"]),
            new_device=tx.get("device_hash") not in profile.get("known_devices", []),
            distance_from_last_tx_km=self._calc_distance(tx, recent_txs),
            time_since_last_tx_minutes=self._time_since_last(recent_txs),
            distinct_merchants_24h=len(set(t.get("merchant_id") for t in recent_txs)),
            distinct_countries_24h=len(set(t.get("country") for t in recent_txs)),
            declined_count_24h=sum(1 for t in recent_txs if t.get("declined")),
        )

    def _get_cardholder_profile(self, card_id: str) -> dict:
        """Récupère le profil comportemental depuis Redis."""
        profile = self.redis.get(f"profile:{card_id}")
        return json.loads(profile) if profile else {}

    def _get_recent_transactions(self, card_id: str, hours: int) -> list:
        """Récupère les transactions récentes depuis Redis."""
        txs = self.redis.lrange(f"txs:{card_id}", 0, 200)
        return [json.loads(t) for t in txs]

    def _calc_distance(self, tx, recent_txs):
        if not recent_txs or "lat" not in tx:
            return 0.0
        last = recent_txs[-1]
        if "lat" not in last:
            return 0.0
        # Haversine simplifié
        dlat = abs(tx["lat"] - last["lat"])
        dlon = abs(tx["lon"] - last["lon"])
        return (dlat**2 + dlon**2)**0.5 * 111  # Approximation km

    def _time_since_last(self, recent_txs):
        if not recent_txs:
            return 999
        last = datetime.fromisoformat(recent_txs[-1]["timestamp"])
        return (datetime.utcnow() - last).total_seconds() / 60`,
            filename: "feature_engine.py",
          },
        ],
      },
      {
        title: "Modèle ML de scoring en temps réel",
        content:
          "Entraînez un modèle XGBoost sur les transactions historiques étiquetées (fraude/légitime). Le modèle doit produire un score en moins de 50ms. Utilisez un pipeline de sérialisation pour le déploiement en production. Le scoring ML filtre 99% des transactions avant intervention du LLM.",
        codeSnippets: [
          {
            language: "python",
            code: `import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score
import joblib

class FraudScoringModel:
    def __init__(self, model_path: str | None = None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = None

    def train(self, labeled_data: pd.DataFrame):
        """Entraîne le modèle sur les transactions étiquetées."""
        feature_cols = [
            "amount", "is_international", "is_online", "hour_of_day",
            "day_of_week", "tx_count_1h", "tx_count_24h", "total_amount_1h",
            "total_amount_24h", "amount_vs_avg_ratio", "amount_vs_max_ratio",
            "new_merchant", "new_country", "new_device",
            "distance_from_last_tx_km", "time_since_last_tx_minutes",
            "distinct_merchants_24h", "distinct_countries_24h", "declined_count_24h",
        ]
        X = labeled_data[feature_cols]
        y = labeled_data["is_fraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Gestion du déséquilibre (fraudes = ~0.1% des transactions)
        scale_pos = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)

        self.model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            use_label_encoder=False,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Métriques
        y_prob = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        print(f"AUC-ROC: {auc:.4f}")

        # Sauvegarder
        joblib.dump(self.model, "fraud_model.joblib")
        return {"auc_roc": auc, "model_path": "fraud_model.joblib"}

    def score(self, features: dict) -> dict:
        """Score une transaction en temps réel (< 50ms)."""
        feature_array = np.array([[
            features.amount, features.is_international, features.is_online,
            features.hour_of_day, features.day_of_week,
            features.tx_count_1h, features.tx_count_24h,
            features.total_amount_1h, features.total_amount_24h,
            features.amount_vs_avg_ratio, features.amount_vs_max_ratio,
            features.new_merchant, features.new_country, features.new_device,
            features.distance_from_last_tx_km, features.time_since_last_tx_minutes,
            features.distinct_merchants_24h, features.distinct_countries_24h,
            features.declined_count_24h,
        ]])
        proba = self.model.predict_proba(feature_array)[0][1]
        score = int(proba * 1000)  # Score 0-1000
        return {
            "score": score,
            "probability": float(proba),
            "risk_level": "high" if score > 700 else ("medium" if score > 400 else "low"),
        }`,
            filename: "scoring_model.py",
          },
        ],
      },
      {
        title: "Analyse contextuelle LLM des transactions suspectes",
        content:
          "Pour les transactions ayant un score ML élevé (> 500), le LLM analyse le contexte complet : profil comportemental du client, historique des transactions récentes, cohérence géographique et temporelle. Il génère une explication humainement lisible et une recommandation d'action.",
        codeSnippets: [
          {
            language: "python",
            code: `from anthropic import Anthropic
import json

client = Anthropic()

FRAUD_ANALYSIS_PROMPT = """Tu es un analyste fraude senior dans une banque française.

Analyse cette transaction suspecte et décide de l'action à prendre.

## Transaction suspecte:
- Montant: {amount} {currency}
- Marchand: {merchant_name} ({merchant_category})
- Pays: {country}
- Canal: {channel}
- Date/Heure: {timestamp}
- Score ML: {ml_score}/1000

## Profil du porteur:
- Client depuis: {client_since}
- Montant moyen habituel: {avg_amount} EUR
- Montant max historique: {max_amount} EUR
- Pays habituels: {usual_countries}
- Catégories marchands habituelles: {usual_categories}
- Dernière transaction: {last_tx_summary}

## Signaux d'alerte:
{alert_signals}

## Transactions récentes (24h):
{recent_transactions}

## Consignes:
1. Analyse la cohérence de la transaction avec le profil
2. Identifie les facteurs suspects ET les facteurs rassurants
3. Détermine l'action: APPROVE, CHALLENGE_3DS, BLOCK, ESCALATE
4. Explique ta décision en 2-3 phrases claires (pour l'analyste humain)
5. Attribue un score de confiance à ta décision (0-100)

Retourne un JSON: action, confidence, explanation, suspicious_factors, reassuring_factors, risk_assessment"""

def analyze_suspicious_transaction(tx: dict, features: dict,
                                    ml_result: dict, profile: dict) -> dict:
    """Analyse contextuelle LLM d'une transaction suspecte."""
    # Construire le résumé des signaux d'alerte
    alerts = []
    if features.new_merchant:
        alerts.append("Marchand jamais utilisé par ce client")
    if features.new_country:
        alerts.append(f"Transaction depuis un nouveau pays: {tx.get('country')}")
    if features.amount_vs_avg_ratio > 3:
        alerts.append(f"Montant {features.amount_vs_avg_ratio:.1f}x supérieur à la moyenne")
    if features.tx_count_1h > 3:
        alerts.append(f"Vélocité élevée: {features.tx_count_1h} transactions en 1h")
    if features.distance_from_last_tx_km > 500:
        alerts.append(f"Distance impossible: {features.distance_from_last_tx_km:.0f}km depuis la dernière tx")

    recent_txs = get_recent_transactions_summary(tx["card_id"])

    response = client.messages.create(
        model="claude-sonnet-4-5-20250514",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": FRAUD_ANALYSIS_PROMPT.format(
                amount=tx["amount"],
                currency=tx.get("currency", "EUR"),
                merchant_name=tx.get("merchant_name", "N/A"),
                merchant_category=tx.get("mcc_description", "N/A"),
                country=tx.get("country", "N/A"),
                channel=tx.get("channel", "N/A"),
                timestamp=tx.get("timestamp", "N/A"),
                ml_score=ml_result["score"],
                client_since=profile.get("client_since", "N/A"),
                avg_amount=profile.get("avg_amount", "N/A"),
                max_amount=profile.get("max_amount", "N/A"),
                usual_countries=", ".join(profile.get("known_countries", [])),
                usual_categories=", ".join(profile.get("usual_categories", [])),
                last_tx_summary=profile.get("last_tx_summary", "N/A"),
                alert_signals="\\n".join(f"- {a}" for a in alerts) or "Aucun signal majeur",
                recent_transactions=recent_txs,
            ),
        }],
    )
    return json.loads(response.content[0].text)`,
            filename: "llm_analyzer.py",
          },
        ],
      },
      {
        title: "Pipeline temps réel et orchestration",
        content:
          "Assemblez le pipeline complet qui traite chaque transaction en moins de 200ms : feature engineering (20ms), scoring ML (30ms), et analyse LLM conditionnelle (150ms pour les cas suspects uniquement). Utilisez Redis pour le cache et les profils en mémoire.",
        codeSnippets: [
          {
            language: "python",
            code: `from fastapi import FastAPI
from pydantic import BaseModel
import time
from typing import Optional

app = FastAPI(title="Agent Détection Fraude")
feature_engine = FeatureEngine()
scoring_model = FraudScoringModel("fraud_model.joblib")

ML_THRESHOLD = 500      # Score ML au-dessus duquel le LLM intervient
BLOCK_THRESHOLD = 800   # Score ML au-dessus duquel on bloque directement

class TransactionRequest(BaseModel):
    card_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_name: str
    mcc_code: str
    country: str
    channel: str
    device_hash: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None

class FraudDecision(BaseModel):
    action: str  # APPROVE, CHALLENGE_3DS, BLOCK, ESCALATE
    ml_score: int
    llm_analysis: Optional[dict] = None
    explanation: str
    processing_time_ms: int

@app.post("/api/fraud/check", response_model=FraudDecision)
async def check_transaction(tx: TransactionRequest):
    """Vérifie une transaction en temps réel."""
    start = time.time()
    tx_dict = tx.model_dump()
    tx_dict["timestamp"] = datetime.utcnow().isoformat()

    # Étape 1: Feature engineering (< 20ms)
    features = feature_engine.compute_features(tx_dict, tx.card_id)

    # Étape 2: Scoring ML (< 30ms)
    ml_result = scoring_model.score(features)

    # Étape 3: Décision rapide pour les cas clairs
    if ml_result["score"] < ML_THRESHOLD:
        elapsed = int((time.time() - start) * 1000)
        save_decision(tx_dict, ml_result, None, "APPROVE", elapsed)
        return FraudDecision(
            action="APPROVE",
            ml_score=ml_result["score"],
            explanation="Transaction conforme au profil habituel du porteur.",
            processing_time_ms=elapsed,
        )

    if ml_result["score"] > BLOCK_THRESHOLD:
        elapsed = int((time.time() - start) * 1000)
        save_decision(tx_dict, ml_result, None, "BLOCK", elapsed)
        return FraudDecision(
            action="BLOCK",
            ml_score=ml_result["score"],
            explanation="Score de risque critique. Transaction bloquée préventivement.",
            processing_time_ms=elapsed,
        )

    # Étape 4: Analyse LLM pour les cas ambigus (score 500-800)
    profile = feature_engine._get_cardholder_profile(tx.card_id)
    llm_result = analyze_suspicious_transaction(tx_dict, features, ml_result, profile)

    elapsed = int((time.time() - start) * 1000)
    action = llm_result.get("action", "CHALLENGE_3DS")

    save_decision(tx_dict, ml_result, llm_result, action, elapsed)
    return FraudDecision(
        action=action,
        ml_score=ml_result["score"],
        llm_analysis=llm_result,
        explanation=llm_result.get("explanation", "Analyse contextuelle effectuée."),
        processing_time_ms=elapsed,
    )

def save_decision(tx, ml_result, llm_result, action, elapsed_ms):
    """Sauvegarde la décision pour audit et entraînement."""
    supabase.table("fraud_decisions").insert({
        "transaction": tx,
        "ml_score": ml_result["score"],
        "llm_analysis": llm_result,
        "action": action,
        "processing_time_ms": elapsed_ms,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()`,
            filename: "fraud_pipeline.py",
          },
        ],
      },
      {
        title: "Monitoring, feedback loop et réentraînement",
        content:
          "Mettez en place le circuit de feedback qui permet aux analystes fraude de confirmer ou infirmer les décisions de l'agent. Ces retours alimentent le réentraînement hebdomadaire du modèle ML. Configurez les alertes de dérive du modèle.",
        codeSnippets: [
          {
            language: "python",
            code: `from datetime import datetime, timedelta

class FraudFeedbackLoop:
    def __init__(self):
        self.metrics = {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }

    def record_analyst_feedback(self, decision_id: str, is_fraud: bool):
        """Enregistre le feedback de l'analyste sur une décision."""
        decision = supabase.table("fraud_decisions").select("*").eq(
            "id", decision_id
        ).single().execute().data

        original_action = decision["action"]
        was_flagged = original_action in ["BLOCK", "CHALLENGE_3DS", "ESCALATE"]

        if is_fraud and was_flagged:
            category = "true_positive"
        elif is_fraud and not was_flagged:
            category = "false_negative"
        elif not is_fraud and was_flagged:
            category = "false_positive"
        else:
            category = "true_negative"

        supabase.table("fraud_feedback").insert({
            "decision_id": decision_id,
            "is_confirmed_fraud": is_fraud,
            "original_action": original_action,
            "feedback_category": category,
            "analyst_id": "system",
            "created_at": datetime.utcnow().isoformat(),
        }).execute()

    def compute_daily_metrics(self) -> dict:
        """Calcule les métriques quotidiennes de performance."""
        yesterday = (datetime.utcnow() - timedelta(days=1)).isoformat()
        feedbacks = supabase.table("fraud_feedback").select("*").gte(
            "created_at", yesterday
        ).execute()

        metrics = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for f in feedbacks.data:
            cat = f["feedback_category"]
            if cat == "true_positive":
                metrics["tp"] += 1
            elif cat == "false_positive":
                metrics["fp"] += 1
            elif cat == "true_negative":
                metrics["tn"] += 1
            elif cat == "false_negative":
                metrics["fn"] += 1

        precision = metrics["tp"] / max(metrics["tp"] + metrics["fp"], 1)
        recall = metrics["tp"] / max(metrics["tp"] + metrics["fn"], 1)

        report = {
            "date": datetime.utcnow().date().isoformat(),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(2 * precision * recall / max(precision + recall, 0.001), 4),
            "false_positive_rate": round(metrics["fp"] / max(sum(metrics.values()), 1), 4),
            "total_reviewed": sum(metrics.values()),
            "metrics": metrics,
        }

        # Alerte si les métriques se dégradent
        if precision < 0.3:
            send_alert("Précision fraude sous 30% - réentraînement nécessaire")
        if recall < 0.8:
            send_alert("Rappel fraude sous 80% - fraudes non détectées en hausse")

        return report`,
            filename: "feedback_loop.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les transactions contiennent des données personnelles sensibles (numéro de carte, géolocalisation, habitudes de consommation). Tokenisation PCI-DSS obligatoire des PAN avant traitement. Les données envoyées au LLM sont pseudonymisées (pas de numéro de carte complet, pas de nom du porteur). Stockage chiffré AES-256 en base. Conformité RGPD : base légale = intérêt légitime (prévention de la fraude, art. 6.1.f).",
      auditLog: "Piste d'audit complète pour chaque décision : timestamp, features calculées, score ML, analyse LLM complète (prompt + réponse), action prise, temps de traitement, feedback analyste ultérieur. Conservation 5 ans (obligation réglementaire ACPR). Export automatique pour les rapports de contrôle interne et les audits de la Banque de France.",
      humanInTheLoop: "Les transactions avec une action ESCALATE sont systématiquement traitées par un analyste fraude dans un SLA de 15 minutes. Les décisions de blocage avec un score de confiance LLM < 60% déclenchent une revue humaine avant notification au client. Un comité hebdomadaire revoit les faux positifs et faux négatifs pour ajuster les seuils.",
      monitoring: "Dashboard temps réel Grafana : volume de transactions/seconde, latence P50/P95/P99, taux de blocage, taux de faux positifs (mise à jour quotidienne via feedback), distribution des scores ML, dérive du modèle (PSI - Population Stability Index), coût LLM par jour. Alertes critiques : latence > 200ms, taux de blocage > 5%, PSI > 0.2.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (nouvelle transaction via API gateway) -> Node Code (feature engineering depuis Redis) -> Node HTTP Request (scoring ML via API interne) -> Node Switch (score < 500 / 500-800 / > 800) -> Branch approuver : Node HTTP Request (réponse approve) -> Branch suspecte : Node HTTP Request (API Claude - analyse contextuelle) -> Node Code (décision finale) -> Node Switch (action) -> Node HTTP Request (réponse) + Node Supabase (log décision) -> Branch bloquer : Node HTTP Request (réponse block) + Node Slack (alerte analyste).",
      nodes: ["Webhook (transaction)", "Code (features Redis)", "HTTP Request (ML scoring)", "Switch (score threshold)", "HTTP Request (Claude analyse)", "Code (décision finale)", "Switch (action)", "HTTP Request (réponse gateway)", "Supabase (audit log)", "Slack (alerte analyste)"],
      triggerType: "Webhook (chaque transaction en temps réel via API gateway PSP)",
    },
    estimatedTime: "12-20h",
    difficulty: "Expert",
    sectors: ["Banque", "Fintech", "Assurance", "E-commerce"],
    metiers: ["Analyse Fraude", "Risk Management", "Data Science"],
    functions: ["Risk", "Sécurité", "Data"],
    metaTitle: "Agent IA de Détection de Fraude Transactionnelle — Guide Expert",
    metaDescription:
      "Construisez un agent IA combinant ML temps réel et analyse contextuelle LLM pour détecter la fraude bancaire. Réduction de 70% des faux positifs. Pipeline complet avec code.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },
  {
    slug: "agent-personnalisation-email-marketing",
    title: "Agent de Personnalisation Email Marketing",
    subtitle: "Générez des emails marketing hyper-personnalisés à grande échelle en combinant segmentation IA et rédaction contextuelle par LLM",
    problem:
      "Les équipes marketing envoient des campagnes email segmentées de manière rudimentaire (âge, sexe, localisation) avec des contenus génériques qui génèrent des taux d'ouverture de 15-20% et des taux de clic inférieurs à 3%. La personnalisation manuelle est impossible au-delà de 5-6 segments. Les A/B tests sont limités à 2-3 variantes par campagne, laissant inexploité le potentiel de personnalisation massive. Les désabonnements augmentent car les destinataires reçoivent du contenu non pertinent. Le coût d'acquisition client via email augmente tandis que le ROI se dégrade.",
    value:
      "Un agent IA analyse le profil comportemental complet de chaque destinataire (historique d'achats, navigation, interactions email précédentes, préférences déclarées) et génère un email entièrement personnalisé : objet, corps du texte, recommandations produits, timing d'envoi optimal, et tonalité adaptée. Chaque destinataire reçoit un email unique. Les taux d'ouverture augmentent de 40% et les conversions de 25%.",
    inputs: [
      "Base de contacts avec données comportementales (achats, navigation, clics email)",
      "Catalogue produits avec descriptions, prix et disponibilité",
      "Historique des campagnes précédentes (performance par segment)",
      "Charte éditoriale et guidelines de marque",
      "Templates HTML email responsive",
      "Règles RGPD et préférences de consentement par contact",
    ],
    outputs: [
      "Email personnalisé par destinataire (objet, contenu, CTA, produits recommandés)",
      "Heure d'envoi optimale par fuseau horaire et habitude du destinataire",
      "Score de pertinence prédictif par email (probabilité d'engagement)",
      "Rapport de campagne avec attribution des conversions",
      "Suggestions d'optimisation pour les prochaines campagnes",
      "Segments dynamiques identifiés par clustering comportemental",
    ],
    risks: [
      "Hyper-personnalisation perçue comme intrusive par les destinataires",
      "Non-conformité RGPD si le profilage n'est pas déclaré dans la politique de confidentialité",
      "Fatigue email si la fréquence d'envoi n'est pas contrôlée",
      "Hallucinations du LLM inventant des caractéristiques produit inexistantes",
      "Coût LLM élevé si chaque email est généré individuellement sans cache",
    ],
    roiIndicatif:
      "Augmentation de 40% du taux d'ouverture. Augmentation de 25% du taux de conversion. Réduction de 50% du taux de désabonnement. ROI email marketing multiplié par 3.",
    recommendedStack: [
      { name: "Anthropic Claude Sonnet 4.5", category: "LLM" },
      { name: "LangChain", category: "Orchestration" },
      { name: "Supabase", category: "Database" },
      { name: "Vercel", category: "Hosting" },
      { name: "Langfuse", category: "Monitoring" },
      { name: "Resend", category: "Other" },
    ],
    lowCostAlternatives: [
      { name: "Ollama + Mistral Large", category: "LLM", isFree: true },
      { name: "n8n", category: "Orchestration", isFree: true },
      { name: "PostgreSQL", category: "Database", isFree: true },
      { name: "Railway", category: "Hosting", isFree: true },
      { name: "Nodemailer", category: "Other", isFree: true },
    ],
    architectureDiagram: `┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  CRM/CDP     │  │  Catalogue   │  │  Analytics   │
│  Contacts    │  │  Produits    │  │  Email       │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────┬───────┴─────────┬───────┘
                 │                 │
          ┌──────▼───────┐  ┌─────▼────────┐
          │  Agent LLM   │  │  Moteur de   │
          │  (Rédaction)  │  │  Recomm.     │
          └──────┬───────┘  └──────────────┘
                 │
       ┌─────────┼─────────┐
       │         │         │
┌──────▼──┐ ┌───▼────┐ ┌──▼───────┐
│ Email   │ │ Send   │ │ Tracking │
│ Rendu   │ │ Queue  │ │ Analytics│
└─────────┘ └────────┘ └──────────┘`,
    tutorial: [
      {
        title: "Segmentation comportementale et profils destinataires",
        content:
          "Construisez le pipeline de segmentation qui enrichit chaque contact avec un profil comportemental complet. Le profil agrège les données d'achat, de navigation, d'interaction email et de préférences pour créer un portrait unique exploitable par le LLM. Utilisez un clustering pour identifier des micro-segments dynamiques.",
        codeSnippets: [
          {
            language: "bash",
            code: `pip install langchain anthropic supabase resend scikit-learn pandas jinja2 python-dotenv`,
            filename: "terminal",
          },
          {
            language: "python",
            code: `from supabase import create_client
from dataclasses import dataclass
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import os

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

@dataclass
class RecipientProfile:
    email: str
    first_name: str
    segment: str
    # Comportement d'achat
    total_purchases: int
    avg_order_value: float
    last_purchase_days_ago: int
    favorite_categories: list[str]
    # Comportement email
    avg_open_rate: float
    avg_click_rate: float
    preferred_send_time: str  # "morning", "afternoon", "evening"
    last_open_days_ago: int
    # Engagement
    engagement_score: float  # 0-100
    churn_risk: str  # "low", "medium", "high"
    # Préférences
    preferred_tone: str  # "formal", "casual", "enthusiastic"
    interests: list[str]
    lifecycle_stage: str  # "new", "active", "at_risk", "dormant", "vip"

class RecipientSegmenter:
    def __init__(self):
        self.cluster_model = None

    def build_profile(self, email: str) -> RecipientProfile:
        """Construit le profil complet d'un destinataire."""
        # Données CRM
        contact = supabase.table("contacts").select("*").eq("email", email).single().execute().data
        # Historique d'achats
        orders = supabase.table("orders").select("*").eq("customer_email", email).order(
            "created_at", desc=True
        ).limit(50).execute().data
        # Historique email
        email_events = supabase.table("email_events").select("*").eq("recipient", email).order(
            "sent_at", desc=True
        ).limit(100).execute().data
        # Calcul des métriques
        total_purchases = len(orders)
        avg_order = np.mean([o["total"] for o in orders]) if orders else 0
        from datetime import datetime, timedelta
        last_purchase = (datetime.utcnow() - datetime.fromisoformat(
            orders[0]["created_at"]
        )).days if orders else 999

        # Taux d'ouverture et de clic
        opens = sum(1 for e in email_events if e.get("opened"))
        clicks = sum(1 for e in email_events if e.get("clicked"))
        total_sent = max(len(email_events), 1)

        # Catégories favorites
        categories = [item["category"] for o in orders for item in o.get("items", [])]
        from collections import Counter
        fav_cats = [c for c, _ in Counter(categories).most_common(3)]

        # Déterminer l'heure préférée d'ouverture
        open_hours = [datetime.fromisoformat(e["opened_at"]).hour
                      for e in email_events if e.get("opened_at")]
        preferred_time = "morning" if np.median(open_hours or [10]) < 12 else (
            "afternoon" if np.median(open_hours or [14]) < 17 else "evening"
        )

        # Score d'engagement (RFM simplifié)
        recency = max(0, 100 - last_purchase * 2)
        frequency = min(total_purchases * 10, 100)
        monetary = min(avg_order / 5, 100)
        engagement = (recency * 0.4 + frequency * 0.3 + monetary * 0.3)

        # Lifecycle stage
        if total_purchases == 0:
            stage = "new"
        elif last_purchase > 180:
            stage = "dormant"
        elif last_purchase > 60:
            stage = "at_risk"
        elif total_purchases > 10 and avg_order > 100:
            stage = "vip"
        else:
            stage = "active"

        return RecipientProfile(
            email=email,
            first_name=contact.get("first_name", ""),
            segment=self._determine_segment(engagement, stage),
            total_purchases=total_purchases,
            avg_order_value=round(avg_order, 2),
            last_purchase_days_ago=last_purchase,
            favorite_categories=fav_cats,
            avg_open_rate=round(opens / total_sent, 3),
            avg_click_rate=round(clicks / total_sent, 3),
            preferred_send_time=preferred_time,
            last_open_days_ago=0,
            engagement_score=round(engagement, 1),
            churn_risk="high" if stage == "at_risk" else ("medium" if stage == "dormant" else "low"),
            preferred_tone="formal" if avg_order > 200 else "casual",
            interests=fav_cats,
            lifecycle_stage=stage,
        )

    def _determine_segment(self, engagement: float, stage: str) -> str:
        if stage == "vip":
            return "vip_fidele"
        elif stage == "new":
            return "nouveau_client"
        elif engagement > 70:
            return "engage_actif"
        elif stage == "at_risk":
            return "risque_churn"
        elif stage == "dormant":
            return "dormant_reactiver"
        else:
            return "standard"`,
            filename: "segmenter.py",
          },
        ],
      },
      {
        title: "Moteur de recommandation produits contextualisé",
        content:
          "Implémentez le moteur de recommandation qui sélectionne les produits les plus pertinents pour chaque destinataire. Il combine le filtrage collaboratif (clients similaires) et le filtrage basé sur le contenu (préférences déclarées et historique) pour proposer 3-5 produits par email.",
        codeSnippets: [
          {
            language: "python",
            code: `from anthropic import Anthropic
import json

client = Anthropic()

RECOMMENDATION_PROMPT = """Tu es un expert en merchandising e-commerce pour le marché français.

Sélectionne les 4 produits les plus pertinents pour ce client parmi le catalogue.

## Profil client:
- Prénom: {first_name}
- Segment: {segment}
- Catégories favorites: {favorite_categories}
- Panier moyen: {avg_order_value} EUR
- Dernier achat il y a: {last_purchase_days} jours
- Score engagement: {engagement}/100
- Stade lifecycle: {lifecycle_stage}

## Historique achats récents:
{recent_purchases}

## Produits disponibles (catalogue):
{catalog_excerpt}

## Consignes:
1. Sélectionne 4 produits cohérents avec le profil et l'historique
2. Inclus au moins 1 produit de cross-sell (catégorie complémentaire)
3. Respecte la gamme de prix habituelle du client (+/- 30%)
4. Ne recommande JAMAIS un produit déjà acheté
5. Pour les clients "at_risk", privilégie les best-sellers ou promotions
6. Pour les VIP, privilégie les nouveautés et éditions limitées

Retourne un JSON: products (array de {{sku, name, price, reason}})"""

class ProductRecommender:
    def __init__(self):
        pass

    def get_recommendations(self, profile: dict, recent_orders: list,
                            catalog: list) -> list[dict]:
        """Génère des recommandations personnalisées."""
        # Filtrer le catalogue pour exclure les produits déjà achetés
        purchased_skus = set()
        for order in recent_orders:
            for item in order.get("items", []):
                purchased_skus.add(item.get("sku"))

        available = [p for p in catalog if p["sku"] not in purchased_skus]

        # Pré-filtrer par gamme de prix compatible
        price_range = (profile["avg_order_value"] * 0.3, profile["avg_order_value"] * 2)
        price_filtered = [p for p in available
                          if price_range[0] <= p["price"] <= price_range[1]]

        # Si pas assez de produits dans la gamme, élargir
        candidates = price_filtered if len(price_filtered) >= 20 else available

        # Limiter à 30 candidats pour le prompt LLM
        candidates = candidates[:30]

        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": RECOMMENDATION_PROMPT.format(
                    first_name=profile.get("first_name", ""),
                    segment=profile.get("segment", ""),
                    favorite_categories=", ".join(profile.get("favorite_categories", [])),
                    avg_order_value=profile.get("avg_order_value", 0),
                    last_purchase_days=profile.get("last_purchase_days_ago", 0),
                    engagement=profile.get("engagement_score", 0),
                    lifecycle_stage=profile.get("lifecycle_stage", ""),
                    recent_purchases=json.dumps(recent_orders[:5], ensure_ascii=False, default=str),
                    catalog_excerpt=json.dumps(
                        [{"sku": p["sku"], "name": p["name"], "price": p["price"],
                          "category": p["category"]} for p in candidates],
                        ensure_ascii=False
                    ),
                ),
            }],
        )
        return json.loads(response.content[0].text)["products"]`,
            filename: "recommender.py",
          },
        ],
      },
      {
        title: "Génération du contenu email personnalisé",
        content:
          "Le coeur de l'agent : le LLM rédige un email complet et unique pour chaque destinataire, adapté à son profil, son historique et les recommandations produits. L'email respecte la charte éditoriale et s'adapte au ton préféré du destinataire. Un système de cache par micro-segment réduit les coûts.",
        codeSnippets: [
          {
            language: "python",
            code: `import hashlib
import json

EMAIL_GENERATION_PROMPT = """Tu es un rédacteur email marketing expert pour une marque e-commerce française.

Rédige un email marketing personnalisé pour ce destinataire.

## Destinataire:
- Prénom: {first_name}
- Segment: {segment}
- Lifecycle: {lifecycle_stage}
- Engagement: {engagement_score}/100
- Ton préféré: {preferred_tone}

## Produits à mettre en avant:
{products_json}

## Objectif de la campagne: {campaign_objective}
## Charte éditoriale: {brand_guidelines}

## Consignes:
1. Objet email: max 50 caractères, personnalisé, incitant à l'ouverture
2. Pré-header: 80-100 caractères complémentant l'objet
3. Introduction: 1-2 phrases personnalisées selon le lifecycle stage
4. Corps: présente les produits recommandés avec bénéfices (pas juste features)
5. CTA principal: un seul call-to-action clair et urgent
6. Ton {preferred_tone}: adapte le niveau de langage
7. Pour les VIP: ton exclusif, accès privilégié
8. Pour les at_risk: offre de rétention, rappel de la valeur
9. Pour les nouveaux: message de bienvenue, guide d'achat
10. Utilise le vouvoiement sauf si le ton est "casual"

Retourne un JSON: subject, preheader, html_body, plain_text, cta_text, cta_url"""

class EmailContentGenerator:
    def __init__(self):
        self.cache = {}

    def generate_email(self, profile: dict, products: list[dict],
                       campaign: dict) -> dict:
        """Génère un email personnalisé pour un destinataire."""
        # Vérifier le cache par micro-segment + produits
        cache_key = self._compute_cache_key(profile, products, campaign)
        if cache_key in self.cache:
            # Personnaliser seulement le prénom sur le template caché
            cached = self.cache[cache_key].copy()
            cached["subject"] = cached["subject"].replace("[PRENOM]", profile.get("first_name", ""))
            cached["html_body"] = cached["html_body"].replace("[PRENOM]", profile.get("first_name", ""))
            return cached

        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": EMAIL_GENERATION_PROMPT.format(
                    first_name=profile.get("first_name", "Cher client"),
                    segment=profile.get("segment", "standard"),
                    lifecycle_stage=profile.get("lifecycle_stage", "active"),
                    engagement_score=profile.get("engagement_score", 50),
                    preferred_tone=profile.get("preferred_tone", "casual"),
                    products_json=json.dumps(products, ensure_ascii=False),
                    campaign_objective=campaign.get("objective", ""),
                    brand_guidelines=campaign.get("brand_guidelines", ""),
                ),
            }],
        )
        email_content = json.loads(response.content[0].text)

        # Mettre en cache avec le prénom générique
        cache_version = email_content.copy()
        cache_version["subject"] = cache_version["subject"].replace(
            profile.get("first_name", ""), "[PRENOM]"
        )
        cache_version["html_body"] = cache_version["html_body"].replace(
            profile.get("first_name", ""), "[PRENOM]"
        )
        self.cache[cache_key] = cache_version

        return email_content

    def _compute_cache_key(self, profile, products, campaign):
        """Clé de cache basée sur segment + produits + campagne."""
        key_data = {
            "segment": profile.get("segment"),
            "lifecycle": profile.get("lifecycle_stage"),
            "tone": profile.get("preferred_tone"),
            "products": [p.get("sku") for p in products],
            "campaign_id": campaign.get("id"),
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()`,
            filename: "email_generator.py",
          },
        ],
      },
      {
        title: "Optimisation du timing d'envoi et pipeline de distribution",
        content:
          "Chaque email est envoyé au moment optimal pour chaque destinataire, calculé à partir de ses habitudes d'ouverture historiques. Le pipeline de distribution gère les batches, respecte les limites de taux, et assure le suivi de délivrabilité.",
        codeSnippets: [
          {
            language: "python",
            code: `import resend
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

resend.api_key = os.getenv("RESEND_API_KEY")

class SendTimeOptimizer:
    def __init__(self):
        pass

    def get_optimal_send_time(self, profile: dict) -> datetime:
        """Calcule l'heure d'envoi optimale pour un destinataire."""
        # Récupérer les heures d'ouverture historiques
        events = supabase.table("email_events").select("opened_at").eq(
            "recipient", profile["email"]
        ).not_.is_("opened_at", "null").limit(50).execute()

        if not events.data:
            # Défaut selon le segment
            defaults = {
                "vip_fidele": 9,
                "engage_actif": 10,
                "nouveau_client": 11,
                "risque_churn": 8,
                "dormant_reactiver": 12,
                "standard": 10,
            }
            hour = defaults.get(profile.get("segment", "standard"), 10)
            return datetime.utcnow().replace(hour=hour, minute=0)

        # Calculer l'heure médiane d'ouverture
        hours = [datetime.fromisoformat(e["opened_at"]).hour for e in events.data]
        optimal_hour = int(np.median(hours))
        optimal_minute = int(np.mean(
            [datetime.fromisoformat(e["opened_at"]).minute for e in events.data]
        ))

        # Programmer pour demain à l'heure optimale si déjà passé
        now = datetime.utcnow()
        send_time = now.replace(hour=optimal_hour, minute=optimal_minute, second=0)
        if send_time <= now:
            send_time += timedelta(days=1)

        return send_time

class CampaignDistributor:
    def __init__(self):
        self.time_optimizer = SendTimeOptimizer()
        self.email_generator = EmailContentGenerator()
        self.recommender = ProductRecommender()
        self.segmenter = RecipientSegmenter()

    async def distribute_campaign(self, campaign: dict, recipient_emails: list[str]):
        """Distribue une campagne personnalisée à tous les destinataires."""
        # Charger le catalogue
        catalog = supabase.table("products").select("*").eq("active", True).execute().data

        send_queue = defaultdict(list)  # {datetime_bucket: [emails]}
        results = {"generated": 0, "scheduled": 0, "errors": 0}

        for email in recipient_emails:
            try:
                # 1. Profil destinataire
                profile = self.segmenter.build_profile(email)

                # 2. Recommandations produits
                recent_orders = supabase.table("orders").select("*").eq(
                    "customer_email", email
                ).order("created_at", desc=True).limit(5).execute().data
                products = self.recommender.get_recommendations(
                    profile.__dict__, recent_orders, catalog
                )

                # 3. Générer l'email
                email_content = self.email_generator.generate_email(
                    profile.__dict__, products, campaign
                )

                # 4. Calculer le timing optimal
                send_time = self.time_optimizer.get_optimal_send_time(profile.__dict__)

                # 5. Mettre en file d'envoi
                supabase.table("email_queue").insert({
                    "campaign_id": campaign["id"],
                    "recipient": email,
                    "subject": email_content["subject"],
                    "html_body": email_content["html_body"],
                    "plain_text": email_content["plain_text"],
                    "scheduled_at": send_time.isoformat(),
                    "status": "queued",
                    "profile_segment": profile.segment,
                }).execute()

                results["generated"] += 1
                results["scheduled"] += 1
            except Exception as e:
                results["errors"] += 1
                print(f"Erreur pour {email}: {e}")

        return results

    def process_send_queue(self):
        """Traite la file d'envoi (appelé par cron toutes les 5 min)."""
        now = datetime.utcnow().isoformat()
        pending = supabase.table("email_queue").select("*").eq(
            "status", "queued"
        ).lte("scheduled_at", now).limit(100).execute()

        for entry in pending.data:
            try:
                result = resend.Emails.send({
                    "from": "marketing@votreentreprise.fr",
                    "to": entry["recipient"],
                    "subject": entry["subject"],
                    "html": entry["html_body"],
                    "text": entry["plain_text"],
                })
                supabase.table("email_queue").update({
                    "status": "sent",
                    "sent_at": datetime.utcnow().isoformat(),
                    "provider_id": result.get("id"),
                }).eq("id", entry["id"]).execute()
            except Exception as e:
                supabase.table("email_queue").update({
                    "status": "error",
                    "error": str(e),
                }).eq("id", entry["id"]).execute()`,
            filename: "distributor.py",
          },
        ],
      },
      {
        title: "Tracking, analytics et optimisation continue",
        content:
          "Mettez en place le suivi complet des performances : taux d'ouverture, taux de clic, conversions et revenus attribués. Le système apprend des résultats pour améliorer en continu les recommandations, le contenu et le timing. Un rapport automatique mesure le ROI de la personnalisation par rapport aux campagnes classiques.",
        codeSnippets: [
          {
            language: "python",
            code: `from datetime import datetime, timedelta

class CampaignAnalytics:
    def __init__(self):
        pass

    def process_webhook_event(self, event: dict):
        """Traite les webhooks Resend (ouverture, clic, bounce, etc.)."""
        event_type = event.get("type")
        email_id = event.get("email_id")

        # Récupérer l'entrée de la file d'envoi
        queue_entry = supabase.table("email_queue").select("*").eq(
            "provider_id", email_id
        ).single().execute()

        if not queue_entry.data:
            return

        entry = queue_entry.data
        now = datetime.utcnow().isoformat()

        if event_type == "email.opened":
            supabase.table("email_events").insert({
                "campaign_id": entry["campaign_id"],
                "recipient": entry["recipient"],
                "event_type": "open",
                "opened_at": now,
                "sent_at": entry["sent_at"],
            }).execute()

        elif event_type == "email.clicked":
            supabase.table("email_events").insert({
                "campaign_id": entry["campaign_id"],
                "recipient": entry["recipient"],
                "event_type": "click",
                "clicked_at": now,
                "clicked_url": event.get("url", ""),
            }).execute()

        elif event_type == "email.bounced":
            supabase.table("email_events").insert({
                "campaign_id": entry["campaign_id"],
                "recipient": entry["recipient"],
                "event_type": "bounce",
                "bounce_type": event.get("bounce_type", "hard"),
            }).execute()
            # Désactiver le contact en cas de hard bounce
            if event.get("bounce_type") == "hard":
                supabase.table("contacts").update(
                    {"email_active": False}
                ).eq("email", entry["recipient"]).execute()

    def generate_campaign_report(self, campaign_id: str) -> dict:
        """Génère le rapport de performance d'une campagne."""
        # Métriques d'envoi
        sent = supabase.table("email_queue").select("*", count="exact").eq(
            "campaign_id", campaign_id
        ).eq("status", "sent").execute()

        # Métriques d'engagement
        events = supabase.table("email_events").select("*").eq(
            "campaign_id", campaign_id
        ).execute()

        opens = sum(1 for e in events.data if e["event_type"] == "open")
        clicks = sum(1 for e in events.data if e["event_type"] == "click")
        bounces = sum(1 for e in events.data if e["event_type"] == "bounce")
        unsubscribes = sum(1 for e in events.data if e["event_type"] == "unsubscribe")

        total_sent = sent.count or 1

        # Conversions attribuées (dans les 7 jours post-clic)
        conversions = supabase.rpc("count_attributed_conversions", {
            "p_campaign_id": campaign_id,
            "p_attribution_window_days": 7,
        }).execute()

        # Métriques par segment
        segment_metrics = {}
        queue_entries = supabase.table("email_queue").select("recipient, profile_segment").eq(
            "campaign_id", campaign_id
        ).execute()

        for entry in queue_entries.data:
            seg = entry["profile_segment"]
            if seg not in segment_metrics:
                segment_metrics[seg] = {"sent": 0, "opens": 0, "clicks": 0}
            segment_metrics[seg]["sent"] += 1

        report = {
            "campaign_id": campaign_id,
            "total_sent": total_sent,
            "open_rate": round(opens / total_sent * 100, 2),
            "click_rate": round(clicks / total_sent * 100, 2),
            "bounce_rate": round(bounces / total_sent * 100, 2),
            "unsubscribe_rate": round(unsubscribes / total_sent * 100, 3),
            "conversions": conversions.data if conversions.data else 0,
            "segment_performance": segment_metrics,
            "generated_at": datetime.utcnow().isoformat(),
        }

        # Sauvegarder le rapport
        supabase.table("campaign_reports").insert(report).execute()
        return report`,
            filename: "analytics.py",
          },
        ],
      },
    ],
    enterprise: {
      piiHandling: "Les données de profilage comportemental (achats, navigation, préférences) sont des données personnelles soumises au RGPD. Base légale : consentement explicite pour le profilage marketing (art. 6.1.a) ou intérêt légitime avec opt-out facile (art. 6.1.f). Les profils ne sont jamais envoyés bruts au LLM : seules les métriques agrégées et anonymisées (segment, score, catégories) sont transmises. Lien de désinscription obligatoire dans chaque email. Respect strict des préférences de fréquence.",
      auditLog: "Chaque email généré est loggué avec : horodatage de génération, profil utilisé (version agrégée), produits recommandés, prompt LLM complet, contenu généré, heure d'envoi programmée, métriques de performance post-envoi. Rétention 24 mois. Export automatique pour audit CNIL si requis. Traçabilité complète du consentement marketing.",
      humanInTheLoop: "Les emails générés pour les segments VIP (> 10K EUR de CA annuel) sont systématiquement revus par le responsable CRM avant envoi. Les campagnes dépassant 50K destinataires nécessitent une validation du directeur marketing. Un échantillon aléatoire de 2% des emails est relu par l'équipe éditoriale pour contrôle qualité.",
      monitoring: "Dashboard Langfuse et Supabase : taux d'ouverture par segment et par campagne, taux de clic, taux de conversion attribué, revenu par email envoyé, coût LLM par email, taux de désabonnement, score de délivrabilité (réputation IP), temps de génération par email. Alertes si le taux de bounce dépasse 2%, si le taux de désabonnement dépasse 0.5%, ou si le coût LLM par email dépasse 0.05 EUR.",
    },
    n8nWorkflow: {
      description: "Workflow n8n : Webhook (déclenchement campagne) -> Node Supabase (récupération liste destinataires) -> Node Loop (pour chaque destinataire) -> Node Supabase (profil comportemental) -> Node HTTP Request (Claude - recommandations produits) -> Node HTTP Request (Claude - génération email) -> Node Code (calcul timing optimal) -> Node Supabase (mise en file d'envoi) -> Cron (toutes les 5 min) : Node Supabase (emails à envoyer maintenant) -> Node HTTP Request (Resend - envoi) -> Node Supabase (mise à jour statut). Webhook tracking : Node Webhook (événements Resend) -> Node Supabase (log événement) -> Node Code (mise à jour métriques).",
      nodes: ["Webhook (lancement campagne)", "Supabase (destinataires)", "Loop (chaque contact)", "Supabase (profil)", "HTTP Request (Claude recommandations)", "HTTP Request (Claude email)", "Code (timing optimal)", "Supabase (file envoi)", "Cron (5 min)", "HTTP Request (Resend)", "Webhook (tracking events)", "Supabase (analytics)"],
      triggerType: "Webhook (déclenchement campagne par le responsable marketing) + Cron (traitement file d'envoi toutes les 5 minutes)",
    },
    estimatedTime: "8-12h",
    difficulty: "Moyen",
    sectors: ["E-commerce", "Retail", "SaaS", "Services"],
    metiers: ["Marketing Digital", "CRM", "Growth"],
    functions: ["Marketing", "CRM"],
    metaTitle: "Agent IA de Personnalisation Email Marketing — Guide Complet",
    metaDescription:
      "Générez des emails marketing hyper-personnalisés à grande échelle avec un agent IA. Segmentation comportementale, recommandations produits et timing d'envoi optimal. Tutoriel complet.",
    createdAt: "2025-02-07",
    updatedAt: "2025-02-07",
  },

];
