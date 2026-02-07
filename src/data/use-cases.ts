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
];
