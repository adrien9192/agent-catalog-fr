import { NextRequest, NextResponse } from "next/server";
import { useCases } from "@/data/use-cases";

const BREVO_API_KEY = process.env.BREVO_API_KEY || "";
const BREVO_LIST_ID = parseInt(process.env.BREVO_LIST_ID || "3", 10);
const BREVO_SENDER_EMAIL = process.env.BREVO_SENDER_EMAIL || "adrienlaine91@gmail.com";
const BREVO_SENDER_NAME = process.env.BREVO_SENDER_NAME || "AgentCatalog";
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://agent-catalog-fr.vercel.app";

interface SubscribeBody {
  email: string;
  sectors?: string[];
  functions?: string[];
}

function buildWelcomeHtml(): string {
  const topUseCases = useCases.slice(0, 3);
  const ucListHtml = topUseCases
    .map(
      (uc) =>
        `<li style="margin-bottom:12px;">
          <a href="${SITE_URL}/use-case/${uc.slug}" style="color:#6d28d9;font-weight:600;text-decoration:none;">${uc.title}</a>
          <br/><span style="color:#6b7280;font-size:13px;">${uc.subtitle}</span>
        </li>`
    )
    .join("");

  return `<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f8f9fa;font-family:system-ui,-apple-system,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:20px;">
  <div style="background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
    <div style="background:linear-gradient(135deg,#6d28d9,#7c3aed);padding:28px 24px;color:#fff;">
      <h1 style="margin:0;font-size:24px;line-height:1.3;">Bienvenue sur AgentCatalog !</h1>
      <p style="margin:8px 0 0;font-size:14px;opacity:0.9;">Votre dose quotidienne de cas d'usage d'Agents IA</p>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 16px;color:#374151;line-height:1.6;font-size:15px;">
        Merci pour votre inscription ! Chaque matin, vous recevrez un nouveau cas d'usage d'Agent IA
        avec tutoriel complet, stack recommandée et estimation de ROI.
      </p>
      <h3 style="margin:20px 0 12px;font-size:16px;color:#111;">Pour commencer, voici nos 3 cas les plus populaires :</h3>
      <ul style="padding-left:20px;color:#374151;line-height:1.6;">
        ${ucListHtml}
      </ul>
      <div style="margin:28px 0 0;text-align:center;">
        <a href="${SITE_URL}/catalogue" style="display:inline-block;background:#6d28d9;color:#fff;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;">
          Explorer le catalogue complet
        </a>
      </div>
    </div>
    <div style="padding:16px 20px;background:#f9fafb;border-top:1px solid #e5e7eb;text-align:center;">
      <p style="margin:0;color:#9ca3af;font-size:12px;">
        AgentCatalog — Cas d'usage d'Agents IA en entreprise
      </p>
      <p style="margin:6px 0 0;color:#9ca3af;font-size:11px;">
        Vous recevez cet email car vous vous êtes inscrit(e) sur AgentCatalog.
        <a href="{{ unsubscribe }}" style="color:#6d28d9;text-decoration:underline;">Se désinscrire</a>
      </p>
    </div>
  </div>
</div>
</body>
</html>`;
}

export async function POST(req: NextRequest) {
  try {
    const body: SubscribeBody = await req.json();
    const { email, sectors, functions } = body;

    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      return NextResponse.json(
        { error: "Adresse email invalide." },
        { status: 400 }
      );
    }

    if (!BREVO_API_KEY) {
      console.error("BREVO_API_KEY not configured");
      return NextResponse.json(
        { error: "Service temporairement indisponible." },
        { status: 503 }
      );
    }

    // 1. Upsert contact in Brevo
    const brevoRes = await fetch("https://api.brevo.com/v3/contacts", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        email,
        listIds: [BREVO_LIST_ID],
        updateEnabled: true,
        attributes: {
          SOURCE: "agentcatalog_website",
          SECTORS: sectors?.join(", ") || "",
          FUNCTIONS: functions?.join(", ") || "",
          OPT_IN_DATE: new Date().toISOString().split("T")[0],
        },
      }),
    });

    if (!brevoRes.ok) {
      const errData = await brevoRes.json().catch(() => ({}));
      // "duplicate_parameter" means contact already exists — treat as success but skip welcome email
      if (
        brevoRes.status === 400 &&
        (errData as { code?: string }).code === "duplicate_parameter"
      ) {
        return NextResponse.json({
          success: true,
          message: "Vous êtes déjà inscrit(e) ! Merci.",
        });
      }
      console.error("Brevo API error:", brevoRes.status, errData);
      return NextResponse.json(
        { error: "Erreur lors de l'inscription. Réessayez." },
        { status: 500 }
      );
    }

    // 2. Send welcome transactional email immediately
    const welcomeRes = await fetch("https://api.brevo.com/v3/smtp/email", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        sender: { name: BREVO_SENDER_NAME, email: BREVO_SENDER_EMAIL },
        to: [{ email }],
        subject: "Bienvenue sur AgentCatalog — votre 1er cas d'usage vous attend",
        htmlContent: buildWelcomeHtml(),
        tags: ["welcome"],
      }),
    });

    if (!welcomeRes.ok) {
      // Log but don't fail — contact was already created successfully
      const welcomeErr = await welcomeRes.json().catch(() => ({}));
      console.error("Welcome email failed:", welcomeRes.status, welcomeErr);
    }

    return NextResponse.json({
      success: true,
      message: "Inscription réussie ! Vérifiez votre boîte mail.",
    });
  } catch (err) {
    console.error("Newsletter subscribe error:", err);
    return NextResponse.json(
      { error: "Erreur interne." },
      { status: 500 }
    );
  }
}
