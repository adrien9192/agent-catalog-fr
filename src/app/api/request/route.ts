import { NextRequest, NextResponse } from "next/server";

const BREVO_API_KEY = process.env.BREVO_API_KEY || "";
const BREVO_SENDER_EMAIL = process.env.BREVO_SENDER_EMAIL || "adrienlaine91@gmail.com";
const BREVO_SENDER_NAME = process.env.BREVO_SENDER_NAME || "AgentCatalog";
const ADMIN_EMAIL = "adrienlaine91@gmail.com";

interface RequestBody {
  name: string;
  email: string;
  description: string;
  searchQuery?: string;
}

function escapeHtml(text: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  };
  return text.replace(/[&<>"']/g, (m) => map[m]);
}

function buildAdminNotificationHtml(data: RequestBody): string {
  const name = escapeHtml(data.name);
  const email = escapeHtml(data.email);
  const description = escapeHtml(data.description);
  const searchQuery = data.searchQuery ? escapeHtml(data.searchQuery) : "";
  return `<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f8f9fa;font-family:system-ui,-apple-system,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:20px;">
  <div style="background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
    <div style="background:linear-gradient(135deg,#dc2626,#ef4444);padding:24px;color:#fff;">
      <h1 style="margin:0;font-size:20px;">Nouvelle demande de workflow</h1>
    </div>
    <div style="padding:24px;">
      <table style="width:100%;border-collapse:collapse;">
        <tr>
          <td style="padding:8px 12px;font-weight:600;color:#374151;border-bottom:1px solid #e5e7eb;">Nom</td>
          <td style="padding:8px 12px;color:#6b7280;border-bottom:1px solid #e5e7eb;">${name}</td>
        </tr>
        <tr>
          <td style="padding:8px 12px;font-weight:600;color:#374151;border-bottom:1px solid #e5e7eb;">Email</td>
          <td style="padding:8px 12px;color:#6b7280;border-bottom:1px solid #e5e7eb;">${email}</td>
        </tr>
        ${searchQuery ? `<tr>
          <td style="padding:8px 12px;font-weight:600;color:#374151;border-bottom:1px solid #e5e7eb;">Recherche initiale</td>
          <td style="padding:8px 12px;color:#6b7280;border-bottom:1px solid #e5e7eb;">${searchQuery}</td>
        </tr>` : ""}
        <tr>
          <td style="padding:8px 12px;font-weight:600;color:#374151;vertical-align:top;">Description</td>
          <td style="padding:8px 12px;color:#6b7280;white-space:pre-wrap;">${description}</td>
        </tr>
      </table>
    </div>
  </div>
</div>
</body>
</html>`;
}

function buildUserConfirmationHtml(rawName: string): string {
  const safeName = escapeHtml(rawName);
  return `<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"></head>
<body style="margin:0;padding:0;background:#f8f9fa;font-family:system-ui,-apple-system,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:20px;">
  <div style="background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
    <div style="background:linear-gradient(135deg,#6d28d9,#7c3aed);padding:28px 24px;color:#fff;">
      <h1 style="margin:0;font-size:22px;">Demande bien reçue !</h1>
    </div>
    <div style="padding:24px;">
      <p style="margin:0 0 16px;color:#374151;line-height:1.6;font-size:15px;">
        Bonjour ${safeName},
      </p>
      <p style="margin:0 0 16px;color:#374151;line-height:1.6;font-size:15px;">
        Nous avons bien reçu votre demande de workflow. Notre équipe va l'analyser
        et développer un cas d'usage adapté à votre besoin.
      </p>
      <p style="margin:0 0 16px;color:#374151;line-height:1.6;font-size:15px;">
        <strong>Vous recevrez un email dès que votre workflow sera en ligne</strong>,
        avec le tutoriel complet, la stack recommandée et l'estimation de ROI.
      </p>
      <p style="margin:0;color:#6b7280;font-size:14px;">
        — L'équipe AgentCatalog
      </p>
    </div>
    <div style="padding:16px 20px;background:#f9fafb;border-top:1px solid #e5e7eb;text-align:center;">
      <p style="margin:0;color:#9ca3af;font-size:12px;">
        AgentCatalog — Cas d'usage d'Agents IA en entreprise
      </p>
    </div>
  </div>
</div>
</body>
</html>`;
}

export async function POST(req: NextRequest) {
  try {
    const body: RequestBody = await req.json();
    const { name, email, description, searchQuery } = body;

    if (!name || !email || !description) {
      return NextResponse.json(
        { error: "Tous les champs sont requis." },
        { status: 400 }
      );
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
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

    // 1. Send admin notification
    const adminRes = await fetch("https://api.brevo.com/v3/smtp/email", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        sender: { name: BREVO_SENDER_NAME, email: BREVO_SENDER_EMAIL },
        to: [{ email: ADMIN_EMAIL, name: "Admin AgentCatalog" }],
        subject: `[AgentCatalog] Nouvelle demande de workflow — ${name}`,
        htmlContent: buildAdminNotificationHtml(body),
        tags: ["workflow-request"],
      }),
    });

    if (!adminRes.ok) {
      const errData = await adminRes.json().catch(() => ({}));
      console.error("Admin notification failed:", adminRes.status, errData);
    }

    // 2. Send user confirmation email
    const userRes = await fetch("https://api.brevo.com/v3/smtp/email", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        sender: { name: BREVO_SENDER_NAME, email: BREVO_SENDER_EMAIL },
        to: [{ email, name }],
        subject: "Votre demande de workflow a bien été reçue",
        htmlContent: buildUserConfirmationHtml(name),
        tags: ["workflow-request-confirmation"],
      }),
    });

    if (!userRes.ok) {
      const errData = await userRes.json().catch(() => ({}));
      console.error("User confirmation failed:", userRes.status, errData);
    }

    // 3. Also add user to Brevo contact list for future notifications
    await fetch("https://api.brevo.com/v3/contacts", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        email,
        updateEnabled: true,
        attributes: {
          FIRSTNAME: name,
          SOURCE: "workflow_request",
          REQUESTED_WORKFLOW: description.substring(0, 200),
          REQUEST_DATE: new Date().toISOString().split("T")[0],
        },
      }),
    });

    return NextResponse.json({
      success: true,
      message: "Demande envoyée ! Vous recevrez un email de confirmation.",
    });
  } catch (err) {
    console.error("Workflow request error:", err);
    return NextResponse.json(
      { error: "Erreur interne." },
      { status: 500 }
    );
  }
}
