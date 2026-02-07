import { NextRequest, NextResponse } from "next/server";
import { useCases } from "@/data/use-cases";

const BREVO_API_KEY = process.env.BREVO_API_KEY || "";
const BREVO_LIST_ID = parseInt(process.env.BREVO_LIST_ID || "3", 10);
const BREVO_SENDER_EMAIL = process.env.BREVO_SENDER_EMAIL || "adrienlaine91@gmail.com";
const BREVO_SENDER_NAME = process.env.BREVO_SENDER_NAME || "AgentCatalog";
const CRON_SECRET = process.env.CRON_SECRET || "";
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL || "https://agent-catalog-fr.vercel.app";

/** Deterministic: day-of-year modulo total use cases → round-robin without duplicates */
function getTodayUseCaseIndex(): number {
  const now = new Date();
  const start = new Date(now.getFullYear(), 0, 0);
  const diff = now.getTime() - start.getTime();
  const dayOfYear = Math.floor(diff / (1000 * 60 * 60 * 24));
  return dayOfYear % useCases.length;
}

function buildEmailHtml(uc: (typeof useCases)[number]): string {
  const url = `${SITE_URL}/use-case/${uc.slug}`;
  const diffColor =
    uc.difficulty === "Facile" ? "#10b981" : uc.difficulty === "Moyen" ? "#f59e0b" : "#ef4444";

  return `<!DOCTYPE html>
<html lang="fr">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"></head>
<body style="margin:0;padding:0;background:#f8f9fa;font-family:system-ui,-apple-system,sans-serif;">
<div style="max-width:600px;margin:0 auto;padding:20px;">
  <div style="background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb;">
    <div style="background:linear-gradient(135deg,#6d28d9,#7c3aed);padding:24px 20px;color:#fff;">
      <p style="margin:0 0 4px;font-size:12px;opacity:0.8;">Cas d'usage du jour — AgentCatalog</p>
      <h1 style="margin:0;font-size:22px;line-height:1.3;">${uc.title}</h1>
    </div>
    <div style="padding:20px;">
      <span style="display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;font-weight:600;color:#fff;background:${diffColor};">${uc.difficulty}</span>
      <span style="display:inline-block;padding:2px 10px;border-radius:99px;font-size:12px;border:1px solid #e5e7eb;margin-left:6px;">${uc.functions.join(", ")}</span>
      <p style="margin:16px 0 0;color:#374151;line-height:1.6;">${uc.subtitle}</p>
      <h3 style="margin:20px 0 8px;font-size:15px;color:#111;">Le problème</h3>
      <p style="margin:0;color:#6b7280;font-size:14px;line-height:1.5;">${uc.problem}</p>
      <h3 style="margin:20px 0 8px;font-size:15px;color:#111;">La valeur</h3>
      <p style="margin:0;color:#6b7280;font-size:14px;line-height:1.5;">${uc.value}</p>
      <h3 style="margin:20px 0 8px;font-size:15px;color:#111;">ROI indicatif</h3>
      <p style="margin:0;color:#6d28d9;font-size:14px;font-weight:600;">${uc.roiIndicatif}</p>
      <div style="margin:24px 0 0;text-align:center;">
        <a href="${url}" style="display:inline-block;background:#6d28d9;color:#fff;padding:12px 28px;border-radius:8px;text-decoration:none;font-weight:600;font-size:14px;">
          Voir le tutoriel complet →
        </a>
      </div>
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

export async function GET(req: NextRequest) {
  // Verify cron secret (Vercel sends this header)
  const authHeader = req.headers.get("authorization");
  if (CRON_SECRET && authHeader !== `Bearer ${CRON_SECRET}`) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  if (!BREVO_API_KEY) {
    return NextResponse.json({ error: "BREVO_API_KEY not configured" }, { status: 503 });
  }

  try {
    const index = getTodayUseCaseIndex();
    const uc = useCases[index];

    // Send campaign via Brevo email campaign API
    const campaignRes = await fetch("https://api.brevo.com/v3/emailCampaigns", {
      method: "POST",
      headers: {
        "api-key": BREVO_API_KEY,
        "Content-Type": "application/json",
        Accept: "application/json",
      },
      body: JSON.stringify({
        name: `Daily UC ${new Date().toISOString().split("T")[0]} — ${uc.title}`,
        subject: `${uc.title} — Cas d'usage du jour`,
        sender: { name: BREVO_SENDER_NAME, email: BREVO_SENDER_EMAIL },
        recipients: { listIds: [BREVO_LIST_ID] },
        htmlContent: buildEmailHtml(uc),
        scheduledAt: new Date().toISOString(),
      }),
    });

    if (!campaignRes.ok) {
      const errData = await campaignRes.json().catch(() => ({}));
      console.error("Brevo campaign error:", campaignRes.status, errData);
      return NextResponse.json(
        { error: "Failed to create campaign", details: errData },
        { status: 500 }
      );
    }

    const result = await campaignRes.json();

    // Send the campaign immediately
    const campaignId = (result as { id?: number }).id;
    if (campaignId) {
      await fetch(`https://api.brevo.com/v3/emailCampaigns/${campaignId}/sendNow`, {
        method: "POST",
        headers: {
          "api-key": BREVO_API_KEY,
          "Content-Type": "application/json",
        },
      });
    }

    return NextResponse.json({
      success: true,
      useCaseIndex: index,
      useCaseTitle: uc.title,
      campaignId,
      date: new Date().toISOString(),
    });
  } catch (err) {
    console.error("Daily email cron error:", err);
    return NextResponse.json({ error: "Internal error" }, { status: 500 });
  }
}
