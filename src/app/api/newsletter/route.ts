import { NextRequest, NextResponse } from "next/server";

const BREVO_API_KEY = process.env.BREVO_API_KEY || "";
const BREVO_LIST_ID = parseInt(process.env.BREVO_LIST_ID || "2", 10);

interface SubscribeBody {
  email: string;
  sectors?: string[];
  functions?: string[];
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

    // Upsert contact in Brevo
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
          PRENOM: "",
          SOURCE: "agentcatalog_website",
          SECTORS: sectors?.join(", ") || "",
          FUNCTIONS: functions?.join(", ") || "",
          OPT_IN_DATE: new Date().toISOString().split("T")[0],
        },
      }),
    });

    if (!brevoRes.ok) {
      const errData = await brevoRes.json().catch(() => ({}));
      // "duplicate parameter" means contact already exists — treat as success
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

    return NextResponse.json({
      success: true,
      message: "Inscription réussie ! Vous recevrez bientôt votre premier cas d'usage.",
    });
  } catch (err) {
    console.error("Newsletter subscribe error:", err);
    return NextResponse.json(
      { error: "Erreur interne." },
      { status: 500 }
    );
  }
}
