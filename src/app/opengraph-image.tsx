import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "AgentCatalog — Cas d'usage d'Agents IA en entreprise";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OgImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background: "linear-gradient(135deg, #6d28d9, #7c3aed, #4f46e5)",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            marginBottom: "32px",
          }}
        >
          <div
            style={{
              width: "64px",
              height: "64px",
              borderRadius: "16px",
              background: "rgba(255,255,255,0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontSize: "28px",
              fontWeight: 700,
            }}
          >
            AC
          </div>
          <span style={{ color: "#fff", fontSize: "36px", fontWeight: 700 }}>
            AgentCatalog
          </span>
        </div>
        <h1
          style={{
            color: "#fff",
            fontSize: "52px",
            fontWeight: 800,
            textAlign: "center",
            lineHeight: 1.2,
            maxWidth: "900px",
            margin: 0,
          }}
        >
          Déployez un Agent IA en quelques heures
        </h1>
        <p
          style={{
            color: "rgba(255,255,255,0.8)",
            fontSize: "24px",
            textAlign: "center",
            maxWidth: "700px",
            marginTop: "16px",
          }}
        >
          50 workflows IA documentés avec tutoriel, stack et ROI. Gratuit.
        </p>
      </div>
    ),
    { ...size }
  );
}
