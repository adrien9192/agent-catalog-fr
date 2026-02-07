import { ImageResponse } from "next/og";
import { guides } from "@/data/guides";

export const alt = "AgentCatalog — Guides pratiques IA";
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
          background: "linear-gradient(135deg, #7c3aed, #8b5cf6, #a78bfa)",
          fontFamily: "system-ui, sans-serif",
          padding: "60px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "12px",
            marginBottom: "auto",
          }}
        >
          <div
            style={{
              width: "40px",
              height: "40px",
              borderRadius: "10px",
              background: "rgba(255,255,255,0.2)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              fontSize: "16px",
              fontWeight: 700,
            }}
          >
            AC
          </div>
          <span style={{ color: "rgba(255,255,255,0.8)", fontSize: "20px", fontWeight: 600 }}>
            AgentCatalog — Ressources gratuites
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", marginTop: "auto" }}>
          <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
            <span
              style={{
                padding: "4px 14px",
                borderRadius: "99px",
                fontSize: "16px",
                fontWeight: 600,
                color: "#fff",
                background: "rgba(255,255,255,0.2)",
              }}
            >
              {guides.length} guides
            </span>
          </div>

          <h1
            style={{
              color: "#fff",
              fontSize: "52px",
              fontWeight: 800,
              lineHeight: 1.2,
              margin: 0,
              maxWidth: "900px",
            }}
          >
            Guides pratiques IA pour l&apos;entreprise
          </h1>

          <p
            style={{
              color: "rgba(255,255,255,0.75)",
              fontSize: "22px",
              marginTop: "16px",
              maxWidth: "800px",
            }}
          >
            Stack technique, ROI, conformite RGPD et implementation
          </p>
        </div>
      </div>
    ),
    { ...size }
  );
}
