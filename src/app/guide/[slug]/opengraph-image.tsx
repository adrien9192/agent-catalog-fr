import { ImageResponse } from "next/og";
import { guides } from "@/data/guides";

export const alt = "AgentCatalog — Guide";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export function generateStaticParams() {
  return guides.map((g) => ({ slug: g.slug }));
}

export default async function OgImage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const guide = guides.find((g) => g.slug === slug);

  const title = guide?.title ?? "Guide pratique";
  const category = guide?.category ?? "IA";
  const readTime = guide?.readTime ?? "";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(135deg, #059669, #10b981, #0d9488)",
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
            AgentCatalog — Guide
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
              {category}
            </span>
            {readTime && (
              <span
                style={{
                  padding: "4px 14px",
                  borderRadius: "99px",
                  fontSize: "16px",
                  color: "#fff",
                  border: "1px solid rgba(255,255,255,0.4)",
                }}
              >
                {readTime} de lecture
              </span>
            )}
          </div>

          <h1
            style={{
              color: "#fff",
              fontSize: "48px",
              fontWeight: 800,
              lineHeight: 1.2,
              margin: 0,
              maxWidth: "900px",
            }}
          >
            {title}
          </h1>

          <p
            style={{
              color: "rgba(255,255,255,0.75)",
              fontSize: "22px",
              marginTop: "16px",
              maxWidth: "800px",
            }}
          >
            Guide pratique — Tutoriels et bonnes pratiques IA pour l&apos;entreprise
          </p>
        </div>
      </div>
    ),
    { ...size }
  );
}
