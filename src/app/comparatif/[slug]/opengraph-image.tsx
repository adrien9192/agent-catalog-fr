import { ImageResponse } from "next/og";
import { comparisons } from "@/data/comparisons";

export const alt = "AgentCatalog — Comparatif";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export function generateStaticParams() {
  return comparisons.map((c) => ({ slug: c.slug }));
}

export default async function OgImage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const comparison = comparisons.find((c) => c.slug === slug);

  const title = comparison?.title ?? "Comparatif";
  const options = comparison?.options.map((o) => o.name) ?? [];

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(135deg, #6d28d9, #7c3aed, #8b5cf6)",
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
            AgentCatalog — Comparatif
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", marginTop: "auto" }}>
          <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
            {options.map((option) => (
              <span
                key={option}
                style={{
                  padding: "4px 14px",
                  borderRadius: "99px",
                  fontSize: "16px",
                  fontWeight: 600,
                  color: "#fff",
                  background: "rgba(255,255,255,0.2)",
                }}
              >
                {option}
              </span>
            ))}
          </div>

          <h1
            style={{
              color: "#fff",
              fontSize: "44px",
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
            Comparatif objectif — Prix, fonctionnalités, cas d&apos;usage
          </p>
        </div>
      </div>
    ),
    { ...size }
  );
}
