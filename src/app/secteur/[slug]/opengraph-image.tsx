import { ImageResponse } from "next/og";
import { sectors } from "@/data/sectors";
import { useCases } from "@/data/use-cases";

export const alt = "AgentCatalog — Secteur";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export function generateStaticParams() {
  return sectors.map((s) => ({ slug: s.slug }));
}

export default async function OgImage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const sector = sectors.find((s) => s.slug === slug);

  const name = sector?.name ?? "Secteur";
  const description = sector?.description ?? "";
  const icon = sector?.icon ?? "";
  const count = useCases.filter((uc) =>
    uc.sectors.some((s) => s.toLowerCase().replace(/\s+/g, "-") === slug || s === name)
  ).length;

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(135deg, #0369a1, #0284c7, #0ea5e9)",
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
            AgentCatalog — Secteur
          </span>
        </div>

        <div style={{ display: "flex", flexDirection: "column", marginTop: "auto" }}>
          <div style={{ display: "flex", gap: "10px", marginBottom: "20px", alignItems: "center" }}>
            <span style={{ fontSize: "48px" }}>{icon}</span>
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
              {count} workflow{count !== 1 ? "s" : ""} IA
            </span>
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
            Agents IA — {name}
          </h1>

          <p
            style={{
              color: "rgba(255,255,255,0.75)",
              fontSize: "22px",
              marginTop: "16px",
              maxWidth: "800px",
              lineHeight: 1.4,
            }}
          >
            {description.slice(0, 120)}
          </p>
        </div>
      </div>
    ),
    { ...size }
  );
}
