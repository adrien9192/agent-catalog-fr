import { ImageResponse } from "next/og";
import { useCases } from "@/data/use-cases";

export const alt = "AgentCatalog";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function OgImage({ params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const uc = useCases.find((u) => u.slug === slug);

  const title = uc?.title ?? "Cas d'usage";
  const subtitle = uc?.subtitle ?? "";
  const difficulty = uc?.difficulty ?? "Moyen";
  const functions = uc?.functions ?? [];

  const diffColor =
    difficulty === "Facile" ? "#10b981" : difficulty === "Moyen" ? "#f59e0b" : "#ef4444";

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          background: "linear-gradient(135deg, #6d28d9, #7c3aed, #4f46e5)",
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
            AgentCatalog
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
                background: diffColor,
              }}
            >
              {difficulty}
            </span>
            {functions.map((fn) => (
              <span
                key={fn}
                style={{
                  padding: "4px 14px",
                  borderRadius: "99px",
                  fontSize: "16px",
                  color: "#fff",
                  border: "1px solid rgba(255,255,255,0.4)",
                }}
              >
                {fn}
              </span>
            ))}
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
              marginTop: "12px",
              maxWidth: "800px",
              lineHeight: 1.4,
            }}
          >
            {subtitle}
          </p>
        </div>
      </div>
    ),
    { ...size }
  );
}
