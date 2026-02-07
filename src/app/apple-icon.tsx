import { ImageResponse } from "next/og";

export const size = { width: 180, height: 180 };
export const contentType = "image/png";

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "linear-gradient(135deg, #6d28d9, #8b5cf6)",
          borderRadius: "36px",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <span style={{ color: "#fff", fontSize: "100px", fontWeight: 800 }}>A</span>
      </div>
    ),
    { ...size }
  );
}
