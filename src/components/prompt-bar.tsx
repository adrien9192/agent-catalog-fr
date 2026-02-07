"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export function PromptBar() {
  const [query, setQuery] = useState("");
  const router = useRouter();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      router.push(`/catalogue?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="w-full max-w-2xl mx-auto">
      <div
        className="flex items-center gap-3 rounded-2xl px-4 py-3 sm:px-5 sm:py-4 shadow-lg border"
        style={{
          backgroundColor: "var(--prompt-bar-bg)",
          borderColor: "var(--prompt-bar-border)",
        }}
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className="shrink-0 opacity-50"
          style={{ color: "var(--prompt-bar-text)" }}
        >
          <circle cx="11" cy="11" r="8" />
          <path d="m21 21-4.3-4.3" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Quel processus voulez-vous automatiser ? ex: triage support, qualification leads..."
          className="flex-1 bg-transparent text-base outline-none placeholder:opacity-40"
          style={{ color: "var(--prompt-bar-text)" }}
        />
        <button
          type="submit"
          className="shrink-0 rounded-xl bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-opacity hover:opacity-90"
        >
          Trouver mon workflow
        </button>
      </div>
    </form>
  );
}
