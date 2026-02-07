"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";

interface StickyCTABarProps {
  title: string;
  difficulty: string;
}

export function StickyCTABar({ title, difficulty }: StickyCTABarProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setVisible(window.scrollY > 400);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  if (!visible) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur-md shadow-lg animate-in slide-in-from-bottom duration-300">
      <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">
            {difficulty} &middot; Tutoriel complet inclus
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button size="sm" variant="outline" asChild>
            <Link href="/demande">Sur mesure</Link>
          </Button>
          <Button size="sm" asChild>
            <Link href="/pricing">Voir les plans</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
