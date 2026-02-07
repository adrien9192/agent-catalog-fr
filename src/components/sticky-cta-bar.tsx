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
    <div className="fixed bottom-0 left-0 right-0 z-50 border-t bg-background/95 backdrop-blur-md shadow-lg animate-in slide-in-from-bottom duration-300 pb-[env(safe-area-inset-bottom)]">
      <div className="mx-auto flex max-w-7xl flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-4 px-4 py-3 sm:px-6">
        <div className="hidden sm:block min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{title}</p>
          <p className="text-xs text-muted-foreground">
            {difficulty} &middot; Livré en 5 jours ouvrés
          </p>
        </div>
        <div className="flex w-full sm:w-auto shrink-0 items-center gap-2">
          <Button size="sm" variant="outline" asChild className="flex-1 sm:flex-none">
            <Link href="/demande">Sur mesure</Link>
          </Button>
          <Button size="sm" asChild className="flex-1 sm:flex-none">
            <Link href="/pricing">Essai gratuit</Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
