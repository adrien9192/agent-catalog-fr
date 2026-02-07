"use client";

import { useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader } from "@/components/ui/card";

type Scenario = "support" | "sales" | "rh" | "finance";

const scenarios: Record<
  Scenario,
  {
    label: string;
    icon: string;
    description: string;
    fields: { key: string; label: string; unit: string; defaultValue: number; min: number; max: number; step: number }[];
    calculate: (values: Record<string, number>) => { timeSaved: number; costSaved: number; payback: string };
  }
> = {
  support: {
    label: "Support client",
    icon: "🎧",
    description: "Automatisation du triage et réponse aux tickets",
    fields: [
      { key: "tickets", label: "Tickets par mois", unit: "tickets", defaultValue: 500, min: 50, max: 5000, step: 50 },
      { key: "avgTime", label: "Temps moyen par ticket", unit: "min", defaultValue: 15, min: 5, max: 60, step: 5 },
      { key: "hourlyRate", label: "Coût horaire agent", unit: "€/h", defaultValue: 35, min: 15, max: 80, step: 5 },
      { key: "automationRate", label: "Taux d'automatisation estimé", unit: "%", defaultValue: 40, min: 10, max: 70, step: 5 },
    ],
    calculate: (v) => {
      const totalHours = (v.tickets * v.avgTime) / 60;
      const automatedHours = totalHours * (v.automationRate / 100);
      const monthlySaving = automatedHours * v.hourlyRate;
      const annualSaving = monthlySaving * 12;
      const toolCost = 300; // monthly cost estimate
      const paybackMonths = Math.ceil(toolCost / monthlySaving);
      return {
        timeSaved: Math.round(automatedHours),
        costSaved: Math.round(annualSaving),
        payback: paybackMonths <= 1 ? "< 1 mois" : `${paybackMonths} mois`,
      };
    },
  },
  sales: {
    label: "Qualification leads",
    icon: "📈",
    description: "Scoring et enrichissement automatique des leads",
    fields: [
      { key: "leads", label: "Leads par mois", unit: "leads", defaultValue: 200, min: 20, max: 2000, step: 20 },
      { key: "qualTime", label: "Temps de qualification par lead", unit: "min", defaultValue: 20, min: 5, max: 45, step: 5 },
      { key: "hourlyRate", label: "Coût horaire commercial", unit: "€/h", defaultValue: 45, min: 20, max: 100, step: 5 },
      { key: "conversionLift", label: "Gain de conversion estimé", unit: "%", defaultValue: 50, min: 10, max: 100, step: 10 },
    ],
    calculate: (v) => {
      const totalHours = (v.leads * v.qualTime) / 60;
      const automatedHours = totalHours * 0.75; // 75% time saved on qualification
      const monthlySaving = automatedHours * v.hourlyRate;
      const annualSaving = monthlySaving * 12;
      const toolCost = 300;
      const paybackMonths = Math.ceil(toolCost / monthlySaving);
      return {
        timeSaved: Math.round(automatedHours),
        costSaved: Math.round(annualSaving),
        payback: paybackMonths <= 1 ? "< 1 mois" : `${paybackMonths} mois`,
      };
    },
  },
  rh: {
    label: "Recrutement & RH",
    icon: "👥",
    description: "Tri automatique de CV et pré-qualification",
    fields: [
      { key: "cvs", label: "CV reçus par mois", unit: "CV", defaultValue: 300, min: 50, max: 3000, step: 50 },
      { key: "screenTime", label: "Temps de tri par CV", unit: "min", defaultValue: 10, min: 3, max: 30, step: 1 },
      { key: "hourlyRate", label: "Coût horaire recruteur", unit: "€/h", defaultValue: 40, min: 20, max: 80, step: 5 },
      { key: "automationRate", label: "Taux d'automatisation estimé", unit: "%", defaultValue: 60, min: 20, max: 80, step: 10 },
    ],
    calculate: (v) => {
      const totalHours = (v.cvs * v.screenTime) / 60;
      const automatedHours = totalHours * (v.automationRate / 100);
      const monthlySaving = automatedHours * v.hourlyRate;
      const annualSaving = monthlySaving * 12;
      const toolCost = 300;
      const paybackMonths = Math.ceil(toolCost / monthlySaving);
      return {
        timeSaved: Math.round(automatedHours),
        costSaved: Math.round(annualSaving),
        payback: paybackMonths <= 1 ? "< 1 mois" : `${paybackMonths} mois`,
      };
    },
  },
  finance: {
    label: "Finance & Comptabilité",
    icon: "💰",
    description: "Rapprochement factures et reporting automatisé",
    fields: [
      { key: "invoices", label: "Factures traitées par mois", unit: "factures", defaultValue: 400, min: 50, max: 5000, step: 50 },
      { key: "processTime", label: "Temps de traitement par facture", unit: "min", defaultValue: 12, min: 3, max: 30, step: 1 },
      { key: "hourlyRate", label: "Coût horaire comptable", unit: "€/h", defaultValue: 38, min: 20, max: 80, step: 2 },
      { key: "automationRate", label: "Taux d'automatisation estimé", unit: "%", defaultValue: 55, min: 20, max: 80, step: 5 },
    ],
    calculate: (v) => {
      const totalHours = (v.invoices * v.processTime) / 60;
      const automatedHours = totalHours * (v.automationRate / 100);
      const monthlySaving = automatedHours * v.hourlyRate;
      const annualSaving = monthlySaving * 12;
      const toolCost = 300;
      const paybackMonths = Math.ceil(toolCost / monthlySaving);
      return {
        timeSaved: Math.round(automatedHours),
        costSaved: Math.round(annualSaving),
        payback: paybackMonths <= 1 ? "< 1 mois" : `${paybackMonths} mois`,
      };
    },
  },
};

export default function CalculateurROIPage() {
  const [activeScenario, setActiveScenario] = useState<Scenario>("support");
  const scenario = scenarios[activeScenario];

  const [values, setValues] = useState<Record<string, number>>(() => {
    const initial: Record<string, number> = {};
    scenario.fields.forEach((f) => {
      initial[f.key] = f.defaultValue;
    });
    return initial;
  });

  const handleScenarioChange = (s: Scenario) => {
    setActiveScenario(s);
    const initial: Record<string, number> = {};
    scenarios[s].fields.forEach((f) => {
      initial[f.key] = f.defaultValue;
    });
    setValues(initial);
  };

  const result = scenario.calculate(values);

  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="text-center mb-12">
        <Badge variant="secondary" className="mb-4 text-xs">
          Outil gratuit
        </Badge>
        <h1 className="text-3xl font-bold sm:text-4xl lg:text-5xl">
          Calculez le <span className="gradient-text">ROI de votre Agent IA</span>
        </h1>
        <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
          Estimez les gains de temps et d&apos;argent que l&apos;IA peut apporter
          à votre équipe. Choisissez votre scénario et ajustez les paramètres.
        </p>
      </div>

      {/* Scenario selector */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mx-auto mb-12">
        {(Object.keys(scenarios) as Scenario[]).map((key) => {
          const s = scenarios[key];
          return (
            <button
              key={key}
              onClick={() => handleScenarioChange(key)}
              className={`rounded-xl border p-4 text-left transition-all ${
                activeScenario === key
                  ? "border-primary bg-primary/5 shadow-sm"
                  : "hover:border-primary/30 hover:bg-accent/50"
              }`}
            >
              <span className="text-2xl">{s.icon}</span>
              <p className="mt-1 text-sm font-semibold">{s.label}</p>
              <p className="text-xs text-muted-foreground line-clamp-2 mt-0.5">{s.description}</p>
            </button>
          );
        })}
      </div>

      <div className="grid gap-8 lg:grid-cols-5 max-w-5xl mx-auto">
        {/* Input sliders */}
        <div className="lg:col-span-3 space-y-6">
          <Card>
            <CardHeader className="pb-2">
              <h2 className="text-lg font-semibold flex items-center gap-2">
                <span>{scenario.icon}</span>
                {scenario.label}
              </h2>
              <p className="text-sm text-muted-foreground">{scenario.description}</p>
            </CardHeader>
            <CardContent className="space-y-6">
              {scenario.fields.map((field) => (
                <div key={field.key}>
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium">{field.label}</label>
                    <span className="text-sm font-semibold text-primary">
                      {values[field.key]?.toLocaleString("fr-FR")} {field.unit}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={field.min}
                    max={field.max}
                    step={field.step}
                    value={values[field.key] ?? field.defaultValue}
                    onChange={(e) =>
                      setValues((prev) => ({ ...prev, [field.key]: Number(e.target.value) }))
                    }
                    className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>{field.min.toLocaleString("fr-FR")} {field.unit}</span>
                    <span>{field.max.toLocaleString("fr-FR")} {field.unit}</span>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Results */}
        <div className="lg:col-span-2 space-y-4">
          <Card className="border-primary/30 bg-primary/5">
            <CardContent className="pt-6">
              <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-6">
                Estimation de gains
              </h3>
              <div className="space-y-6">
                <div>
                  <p className="text-sm text-muted-foreground">Temps gagné / mois</p>
                  <p className="text-3xl font-bold text-primary">
                    {result.timeSaved.toLocaleString("fr-FR")}h
                  </p>
                  <p className="text-xs text-muted-foreground">
                    soit ~{Math.round(result.timeSaved / 7)} jours ouvrés
                  </p>
                </div>
                <div className="border-t pt-4">
                  <p className="text-sm text-muted-foreground">Économie annuelle estimée</p>
                  <p className="text-3xl font-bold text-primary">
                    {result.costSaved.toLocaleString("fr-FR")}&nbsp;&euro;
                  </p>
                  <p className="text-xs text-muted-foreground">
                    soit {Math.round(result.costSaved / 12).toLocaleString("fr-FR")}&nbsp;&euro;/mois
                  </p>
                </div>
                <div className="border-t pt-4">
                  <p className="text-sm text-muted-foreground">Retour sur investissement</p>
                  <p className="text-3xl font-bold text-primary">{result.payback}</p>
                  <p className="text-xs text-muted-foreground">
                    coût estimé de la solution : ~300&nbsp;&euro;/mois
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="pt-6 space-y-3">
              <Button className="w-full" size="lg" asChild>
                <Link href={`/catalogue?fn=${scenario.label.split(" ")[0]}`}>
                  Voir les workflows {scenario.label}
                </Link>
              </Button>
              <Button className="w-full" variant="outline" asChild>
                <Link href="/demande">Demander un workflow sur mesure</Link>
              </Button>
              <p className="text-xs text-center text-muted-foreground">
                Estimation basée sur les retours de nos utilisateurs.
                Les résultats réels peuvent varier.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Trust section */}
      <div className="mt-16 text-center">
        <h2 className="text-xl font-bold mb-6">Comment ces chiffres sont-ils calculés ?</h2>
        <div className="grid gap-4 sm:grid-cols-3 max-w-3xl mx-auto text-left">
          <div className="rounded-xl border p-4">
            <h3 className="text-sm font-semibold mb-1">Données réelles</h3>
            <p className="text-xs text-muted-foreground">
              Nos estimations sont basées sur les retours de 50+ entreprises françaises
              ayant déployé des agents IA.
            </p>
          </div>
          <div className="rounded-xl border p-4">
            <h3 className="text-sm font-semibold mb-1">Hypothèses conservatrices</h3>
            <p className="text-xs text-muted-foreground">
              Le coût de la solution (~300&euro;/mois) inclut l&apos;hébergement,
              les API IA, et l&apos;orchestration n8n/Make.
            </p>
          </div>
          <div className="rounded-xl border p-4">
            <h3 className="text-sm font-semibold mb-1">Personnalisable</h3>
            <p className="text-xs text-muted-foreground">
              Ajustez chaque paramètre pour refléter votre contexte.
              Contactez-nous pour un calcul détaillé.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
