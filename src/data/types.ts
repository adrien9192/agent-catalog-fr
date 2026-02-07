export interface StackItem {
  name: string;
  category: "LLM" | "Orchestration" | "Database" | "Hosting" | "Monitoring" | "Other";
  url?: string;
  isFree?: boolean;
}

export interface CodeSnippet {
  language: string;
  code: string;
  filename?: string;
}

export interface TutorialSection {
  title: string;
  content: string;
  codeSnippets?: CodeSnippet[];
}

export type Difficulty = "Facile" | "Moyen" | "Expert";

export interface UseCase {
  slug: string;
  title: string;
  subtitle: string;
  problem: string;
  value: string;
  inputs: string[];
  outputs: string[];
  risks: string[];
  roiIndicatif: string;
  recommendedStack: StackItem[];
  lowCostAlternatives: StackItem[];
  architectureDiagram: string;
  tutorial: TutorialSection[];
  difficulty: Difficulty;
  sectors: string[];
  metiers: string[];
  functions: string[];
  metaTitle: string;
  metaDescription: string;
  createdAt: string;
  updatedAt: string;
}

export interface SectorInfo {
  slug: string;
  name: string;
  description: string;
  icon: string;
}

export interface MetierInfo {
  slug: string;
  name: string;
  description: string;
  icon: string;
}
