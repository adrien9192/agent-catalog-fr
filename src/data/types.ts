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

export interface EnterpriseSection {
  piiHandling: string;
  auditLog: string;
  humanInTheLoop: string;
  monitoring: string;
}

export interface N8nWorkflow {
  description: string;
  nodes: string[];
  triggerType: string;
}

export interface N8nTutorialStep {
  nodeLabel: string;
  nodeType: string;
  nodeIcon: string;
  description: string;
  configuration: string;
  customization: string;
  errorHandling?: string;
}

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
  n8nTutorial?: N8nTutorialStep[];
  enterprise: EnterpriseSection;
  n8nWorkflow: N8nWorkflow;
  estimatedTime: string;
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
