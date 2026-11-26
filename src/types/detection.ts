import type { ComputedImageSignals } from '../utils/imageForensics';

export type DetectionEngine = 'pytorch' | 'gemini';

export interface ModelCheckpointInfo {
  name: string;
  label: string;
  note: string;
  sizeBytes: number;
  recommended: boolean;
}

export interface ForensicIndicator {
  category: string;
  observation: string;
  signal: 'synthetic' | 'authentic' | 'neutral';
}

export interface DetectionResult {
  fileName: string;
  verdict: 'ai_generated' | 'authentic' | 'inconclusive';
  fakeProbability: number;
  realProbability: number;
  confidenceLevel: 'High' | 'Moderate' | 'Low';
  domainCategory: string;
  summary: string;
  indicators: ForensicIndicator[];
  analyzedAt: string;
  engine: string;
  engineMode: DetectionEngine;
  modelName?: string;
  latencyMs?: number;
}

export interface HistoryEntry {
  id: string;
  fileName: string;
  thumbnailDataUrl: string;
  verdict: DetectionResult['verdict'];
  fakeProbability: number;
  realProbability: number;
  width: number;
  height: number;
  analyzedAt: string;
  result: DetectionResult;
  signals: Omit<ComputedImageSignals, 'uploadBase64'>;
  previewUrl: string;
}
