import type { AiInsights, GlobalShapData, HealthResponse, LocalShapExplanation, ModelMetrics, Recommendation } from '../types';
import insightsJson from './fixtures/ai/insights.json';
import localJson from './fixtures/ai/shap_local_5.json';
import globalJson from './fixtures/ai/shap_global.json';
import metricsJson from './fixtures/ai/metrics.json';
import healthJson from './fixtures/ai/health.json';
import recsJson from './fixtures/ai/recommendations.json';
import plantsJson from './fixtures/ai/plants_min.json';

/** Captured from the live backend after a full Run (see src/test/fixtures/ai). Deep copies so tests can mutate. */
const copy = <T,>(v: unknown): T => JSON.parse(JSON.stringify(v)) as T;

export const insightsFixture = (): AiInsights => copy(insightsJson);
export const localShapFixture = (): LocalShapExplanation => copy(localJson);
export const globalShapFixture = (): GlobalShapData => copy(globalJson);
export const metricsFixture = (): ModelMetrics => copy(metricsJson);
export const healthFixture = (): HealthResponse => copy(healthJson);
export const recommendationsFixture = (): Recommendation[] => copy(recsJson);
export const plantsFixture = (): Array<{ id: number; name: string; species: string; cluster: string; nodeId: string }> => copy(plantsJson);

export const errorEnvelope = (code: string, message: string, details: unknown = null) => ({ error: { code, message, details } });
