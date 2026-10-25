import type { Device, Overview, Plant, PlantMonitoring, Prediction, SensorRecord, Session, SystemSettings } from '../types';
import overviewJson from './fixtures/overview.json';
import plantsJson from './fixtures/plants.json';
import monitoringJson from './fixtures/plant5-monitoring.json';
import readingsJson from './fixtures/readings.json';
import flaggedJson from './fixtures/flagged-logs.json';
import sessionsJson from './fixtures/sessions.json';
import devicesJson from './fixtures/devices.json';
import settingsJson from './fixtures/settings.json';
import predictionJson from './fixtures/prediction-r5365.json';

/** Captured from the live backend after a full Run. Deep copies so tests can mutate freely. */
const copy = <T,>(v: unknown): T => JSON.parse(JSON.stringify(v)) as T;

export const overviewFixture = (): Overview => copy(overviewJson);
/** Pots 5, 1, 9, 13 (pot 5 first so it is the default focus / trend pot). */
export const plantsFixture = (): Plant[] => {
  const all = copy<Plant[]>(plantsJson);
  return [all.find((p) => p.id === 5)!, ...all.filter((p) => p.id !== 5)];
};
export const monitoringFixture = (): PlantMonitoring => copy(monitoringJson);
export const readingsFixture = (): SensorRecord[] => copy(readingsJson);
export const flaggedFixture = (): SensorRecord[] => copy(flaggedJson);
export const sessionsFixture = (): Session[] => copy(sessionsJson);
export const devicesFixture = (): Device[] => copy(devicesJson);
export const settingsFixture = (): SystemSettings => copy(settingsJson);
export const predictionFixture = (): Prediction[] => copy(predictionJson);

export const errorEnvelope = (code: string, message: string, details: unknown = null) => ({ error: { code, message, details } });

/** Overview as served before any Run: readings exist, no predictions, no model-dependent lists. */
export function preRunOverview(): Overview {
  const o = overviewFixture();
  o.totals.predictions = { ok: 0, unavailable: 0, rejected: 0, total: 0 };
  o.totals.anomalies = 0;
  o.plantStatusCounts = { healthy: 0, attention: 0, critical: 0, stale: 0, unmeasured: 16 };
  o.severityCounts = { healthy: 0, low: 0, moderate: 0, high: 0, unmeasured: 16 };
  for (const c of o.clusterSummaries) {
    c.severityCounts = { healthy: 0, low: 0, moderate: 0, high: 0, unmeasured: c.plants };
    c.topCondition = null;
  }
  o.recentAlerts = [];
  o.topRecommendations = [];
  o.modelVersion = null;
  return o;
}

/** Plants before a Run: no prediction, status unmeasured. */
export function preRunPlants(): Plant[] {
  return plantsFixture().map((p) => ({ ...p, conditionPrediction: null, status: 'unmeasured' as const }));
}

/** Monitoring payload before a Run. */
export function preRunMonitoring(): PlantMonitoring {
  const m = monitoringFixture();
  m.plant.conditionPrediction = null;
  m.plant.status = 'unmeasured';
  m.predictionHistory = [];
  m.recommendations = [];
  return m;
}
