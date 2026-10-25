import type { PlantHealthStatus, Severity } from '../types';

/** Copy for the model's severity classes. Used by every severity badge (single source). */
export const SEVERITY_LABELS: Record<Severity, string> = {
  healthy: 'Healthy',
  low: 'Low stress',
  moderate: 'Moderate stress',
  high: 'High stress',
};

/** Copy for the plant status derived from the severity. Not the experimental group (a control pot can be rated stressed). */
export const PLANT_STATUS_LABELS: Record<PlantHealthStatus, string> = {
  healthy: 'Rated healthy',
  attention: 'Stress detected',
  critical: 'Severe stress',
  stale: 'Stale reading',
  unmeasured: 'Not analyzed yet',
};
