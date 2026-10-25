export type ClusterId = 'Cluster 1' | 'Cluster 2' | 'Cluster 3' | 'Cluster 4';

export type NodeId = '1' | '2';

export interface PlantOptimalRange {
  soil: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  temperature: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  humidity: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  light: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
}

export interface DatasetLabels {
  stressLevel: string | null;
  stressType: string | null;
  isSyntheticAnomaly: boolean | null;
}

/** One stored reading. Sensor values are null when the dataset row was missing them (never filled in). */
export interface SensorRecord {
  record_id: string; // "R-<db id>", always present
  timestamp: string; // ISO 8601 with +05:30 offset
  session_id: number | null; // integer >= 1
  node_id: NodeId; // "1" or "2"
  sampling_point: number | null; // 1 to 12 (multiplexer channel position)
  plant_id: number; // 1 to 16 (plant index / pot number)
  soil: number | null; // % moisture
  temperature: number | null; // degrees C
  humidity: number | null; // % relative humidity
  light: number | null; // lux
  received_at: string | null;
  accepted_at: string | null;
  status: 'accepted' | 'rejected' | 'flagged';
  validation_error: string | null;
  rssi: number | null; // always null for dataset rows
  crc_ok: boolean | null; // always null for dataset rows
  /** Dataset ground truth (synthetic). Never model output. */
  datasetLabels: DatasetLabels;
}

export type PlantHealthStatus = 'healthy' | 'attention' | 'critical' | 'stale' | 'unmeasured';

export type TulsiCondition =
  | 'Healthy'
  | 'Water Stress'
  | 'Light Stress'
  | 'Temperature/Humidity Stress'
  | 'Mixed Stress'
  | 'Unclassified Stress';

export type Severity = 'healthy' | 'low' | 'moderate' | 'high';

/** KMeans environmental state group. Not a stress level. */
export interface EnvState {
  id: number;
  label: string; // "Environmental state group N"
  distance: number | null;
}

/** Isolation Forest output. An unusual reading (check the sensor), not a plant-health verdict. */
export interface AnomalyInfo {
  isAnomaly: boolean | null;
  score: number | null;
}

export interface PlantPrediction {
  condition: TulsiCondition | null; // heuristic stress-type guess
  conditionSource: 'heuristic';
  severity: Severity | null;
  probabilities: Record<string, number> | null;
  confidence: number | null;
  timestamp: string;
  modelVersion: string | null;
  recordId: string;
  simpleExplanation: string;
  envState: EnvState | null;
  anomaly: AnomalyInfo;
}

export interface Plant {
  id: number; // 1 to 16
  name: string; // e.g. "Pot 1: Rama Tulsi"
  species: string; // Ocimum sanctum or Ocimum tenuiflorum
  variety: 'Rama Tulsi (Green)' | 'Krishna Tulsi (Purple)';
  cluster: ClusterId; // experimental group, not a KMeans cluster
  clusterLabel: string; // e.g. "Cluster 1: Healthy Control"
  stressType: TulsiCondition; // the group's designed treatment
  nodeId: NodeId; // "1" or "2"
  samplingPoint: number | null; // multiplexer channel of the latest reading
  optimalRanges: PlantOptimalRange;
  latestReading: SensorRecord | null;
  status: PlantHealthStatus;
  conditionPrediction: PlantPrediction | null;
  description: string;
  simpleCareTip: string;
  location: string;
}

export interface Session {
  session_id: number; // integer >= 1
  scheduled_at: string | null;
  started_at: string | null;
  ended_at: string | null;
  status: 'completed' | 'in_progress' | 'scheduled' | 'failed' | 'partial';
  expected_records: number;
  received_records: number;
  rejected_records: number;
  flagged_records: number;
  duration_seconds: number;
  trigger_type: 'scheduled' | 'manual' | 'retry';
  node_progress: {
    [key: string]: {
      expected: number;
      received: number;
      status: 'completed' | 'in_progress' | 'waiting' | 'failed';
    };
  };
}

export interface HardwareDetails {
  model: string;
  cpu_usage: number | null;
  gpu_usage: number | null;
  ram_used_gb: number | null;
  ram_total_gb: number | null;
  disk_used_gb: number | null;
  disk_total_gb: number | null;
  temperature_c: number | null;
  rssi_dbm: number | null;
  channels: number | null;
  uptime: string | null;
  firmware_version: string | null;
}

export interface Device {
  id: string;
  name: string;
  type: 'jetson' | 'esp32';
  nodeNumber: '1' | '2' | null;
  ip_address: string | null;
  mac_address: string | null;
  status: 'online' | 'offline' | 'unknown';
  last_heartbeat: string | null;
  last_transmission: string | null;
  hardware_details: HardwareDetails;
  role: string;
  simpleRole: string;
  readingCount: number;
  plantIds: number[];
}

export interface ShapFeatureContribution {
  name: string;
  displayName: string;
  observed_value: number | null;
  unit: string;
  shap_value: number;
  direction: 'increases_prediction' | 'decreases_prediction' | 'neutral';
  description: string;
  laymanExplanation: string;
}

export interface LocalShapExplanation {
  plant_id: number;
  record_id: string;
  model_version: string | null;
  target_class: string;
  predicted_probability: number | null;
  base_value: number | null; // always null: the ML service gives no base value
  model_output: number | null; // always null
  features: ShapFeatureContribution[];
  laymanSummary: string;
  explanation_note: string;
  timestamp: string;
}

export interface GlobalShapFeature {
  name: string;
  displayName: string;
  mean_abs_shap: number;
  importance_rank: number;
  interpretation: string;
  laymanTip: string;
}

export interface BeeswarmPoint {
  id: string;
  feature: string;
  feature_value_norm: number | null;
  shap_value: number;
  plant_id: number | null;
  predicted_class?: string | null;
}

export interface PerClassFeature {
  name: string;
  displayName: string;
  mean_abs_shap: number;
}

export interface GlobalShapData {
  model_version: string | null;
  sample_count: number;
  features: GlobalShapFeature[];
  beeswarm_points: BeeswarmPoint[];
  perClassTop15: Record<string, PerClassFeature[]>;
  source: { path: string; generatedAt: string };
  approximate: { beeswarm: string };
  units: string;
  note?: string | null;
}

export interface BaselineMetrics {
  accuracy: number;
  macro_f1: number;
}

export interface ModelMetrics {
  model_name: string;
  model_version: string;
  evaluated_at: string;
  dataset_name: string;
  total_samples: number;
  split: string;
  overall: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  per_class: Array<{
    class_name: string;
    precision: number;
    recall: number;
    f1_score: number;
    support: number;
    laymanStatus: string;
  }>;
  confusion_matrix: {
    labels: string[]; // number of classes = labels.length (4)
    matrix: number[][];
  };
  provenance: 'synthetic';
  validatedOnRealPlants: boolean;
  runId: string;
  balancedAccuracy: number | null;
  weightedF1: number | null;
  baselines: Record<string, BaselineMetrics>;
  anomalyModel: Record<string, unknown> | null;
  clusteringModel: Record<string, unknown> | null;
  note: string;
}

export type RecommendationPriority = 'high' | 'medium' | 'low';
export type RecommendationStatus = 'new' | 'in_progress' | 'completed' | 'dismissed' | 'superseded';
/** Statuses a user may set (superseded is system-set). */
export type UserRecommendationStatus = Exclude<RecommendationStatus, 'superseded'>;
export type RecommendationCategory =
  | 'irrigation'
  | 'lighting'
  | 'nutrition' // kept in the type, never produced
  | 'sensor_check'
  | 'climate'
  | 'monitoring';
export type RecommendationSource = 'ml' | 'heuristic';

export interface Recommendation {
  id: string;
  plant_id: number; // 1 to 16
  plant_name: string; // e.g. "Pot 5: Rama Tulsi"
  cluster: ClusterId;
  title: string;
  simpleAction: string; // Layman straightforward action step, e.g. "Add 1 cup (250ml) water"
  priority: RecommendationPriority;
  status: RecommendationStatus;
  category: RecommendationCategory;
  supporting_sensor_values: Array<{
    metric: string;
    value: string;
    target: string;
    status: 'good' | 'low' | 'high' | 'attention';
  }>;
  reason: string;
  scientificContext: string; // A bit technical yet simple explanation
  guidance_source: string;
  created_at: string;
  updated_at: string | null;
  resolved_at: string | null;
  source: RecommendationSource;
  confidence: number | null;
  basedOn: { timestamp: string; modelVersion: string | null; runId: string | null } | null;
  disclaimer: true;
  mlRecId: string | null;
}

export interface SystemSettings {
  polling_interval_seconds: number;
  jetson_receiver_port: number;
  ph_sensor_installed: boolean;
  collection_times: string[];
  log_retention_days: number;
  storage_path: string;
  soil_adc_dry: number;
  soil_adc_wet: number;
}

// ---------------------------------------------------------------------------
// Backend response types added in Phase 4a (match backend/app/schemas/*.py)
// ---------------------------------------------------------------------------

export interface ApiErrorEnvelope {
  error: { code: string; message: string; details: unknown };
}

/** `{items, total}` for list endpoints (`total` comes from the X-Total-Count header). */
export interface Page<T> {
  items: T[];
  total: number;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | string;
  db: { ok: boolean; readings: number; plants: number };
  mlService: {
    ok: boolean;
    url: string;
    detail: string;
    status: string | null;
    modelsLoaded: boolean | null;
    explainabilityAvailable: boolean | null;
  };
  modelVersion: string | null;
  runId: string | null;
  datasetName: string;
  seeded: boolean;
}

export type PipelineState = 'idle' | 'queued' | 'running' | 'completed' | 'failed';

export interface PipelineStatus {
  runId: number | null;
  state: PipelineState;
  total: number;
  processed: number;
  failed: number;
  percent: number;
  currentPlantId: number | null;
  startedAt: string | null;
  finishedAt: string | null;
  message: string | null;
  mlServiceOk: boolean;
  modelVersion: string | null;
  readyToRun: boolean;
}

export interface PipelineRunResponse {
  runId: number;
  state: PipelineState;
}

export interface PredictionTotals {
  ok: number;
  unavailable: number;
  rejected: number;
  total: number;
}

export interface OverviewTotals {
  plants: number;
  readings: number;
  flaggedReadings: number;
  predictions: PredictionTotals;
  anomalies: number;
}

export interface SensorAverages {
  soil: number | null;
  temperature: number | null;
  humidity: number | null;
  light: number | null;
}

export interface ClusterSummary {
  cluster: ClusterId;
  clusterLabel: string;
  plantIds: number[];
  plants: number;
  averages: SensorAverages;
  severityCounts: Record<string, number>; // healthy/low/moderate/high/unmeasured
  topCondition: TulsiCondition | null;
}

export interface Alert {
  kind: 'anomaly' | 'recommendation';
  plantId: number;
  plantName: string;
  timestamp: string;
  message: string;
  severity: string | null;
  recordId: string | null;
  recommendationId: string | null;
}

export interface DevicesSummary {
  total: number;
  online: number;
  offline: number;
  unknown: number;
  lastTransmission: string | null;
}

export interface Overview {
  totals: OverviewTotals;
  plantStatusCounts: Record<string, number>; // healthy/attention/critical/stale/unmeasured
  severityCounts: Record<string, number>; // healthy/low/moderate/high/unmeasured
  clusterSummaries: ClusterSummary[];
  recentAlerts: Alert[];
  topRecommendations: Recommendation[];
  latestSession: Session | null;
  devicesSummary: DevicesSummary;
  modelVersion: string | null;
  provenance: string;
  disclaimer: string;
  pipeline: PipelineStatus;
}

export interface SeriesPoint {
  timestamp: string;
  soil: number | null;
  temperature: number | null;
  humidity: number | null;
  light: number | null;
}

/** Mixed case on purpose: `record_id` is snake_case, the rest camelCase. */
export interface PredictionHistoryPoint {
  timestamp: string;
  record_id: string;
  status: string;
  severity: Severity | null;
  probabilities: Record<string, number> | null;
  confidence: number | null;
  condition: TulsiCondition | null;
  isAnomaly: boolean | null;
  anomalyScore: number | null;
  envState: EnvState | null;
}

export interface PlantMonitoring {
  plant: Plant;
  series: SeriesPoint[];
  predictionHistory: PredictionHistoryPoint[];
  recommendations: Recommendation[];
}

/** One row of GET/POST /api/predictions (`record_id` and `plant_id` are snake_case, the rest camelCase). */
export interface Prediction {
  id: number;
  readingId: number;
  record_id: string;
  plant_id: number;
  timestamp: string;
  status: 'ok' | 'unavailable' | 'rejected' | string;
  error: string | null;
  warnings: string[];
  modelVersion: string | null;
  runId: string | null;
  severity: Severity | null;
  probabilities: Record<string, number> | null;
  confidence: number | null;
  condition: TulsiCondition | null;
  conditionSource: 'heuristic';
  envState: EnvState | null;
  anomaly: AnomalyInfo;
}

export interface AnomalyRecord {
  id: string;
  plant_id: number;
  plant_name: string;
  record_id: string;
  timestamp: string;
  score: number;
  threshold: number;
  interpretation: string;
  readings: { soil: number | null; temperature: number | null; humidity: number | null; light: number | null };
  datasetLabelIsSyntheticAnomaly: boolean | null;
}

export interface AiInsights {
  hasPredictions: boolean;
  predictions: Record<string, number>; // by status: ok/unavailable/rejected
  severityCounts: Record<string, number>;
  latestSeverityCounts: Record<string, number>;
  conditionGuessCounts: Record<string, number>;
  latestConditionGuessCounts: Record<string, number>;
  envStateCounts: Array<{ id: number; label: string; count: number; note: string }>;
  anomalies: { count: number; rate: number; rateBasis: string; interpretation: string };
  explanationCoverage: { plantsWithLocalShap: number; totalPlants: number; plantIds: number[] };
  topFeatures: {
    global: Array<{ name: string; displayName: string; meanAbsShap: number }> | null;
    globalSource: { path: string; generatedAt: string } | null;
    localFrequency: Array<{ name: string; displayName: string; count: number; unit: string }>;
  };
  model: {
    mlServiceOk: boolean;
    modelVersion: string | null;
    runId: string | null;
    explainabilityAvailable: boolean | null;
    trainedAt: string | null;
  };
  lastPipelineRun: {
    runId: number;
    status: string;
    total: number;
    processed: number;
    failed: number;
    startedAt: string | null;
    finishedAt: string | null;
    message: string | null;
  } | null;
  provenance: string;
  disclaimer: string;
}

export interface IngestResult {
  accepted: number;
  flagged: number;
  duplicates: number;
  items: Array<{
    record_id: string | null;
    plant_id: number;
    timestamp: string;
    status: 'accepted' | 'flagged' | 'duplicate';
  }>;
}

/** 404 `global_shap_unavailable`: the empty-state hint says which command to run. */
export interface GlobalShapUnavailable {
  message: string;
  hint: string | null;
}
