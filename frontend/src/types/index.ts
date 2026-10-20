export type ClusterId = 'Cluster 1' | 'Cluster 2' | 'Cluster 3' | 'Cluster 4';

export type NodeId = '1' | '2';

export interface PlantOptimalRange {
  soil: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  temperature: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  humidity: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
  light: { min: number; max: number; unit: string; lowLabel: string; idealLabel: string; highLabel: string };
}

export interface SensorRecord {
  record_id?: string;
  timestamp: string; // ISO 8601 string, e.g. 2024-01-15T12:00:00
  session_id: number; // integer >= 1 (session count)
  node_id: NodeId; // "1" or "2"
  sampling_point: number; // 1 to 12 (multiplexer channel position)
  plant_id: number; // 1 to 16 (plant index / pot number)
  soil: number; // 0.0 to 100.0 (% moisture)
  temperature: number; // -40.0 to 80.0 (°C)
  humidity: number; // 0.0 to 100.0 (% relative humidity)
  light: number; // 0.0 to 100000.0 (lux)
  received_at?: string; // Jetson receipt timestamp
  accepted_at?: string | null;
  status?: 'accepted' | 'rejected' | 'flagged';
  validation_error?: string;
  rssi?: number;
  crc_ok?: boolean;
}

export type PlantHealthStatus = 'healthy' | 'attention' | 'critical' | 'stale' | 'unmeasured';

export type TulsiCondition = 'Healthy' | 'Water Stress' | 'Light Stress' | 'Nutrient Stress';

export interface PlantPrediction {
  condition: TulsiCondition;
  confidence: number;
  timestamp: string;
  modelVersion: string;
  simpleExplanation?: string;
}

export interface Plant {
  id: number; // 1 to 16
  name: string; // e.g. "Pot 1: Rama Tulsi"
  species: string; // Ocimum sanctum or Ocimum tenuiflorum
  variety: 'Rama Tulsi (Green)' | 'Krishna Tulsi (Purple)';
  cluster: ClusterId;
  clusterLabel: string; // e.g. "Cluster 1: Healthy Control"
  stressType: TulsiCondition;
  nodeId: NodeId; // "1" or "2"
  samplingPoint: number; // 1 to 12
  optimalRanges: PlantOptimalRange;
  latestReading?: SensorRecord;
  status: PlantHealthStatus;
  conditionPrediction?: PlantPrediction;
  description: string;
  simpleCareTip: string;
  location: string;
}

export interface Session {
  session_id: number; // integer >= 1
  scheduled_at: string;
  started_at: string;
  ended_at: string | null;
  status: 'completed' | 'in_progress' | 'scheduled' | 'failed' | 'partial';
  expected_records: number;
  received_records: number;
  rejected_records: number;
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
  cpu_usage?: number;
  gpu_usage?: number;
  ram_used_gb?: number;
  ram_total_gb?: number;
  disk_used_gb?: number;
  disk_total_gb?: number;
  temperature_c?: number;
  rssi_dbm?: number;
  channels?: number;
  uptime?: string;
  firmware_version?: string;
}

export interface Device {
  id: string;
  name: string;
  type: 'jetson' | 'esp32';
  nodeNumber?: '1' | '2';
  ip_address: string;
  mac_address: string;
  status: 'online' | 'offline' | 'unknown' | 'demo';
  last_heartbeat: string;
  last_transmission?: string;
  hardware_details: HardwareDetails;
  role: string;
  simpleRole: string;
}

export interface ShapFeatureContribution {
  name: string;
  displayName: string;
  observed_value: number;
  unit: string;
  shap_value: number;
  direction: 'increases_prediction' | 'decreases_prediction' | 'neutral';
  description: string;
  laymanExplanation: string;
}

export interface LocalShapExplanation {
  plant_id: number;
  record_id?: string;
  model_version: string;
  target_class: string;
  predicted_probability: number;
  base_value: number;
  model_output: number;
  features: ShapFeatureContribution[];
  laymanSummary: string;
  explanation_note: string;
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
  feature_value_norm: number;
  shap_value: number;
  plant_id: number;
}

export interface GlobalShapData {
  model_version: string;
  sample_count: number;
  features: GlobalShapFeature[];
  beeswarm_points: BeeswarmPoint[];
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
    labels: string[];
    matrix: number[][];
  };
}

export type RecommendationPriority = 'high' | 'medium' | 'low';
export type RecommendationStatus = 'new' | 'in_progress' | 'completed' | 'dismissed';
export type RecommendationCategory = 'irrigation' | 'lighting' | 'nutrition' | 'sensor_check' | 'climate';

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
  updated_at?: string;
  resolved_at?: string;
}

export interface SystemSettings {
  api_base_url: string;
  polling_interval_seconds: number;
  demo_mode: boolean;
  jetson_receiver_port: number;
  ph_sensor_installed: boolean;
  collection_times: string[];
  log_retention_days: number;
  storage_path: string;
  soil_adc_dry: number;
  soil_adc_wet: number;
}
