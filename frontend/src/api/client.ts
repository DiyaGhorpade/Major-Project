import {
  Plant,
  SensorRecord,
  Session,
  Device,
  LocalShapExplanation,
  GlobalShapData,
  ModelMetrics,
  Recommendation,
  RecommendationStatus,
  SystemSettings,
} from '../types';
import {
  mockPlants,
  mockRecords,
  mockDevices,
  mockSessions,
  mockLocalShapExplanation,
  mockGlobalShapData,
  mockModelMetrics,
  mockRecommendations,
  initialSystemSettings,
} from './mockData';

const STORAGE_KEY_RECS = 'plant_monitor_tulsi_recs_v3';
const STORAGE_KEY_SETTINGS = 'plant_monitor_tulsi_settings_v3';

class ApiClient {
  private inMemoryRecords: SensorRecord[] = [...mockRecords];
  private inMemoryPlants: Plant[] = [...mockPlants];
  private inMemoryRecommendations: Recommendation[] = [];
  private inMemorySettings: SystemSettings = { ...initialSystemSettings };
  private modelConnected: boolean = true;

  constructor() {
    this.loadState();
  }

  private loadState() {
    try {
      const savedRecs = localStorage.getItem(STORAGE_KEY_RECS);
      if (savedRecs) {
        this.inMemoryRecommendations = JSON.parse(savedRecs);
      } else {
        this.inMemoryRecommendations = [...mockRecommendations];
        this.saveRecommendations();
      }

      const savedSettings = localStorage.getItem(STORAGE_KEY_SETTINGS);
      if (savedSettings) {
        this.inMemorySettings = { ...initialSystemSettings, ...JSON.parse(savedSettings) };
      }
    } catch {
      this.inMemoryRecommendations = [...mockRecommendations];
      this.inMemorySettings = { ...initialSystemSettings };
    }
  }

  private saveRecommendations() {
    try {
      localStorage.setItem(STORAGE_KEY_RECS, JSON.stringify(this.inMemoryRecommendations));
    } catch (e) {
      console.warn('Could not persist recommendations to local storage', e);
    }
  }

  private saveSettings() {
    try {
      localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(this.inMemorySettings));
    } catch (e) {
      console.warn('Could not persist settings to local storage', e);
    }
  }

  // Settings
  public async getSettings(): Promise<SystemSettings> {
    return { ...this.inMemorySettings };
  }

  public async updateSettings(newSettings: Partial<SystemSettings>): Promise<SystemSettings> {
    this.inMemorySettings = { ...this.inMemorySettings, ...newSettings };
    this.saveSettings();
    return { ...this.inMemorySettings };
  }

  public isDemoMode(): boolean {
    return this.inMemorySettings.demo_mode;
  }

  public setDemoMode(demo: boolean): void {
    this.inMemorySettings.demo_mode = demo;
    this.saveSettings();
  }

  public isModelConnected(): boolean {
    return this.modelConnected;
  }

  public setModelConnected(connected: boolean): void {
    this.modelConnected = connected;
  }

  // Devices & Network
  public async getDevices(): Promise<Device[]> {
    if (!this.inMemorySettings.demo_mode) {
      try {
        const res = await fetch(`${this.inMemorySettings.api_base_url}/api/v1/devices`);
        if (res.ok) return await res.json();
      } catch {
        console.warn('Failed to fetch devices from live backend, falling back to cached state.');
      }
    }
    return [...mockDevices];
  }

  // Plants
  public async getPlants(): Promise<Plant[]> {
    return this.inMemoryPlants.map((plant) => {
      const recordsForPlant = this.inMemoryRecords.filter((r) => r.plant_id === plant.id);
      const latest = recordsForPlant.length > 0 ? recordsForPlant[recordsForPlant.length - 1] : undefined;
      return {
        ...plant,
        latestReading: latest,
      };
    });
  }

  public async getPlantById(id: number | string): Promise<Plant | null> {
    const numId = typeof id === 'string' ? parseInt(id.replace('plant-', ''), 10) : id;
    const plants = await this.getPlants();
    return plants.find((p) => p.id === numId) || null;
  }

  // Records
  public async getRecords(filter?: {
    plantId?: number | string;
    cluster?: string;
    nodeId?: string;
    status?: string;
    limit?: number;
  }): Promise<SensorRecord[]> {
    let result = [...this.inMemoryRecords];

    if (filter?.plantId !== undefined && filter.plantId !== 'all') {
      const targetPlantId = typeof filter.plantId === 'string' ? parseInt(filter.plantId.replace('plant-', ''), 10) : filter.plantId;
      result = result.filter((r) => r.plant_id === targetPlantId);
    }
    if (filter?.nodeId && filter.nodeId !== 'all') {
      result = result.filter((r) => r.node_id === filter.nodeId);
    }
    if (filter?.status && filter.status !== 'all') {
      result = result.filter((r) => r.status === filter.status);
    }
    if (filter?.cluster && filter.cluster !== 'all') {
      const matchingPlantIds = this.inMemoryPlants.filter((p) => p.cluster === filter.cluster).map((p) => p.id);
      result = result.filter((r) => matchingPlantIds.includes(r.plant_id));
    }

    result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    if (filter?.limit) {
      result = result.slice(0, filter.limit);
    }

    return result;
  }

  public async getLatestRecords(): Promise<SensorRecord[]> {
    const latestMap = new Map<number, SensorRecord>();
    const sorted = [...this.inMemoryRecords].sort(
      (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
    );

    for (const rec of sorted) {
      if (!latestMap.has(rec.plant_id)) {
        latestMap.set(rec.plant_id, rec);
      }
      if (latestMap.size === this.inMemoryPlants.length) break;
    }

    return Array.from(latestMap.values());
  }

  // Sessions
  public async getSessions(): Promise<Session[]> {
    return [...mockSessions];
  }

  // Intelligence & SHAP for Tulsi Stress Clusters
  public async getLocalShap(plantId: number | string): Promise<LocalShapExplanation | null> {
    if (!this.modelConnected) return null;

    const numId = typeof plantId === 'string' ? parseInt(plantId.replace('plant-', ''), 10) : plantId;
    const plant = this.inMemoryPlants.find((p) => p.id === numId) || this.inMemoryPlants[4];
    const latest = plant.latestReading || this.inMemoryRecords[this.inMemoryRecords.length - 1];

    if (plant.id === 5 || plant.id === 7) {
      return {
        ...mockLocalShapExplanation,
        plant_id: plant.id,
        record_id: latest.record_id,
      };
    }

    if (plant.cluster === 'Cluster 3') {
      // Light Stress Tulsi
      return {
        plant_id: plant.id,
        record_id: latest.record_id,
        model_version: 'v1.2.0-jetson',
        target_class: 'Light Stress',
        predicted_probability: 0.94,
        base_value: 0.25,
        model_output: 0.94,
        laymanSummary: `Light is the biggest culprit here. At ${latest.light} lux, the pot is in deep shade (ideal is 2,500–6,000 lux). This lack of light pushed the AI by +59% toward Light Stress.`,
        explanation_note:
          'SHAP shows how each sensor reading influenced the AI model’s prediction. It reveals the exact reasons behind the stress classification.',
        features: [
          {
            name: 'light',
            displayName: 'Light Level (lux)',
            observed_value: latest.light,
            unit: 'lux',
            shap_value: 0.59,
            direction: 'increases_prediction',
            description: `Critically low lighting (${latest.light} lux vs 2,500+ lux target) drives the largest positive contribution to Light Stress.`,
            laymanExplanation: 'Very dim lighting is starving the leaves of sunlight energy.',
          },
          {
            name: 'humidity',
            displayName: 'Air Humidity (%)',
            observed_value: latest.humidity,
            unit: '%',
            shap_value: 0.11,
            direction: 'increases_prediction',
            description: 'Elevated humidity in shaded enclosure reflects slower transpiration.',
            laymanExplanation: 'Moist trapped air confirms water is not evaporating quickly.',
          },
          {
            name: 'soil',
            displayName: 'Soil Moisture (%)',
            observed_value: latest.soil,
            unit: '%',
            shap_value: -0.05,
            direction: 'decreases_prediction',
            description: `Soil moisture (${latest.soil}%) is adequate, proving this is not a drought issue.`,
            laymanExplanation: 'Soil has plenty of water, so underwatering is ruled out.',
          },
          {
            name: 'temperature',
            displayName: 'Temperature (°C)',
            observed_value: latest.temperature,
            unit: '°C',
            shap_value: -0.01,
            direction: 'neutral',
            description: `Room temperature (${latest.temperature}°C) is normal.`,
            laymanExplanation: 'Temperature is comfortable.',
          },
        ],
      };
    }

    if (plant.cluster === 'Cluster 4') {
      // Nutrient Stress Tulsi
      return {
        plant_id: plant.id,
        record_id: latest.record_id,
        model_version: 'v1.2.0-jetson',
        target_class: 'Nutrient Stress',
        predicted_probability: 0.89,
        base_value: 0.25,
        model_output: 0.89,
        laymanSummary: `The pot has adequate water (${latest.soil}%) and plenty of light (${latest.light} lux), but the potting mix has run out of nutrients. The AI recognizes this combination as Nutrient Deficiency.`,
        explanation_note:
          'SHAP shows how each sensor reading influenced the AI model’s prediction. It reveals the exact reasons behind the stress classification.',
        features: [
          {
            name: 'soil',
            displayName: 'Soil & Substrate Moisture (%)',
            observed_value: latest.soil,
            unit: '%',
            shap_value: 0.38,
            direction: 'increases_prediction',
            description: 'Adequate hydration in an inert potting substrate flags macronutrient depletion rather than water stress.',
            laymanExplanation: 'Water is present, but plant food in the soil is exhausted.',
          },
          {
            name: 'light',
            displayName: 'Light Level (lux)',
            observed_value: latest.light,
            unit: 'lux',
            shap_value: -0.15,
            direction: 'decreases_prediction',
            description: `Ample light (${latest.light} lux) confirms lack of sun is not the cause.`,
            laymanExplanation: 'Sunlight is good, ruling out light deficiency.',
          },
          {
            name: 'temperature',
            displayName: 'Temperature (°C)',
            observed_value: latest.temperature,
            unit: '°C',
            shap_value: 0.08,
            direction: 'increases_prediction',
            description: `Stable temperature (${latest.temperature}°C) matches standard room conditions.`,
            laymanExplanation: 'Normal room temperature.',
          },
          {
            name: 'humidity',
            displayName: 'Air Humidity (%)',
            observed_value: latest.humidity,
            unit: '%',
            shap_value: 0.04,
            direction: 'neutral',
            description: 'Canopy humidity remains within typical range.',
            laymanExplanation: 'Air humidity is pleasant.',
          },
        ],
      };
    }

    // Cluster 1 — Healthy Control Tulsi
    return {
      plant_id: plant.id,
      record_id: latest.record_id,
      model_version: 'v1.2.0-jetson',
      target_class: 'Healthy (Control)',
      predicted_probability: 0.98,
      base_value: 0.25,
      model_output: 0.98,
      laymanSummary: `All green! Soil moisture (${latest.soil}%) and sun (${latest.light} lux) are in the ideal zone. The AI model is 98% confident that this Tulsi pot is thriving.`,
      explanation_note:
        'SHAP shows how each sensor reading influenced the AI model’s prediction. It reveals the exact reasons behind the stress classification.',
      features: [
        {
          name: 'soil',
          displayName: 'Soil Moisture (%)',
          observed_value: latest.soil,
          unit: '%',
          shap_value: 0.36,
          direction: 'increases_prediction',
          description: `Optimal root zone hydration (${latest.soil}%) strongly reinforces healthy baseline classification.`,
          laymanExplanation: 'Soil has just the right amount of moisture (not dry, not soggy).',
        },
        {
          name: 'light',
          displayName: 'Light Level (lux)',
          observed_value: latest.light,
          unit: 'lux',
          shap_value: 0.29,
          direction: 'increases_prediction',
          description: `Generous sunlight (${latest.light} lux) fuels active photosynthesis.`,
          laymanExplanation: 'Bright sunlight is giving the leaves plenty of energy.',
        },
        {
          name: 'temperature',
          displayName: 'Temperature (°C)',
          observed_value: latest.temperature,
          unit: '°C',
          shap_value: 0.06,
          direction: 'increases_prediction',
          description: `Comfortable room temperature (${latest.temperature}°C).`,
          laymanExplanation: 'Temperature is just right.',
        },
        {
          name: 'humidity',
          displayName: 'Air Humidity (%)',
          observed_value: latest.humidity,
          unit: '%',
          shap_value: 0.03,
          direction: 'neutral',
          description: `Air humidity at ${latest.humidity}%.`,
          laymanExplanation: 'Room humidity is comfortable.',
        },
      ],
    };
  }

  public async getGlobalShap(): Promise<GlobalShapData | null> {
    if (!this.modelConnected) return null;
    return { ...mockGlobalShapData };
  }

  public async getModelMetrics(): Promise<ModelMetrics | null> {
    if (!this.modelConnected) return null;
    return { ...mockModelMetrics };
  }

  // Recommendations
  public async getRecommendations(): Promise<Recommendation[]> {
    return [...this.inMemoryRecommendations];
  }

  public async updateRecommendationStatus(
    id: string,
    newStatus: RecommendationStatus
  ): Promise<Recommendation | null> {
    const idx = this.inMemoryRecommendations.findIndex((r) => r.id === id);
    if (idx === -1) return null;

    const updated: Recommendation = {
      ...this.inMemoryRecommendations[idx],
      status: newStatus,
      updated_at: new Date().toISOString(),
      resolved_at: newStatus === 'completed' || newStatus === 'dismissed' ? new Date().toISOString() : undefined,
    };

    this.inMemoryRecommendations[idx] = updated;
    this.saveRecommendations();
    return updated;
  }

  // Live simulation adhering strictly to ranges:
  // timestamp: ISO 8601 string, e.g. 2024-01-15T12:00:00
  // session_id: integer >= 1
  // node_id: "1" or "2"
  // sampling_point: 1 to 12
  // plant_id: 1 to 16
  // soil: 0.0 to 100.0
  // temperature: -40.0 to 80.0
  // humidity: 0.0 to 100.0
  // light: 0.0 to 100000.0
  public simulateLiveReadingPush(): SensorRecord {
    const plantIdx = Math.floor(Math.random() * this.inMemoryPlants.length);
    const plant = this.inMemoryPlants[plantIdx];
    const now = new Date();
    const isoTimestamp = now.toISOString().slice(0, 19);

    let newSoil = 55.0;
    let newTemp = 24.0;
    let newHum = 60.0;
    let newLight = 4000.0;

    if (plant.cluster === 'Cluster 1') {
      newSoil = 55.0 + Math.random() * 6.0;
      newTemp = 23.0 + Math.random() * 3.0;
      newHum = 58.0 + Math.random() * 6.0;
      newLight = 4200.0 + Math.random() * 800.0;
    } else if (plant.cluster === 'Cluster 2') {
      // Water stress (soil 14 - 19%)
      newSoil = 15.0 + Math.random() * 3.5;
      newTemp = 26.0 + Math.random() * 3.0;
      newHum = 40.0 + Math.random() * 5.0;
      newLight = 4000.0 + Math.random() * 800.0;
    } else if (plant.cluster === 'Cluster 3') {
      // Light stress (< 500 lux)
      newSoil = 52.0 + Math.random() * 5.0;
      newTemp = 22.0 + Math.random() * 2.0;
      newHum = 62.0 + Math.random() * 6.0;
      newLight = 320.0 + Math.random() * 160.0;
    } else {
      // Nutrient stress
      newSoil = 48.0 + Math.random() * 6.0;
      newTemp = 23.0 + Math.random() * 2.5;
      newHum = 54.0 + Math.random() * 5.0;
      newLight = 3800.0 + Math.random() * 600.0;
    }

    // Strict boundary clamps
    newSoil = Math.max(0, Math.min(100, Number(newSoil.toFixed(1))));
    newTemp = Math.max(-40, Math.min(80, Number(newTemp.toFixed(1))));
    newHum = Math.max(0, Math.min(100, Number(newHum.toFixed(1))));
    newLight = Math.max(0, Math.min(100000, Number(newLight.toFixed(1))));

    const maxSessionId = Math.max(...this.inMemoryRecords.map((r) => r.session_id), 8);

    const newRecord: SensorRecord = {
      record_id: `REC-${1000 + this.inMemoryRecords.length + 1}`,
      timestamp: isoTimestamp,
      session_id: maxSessionId,
      node_id: plant.nodeId,
      sampling_point: plant.samplingPoint,
      plant_id: plant.id,
      soil: newSoil,
      temperature: newTemp,
      humidity: newHum,
      light: newLight,
      received_at: new Date(now.getTime() + 150).toISOString().slice(0, 19),
      accepted_at: new Date(now.getTime() + 150).toISOString().slice(0, 19),
      status: 'accepted',
      rssi: plant.nodeId === '1' ? -56 : -62,
      crc_ok: true,
    };

    this.inMemoryRecords.push(newRecord);
    if (this.inMemoryRecords.length > 500) {
      this.inMemoryRecords.shift();
    }

    plant.latestReading = newRecord;
    return newRecord;
  }
}

export const apiClient = new ApiClient();
