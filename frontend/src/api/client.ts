import type {
  AiInsights,
  AnomalyRecord,
  ApiErrorEnvelope,
  Device,
  GlobalShapData,
  GlobalShapUnavailable,
  HealthResponse,
  LocalShapExplanation,
  ModelMetrics,
  Overview,
  Page,
  PipelineRunResponse,
  PipelineStatus,
  Plant,
  PlantMonitoring,
  Prediction,
  Recommendation,
  RecommendationCategory,
  RecommendationPriority,
  RecommendationSource,
  RecommendationStatus,
  SensorRecord,
  Session,
  SystemSettings,
  UserRecommendationStatus,
} from '../types';

/** Relative on purpose: the Vite dev/preview server proxies `/api` to the backend. */
export const API_BASE = '/api';

/** Error thrown for every failed request. `code` is `network` when the backend could not be reached. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: unknown;

  constructor(status: number, code: string, message: string, details: unknown = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }

  get isNetwork(): boolean {
    return this.code === 'network';
  }
}

export function toApiError(err: unknown): ApiError {
  if (err instanceof ApiError) return err;
  const message = err instanceof Error ? err.message : String(err);
  return new ApiError(0, 'unknown', message);
}

function isEnvelope(body: unknown): body is ApiErrorEnvelope {
  if (typeof body !== 'object' || body === null || !('error' in body)) return false;
  const e = (body as { error: unknown }).error;
  return typeof e === 'object' && e !== null && typeof (e as { message?: unknown }).message === 'string';
}

async function parseError(res: Response): Promise<ApiError> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (isEnvelope(body)) {
    const { code, message, details } = body.error;
    return new ApiError(res.status, String(code ?? `http_${res.status}`), message, details ?? null);
  }
  return new ApiError(res.status, `http_${res.status}`, res.statusText || `Request failed (${res.status})`, body);
}

type Query = Record<string, string | number | boolean | null | undefined>;

export function buildQuery(params?: Query): string {
  if (!params) return '';
  const usp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '' || value === 'all') continue;
    usp.set(key, String(value));
  }
  const text = usp.toString();
  return text ? `?${text}` : '';
}

interface RequestOptions {
  method?: string;
  query?: Query;
  body?: unknown;
}

async function send(path: string, opts: RequestOptions = {}): Promise<Response> {
  const init: RequestInit = { method: opts.method ?? 'GET', headers: { Accept: 'application/json' } };
  if (opts.body !== undefined) {
    init.headers = { ...init.headers, 'Content-Type': 'application/json' };
    init.body = JSON.stringify(opts.body);
  }
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}${buildQuery(opts.query)}`, init);
  } catch (err) {
    const reason = err instanceof Error ? err.message : 'request failed';
    throw new ApiError(0, 'network', `Backend not reachable (${reason})`);
  }
  if (!res.ok) throw await parseError(res);
  return res;
}

async function getJson<T>(path: string, query?: Query): Promise<T> {
  const res = await send(path, { query });
  return (await res.json()) as T;
}

async function getPage<T>(path: string, query?: Query): Promise<Page<T>> {
  const res = await send(path, { query });
  const items = (await res.json()) as T[];
  const header = res.headers.get('X-Total-Count');
  const parsed = header === null ? NaN : Number(header);
  return { items, total: Number.isFinite(parsed) ? parsed : items.length };
}

/** GET that answers `null` on 404 (any other failure still throws). */
async function getOrNull<T>(path: string, query?: Query): Promise<T | null> {
  try {
    return await getJson<T>(path, query);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

function plantNumber(id: number | string): number {
  return typeof id === 'string' ? parseInt(id.replace('plant-', ''), 10) : id;
}

export interface PageParams {
  limit?: number;
  offset?: number;
}

/** Filters shared by `/sensors/readings` and `/data-logs`. `'all'` and empty values are dropped. */
export interface ReadingFilter extends PageParams {
  plantId?: number | string;
  cluster?: string;
  nodeId?: string;
  status?: string;
  sessionId?: number;
  from?: string;
  to?: string;
  q?: string;
  order?: 'asc' | 'desc';
}

export interface RecommendationFilter extends PageParams {
  plantId?: number | string;
  status?: RecommendationStatus;
  priority?: RecommendationPriority;
  category?: RecommendationCategory;
  source?: RecommendationSource;
  includeSuperseded?: boolean;
}

export interface PredictionFilter extends PageParams {
  plantId?: number | string;
  status?: 'ok' | 'unavailable' | 'rejected';
  severity?: 'healthy' | 'low' | 'moderate' | 'high';
  isAnomaly?: boolean;
  from?: string;
  to?: string;
}

export interface GlobalShapResult {
  data: GlobalShapData | null;
  /** Set when the report has not been generated yet; `hint` is the command to run. */
  unavailable: GlobalShapUnavailable | null;
}

function readingQuery(f: ReadingFilter = {}): Query {
  return {
    plantId: f.plantId === undefined || f.plantId === 'all' ? undefined : plantNumber(f.plantId),
    cluster: f.cluster,
    nodeId: f.nodeId,
    status: f.status,
    sessionId: f.sessionId,
    from: f.from,
    to: f.to,
    q: f.q,
    order: f.order,
    limit: f.limit,
    offset: f.offset,
  };
}

function recommendationQuery(f: RecommendationFilter = {}): Query {
  return {
    plantId: f.plantId === undefined || f.plantId === 'all' ? undefined : plantNumber(f.plantId),
    status: f.status,
    priority: f.priority,
    category: f.category,
    source: f.source,
    includeSuperseded: f.includeSuperseded,
    limit: f.limit,
    offset: f.offset,
  };
}

class ApiClient {
  // Health & pipeline
  public getHealth(): Promise<HealthResponse> {
    return getJson<HealthResponse>('/health');
  }

  public async runPipeline(reset = false): Promise<PipelineRunResponse> {
    const res = await send('/pipeline/run', { method: 'POST', body: { reset } });
    return (await res.json()) as PipelineRunResponse;
  }

  public getPipelineStatus(): Promise<PipelineStatus> {
    return getJson<PipelineStatus>('/pipeline/status');
  }

  // Monitoring
  public getOverview(): Promise<Overview> {
    return getJson<Overview>('/monitoring/overview');
  }

  public getPlantMonitoring(id: number | string, limit?: number): Promise<PlantMonitoring | null> {
    return getOrNull<PlantMonitoring>(`/monitoring/plants/${plantNumber(id)}`, { limit });
  }

  // Plants
  public getPlantsPage(params?: PageParams): Promise<Page<Plant>> {
    return getPage<Plant>('/plants', { limit: params?.limit ?? 100, offset: params?.offset });
  }

  public async getPlants(): Promise<Plant[]> {
    return (await this.getPlantsPage()).items;
  }

  public getPlantById(id: number | string): Promise<Plant | null> {
    return getOrNull<Plant>(`/plants/${plantNumber(id)}`);
  }

  // Readings and data logs
  public getRecordsPage(filter?: ReadingFilter): Promise<Page<SensorRecord>> {
    return getPage<SensorRecord>('/sensors/readings', readingQuery(filter));
  }

  public async getRecords(filter?: ReadingFilter): Promise<SensorRecord[]> {
    return (await this.getRecordsPage(filter)).items;
  }

  public getDataLogsPage(filter?: ReadingFilter): Promise<Page<SensorRecord>> {
    return getPage<SensorRecord>('/data-logs', readingQuery(filter));
  }

  public async getDataLogs(filter?: ReadingFilter): Promise<SensorRecord[]> {
    return (await this.getDataLogsPage(filter)).items;
  }

  /** URL of the server-side CSV export for the same filters (page params are ignored by the server). */
  public dataLogsCsvUrl(filter?: ReadingFilter): string {
    return `${API_BASE}/data-logs${buildQuery({ ...readingQuery(filter), limit: undefined, offset: undefined, format: 'csv' })}`;
  }

  // Predictions, sessions, devices
  public getPredictionsPage(filter?: PredictionFilter): Promise<Page<Prediction>> {
    return getPage<Prediction>('/predictions', {
      plantId: filter?.plantId === undefined || filter.plantId === 'all' ? undefined : plantNumber(filter.plantId),
      status: filter?.status,
      severity: filter?.severity,
      isAnomaly: filter?.isAnomaly,
      from: filter?.from,
      to: filter?.to,
      limit: filter?.limit,
      offset: filter?.offset,
    });
  }

  public async getPredictions(filter?: PredictionFilter): Promise<Prediction[]> {
    return (await this.getPredictionsPage(filter)).items;
  }

  public getSessionsPage(params?: PageParams): Promise<Page<Session>> {
    return getPage<Session>('/sessions', { ...params });
  }

  public async getSessions(): Promise<Session[]> {
    return (await this.getSessionsPage()).items;
  }

  public getDevicesPage(params?: PageParams): Promise<Page<Device>> {
    return getPage<Device>('/devices', { ...params });
  }

  public async getDevices(): Promise<Device[]> {
    return (await this.getDevicesPage()).items;
  }

  // AI
  /** 404 `explanation_unavailable` (before a Run) maps to null. */
  public getLocalShap(plantId: number | string): Promise<LocalShapExplanation | null> {
    return getOrNull<LocalShapExplanation>(`/ai/shap/${plantNumber(plantId)}`);
  }

  public async getGlobalShapResult(): Promise<GlobalShapResult> {
    try {
      return { data: await getJson<GlobalShapData>('/ai/shap/global'), unavailable: null };
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        const details = err.details as { hint?: unknown } | null;
        const hint = details && typeof details.hint === 'string' ? details.hint : null;
        return { data: null, unavailable: { message: err.message, hint } };
      }
      throw err;
    }
  }

  public async getGlobalShap(): Promise<GlobalShapData | null> {
    return (await this.getGlobalShapResult()).data;
  }

  public getModelMetrics(): Promise<ModelMetrics> {
    return getJson<ModelMetrics>('/models/performance');
  }

  public getInsights(): Promise<AiInsights> {
    return getJson<AiInsights>('/ai/insights');
  }

  public getAnomaliesPage(params?: PageParams & { plantId?: number | string }): Promise<Page<AnomalyRecord>> {
    return getPage<AnomalyRecord>('/ai/anomalies', {
      plantId: params?.plantId === undefined || params.plantId === 'all' ? undefined : plantNumber(params.plantId),
      limit: params?.limit,
      offset: params?.offset,
    });
  }

  public async getAnomalies(params?: PageParams & { plantId?: number | string }): Promise<AnomalyRecord[]> {
    return (await this.getAnomaliesPage(params)).items;
  }

  // Recommendations
  public getRecommendationsPage(filter?: RecommendationFilter): Promise<Page<Recommendation>> {
    return getPage<Recommendation>('/recommendations', recommendationQuery(filter));
  }

  public async getRecommendations(filter?: RecommendationFilter): Promise<Recommendation[]> {
    return (await this.getRecommendationsPage(filter)).items;
  }

  /** Rejects with an `ApiError` (status 409, code `conflict`) when reopening would duplicate an active one. */
  public async updateRecommendationStatus(id: string, status: UserRecommendationStatus): Promise<Recommendation> {
    const res = await send(`/recommendations/${encodeURIComponent(id)}`, { method: 'PATCH', body: { status } });
    return (await res.json()) as Recommendation;
  }

  // Settings
  public getSettings(): Promise<SystemSettings> {
    return getJson<SystemSettings>('/settings');
  }

  public async updateSettings(patch: Partial<SystemSettings>): Promise<SystemSettings> {
    const res = await send('/settings', { method: 'PUT', body: patch });
    return (await res.json()) as SystemSettings;
  }
}

export const apiClient = new ApiClient();
