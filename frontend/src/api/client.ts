import type {
  AiInsights,
  Anomaly,
  AnomalySummary,
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

/** Free-text search keys: their value is user input, so "all" is a real search term rather than the select sentinel. */
const FREE_TEXT_KEYS = new Set(['q']);

export function buildQuery(params?: Query): string {
  if (!params) return '';
  const usp = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue;
    if (value === 'all' && !FREE_TEXT_KEYS.has(key)) continue; // 'all' = "no filter" for select controls
    usp.set(key, String(value));
  }
  const text = usp.toString();
  return text ? `?${text}` : '';
}

interface RequestOptions {
  method?: string;
  query?: Query;
  body?: unknown;
  /** Per-request timeout in ms (default `REQUEST_TIMEOUT_MS`). */
  timeoutMs?: number;
}

/** Every request is aborted after this long and fails with ApiError code `timeout`. */
export const REQUEST_TIMEOUT_MS = 10000;

/**
 * One request with a timeout: the timer covers the headers and the body read. Non-2xx answers become ApiErrors,
 * unreachable backends become `network`, and an aborted request becomes `timeout`.
 */
async function request<T>(path: string, opts: RequestOptions, read: (res: Response) => Promise<T>): Promise<T> {
  const timeoutMs = opts.timeoutMs ?? REQUEST_TIMEOUT_MS;
  const controller = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const init: RequestInit = { method: opts.method ?? 'GET', headers: { Accept: 'application/json' }, signal: controller.signal };
  if (opts.body !== undefined) {
    init.headers = { ...init.headers, 'Content-Type': 'application/json' };
    init.body = JSON.stringify(opts.body);
  }
  try {
    const res = await fetch(`${API_BASE}${path}${buildQuery(opts.query)}`, init);
    if (!res.ok) throw await parseError(res);
    return await read(res);
  } catch (err) {
    if (err instanceof ApiError) throw err;
    if (timedOut) throw new ApiError(0, 'timeout', `Request timed out after ${Math.round(timeoutMs / 1000)} s`);
    const reason = err instanceof Error ? err.message : 'request failed';
    throw new ApiError(0, 'network', `Backend not reachable (${reason})`);
  } finally {
    clearTimeout(timer);
  }
}

function getJson<T>(path: string, query?: Query): Promise<T> {
  return request(path, { query }, async (res) => (await res.json()) as T);
}

function sendJson<T>(path: string, method: string, body: unknown): Promise<T> {
  return request(path, { method, body }, async (res) => (await res.json()) as T);
}

function getPage<T>(path: string, query?: Query): Promise<Page<T>> {
  return request(path, { query }, async (res) => {
    const items = (await res.json()) as T[];
    const header = res.headers.get('X-Total-Count');
    const parsed = header === null ? NaN : Number(header);
    return { items, total: Number.isFinite(parsed) ? parsed : items.length };
  });
}

/**
 * GET that answers `null` only for the intended 404s whose error code is in `nullCodes` (e.g. `not_found` for a
 * missing plant, `explanation_unavailable` for "nothing there yet"). The backend answers an unknown URL with the
 * distinct code `route_not_found`, so a wrong path, like every other failure, still throws.
 */
async function getOrNull<T>(path: string, nullCodes: readonly string[], query?: Query): Promise<T | null> {
  try {
    return await getJson<T>(path, query);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404 && nullCodes.includes(err.code)) return null;
    throw err;
  }
}

/** Error codes the backend uses when a plant id does not exist. */
const PLANT_NOT_FOUND = ['not_found'] as const;

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
  /** Case-insensitive text search over title, action, reason, plant name and plant id (server side). */
  q?: string;
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

export interface AnomalyFilter extends PageParams {
  plantId?: number | string;
  sort?: 'recent' | 'score';
  minScore?: number;
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
    q: f.q?.trim() || undefined,
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
    return sendJson<PipelineRunResponse>('/pipeline/run', 'POST', { reset });
  }

  public getPipelineStatus(): Promise<PipelineStatus> {
    return getJson<PipelineStatus>('/pipeline/status');
  }

  // Monitoring
  public getOverview(): Promise<Overview> {
    return getJson<Overview>('/monitoring/overview');
  }

  public getPlantMonitoring(id: number | string, limit?: number): Promise<PlantMonitoring | null> {
    return getOrNull<PlantMonitoring>(`/monitoring/plants/${plantNumber(id)}`, PLANT_NOT_FOUND, { limit });
  }

  // Plants
  public getPlantsPage(params?: PageParams): Promise<Page<Plant>> {
    return getPage<Plant>('/plants', { limit: params?.limit ?? 100, offset: params?.offset });
  }

  public getPlantById(id: number | string): Promise<Plant | null> {
    return getOrNull<Plant>(`/plants/${plantNumber(id)}`, PLANT_NOT_FOUND);
  }

  // Readings and data logs
  public getRecordsPage(filter?: ReadingFilter): Promise<Page<SensorRecord>> {
    return getPage<SensorRecord>('/sensors/readings', readingQuery(filter));
  }

  public getDataLogsPage(filter?: ReadingFilter): Promise<Page<SensorRecord>> {
    return getPage<SensorRecord>('/data-logs', readingQuery(filter));
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

  public async getSessions(): Promise<Session[]> {
    return (await getPage<Session>('/sessions')).items;
  }

  public async getDevices(): Promise<Device[]> {
    return (await getPage<Device>('/devices')).items;
  }

  // AI
  /** 404 `explanation_unavailable` (before a Run) maps to null. */
  public getLocalShap(plantId: number | string): Promise<LocalShapExplanation | null> {
    return getOrNull<LocalShapExplanation>(`/ai/shap/${plantNumber(plantId)}`, ['explanation_unavailable']);
  }

  public async getGlobalShapResult(): Promise<GlobalShapResult> {
    try {
      return { data: await getJson<GlobalShapData>('/ai/shap/global'), unavailable: null };
    } catch (err) {
      if (err instanceof ApiError && err.status === 404 && err.code === 'global_shap_unavailable') {
        const details = err.details as { hint?: unknown } | null;
        const hint = details && typeof details.hint === 'string' ? details.hint : null;
        return { data: null, unavailable: { message: err.message, hint } };
      }
      throw err;
    }
  }

  public getModelMetrics(): Promise<ModelMetrics> {
    return getJson<ModelMetrics>('/models/performance');
  }

  public getInsights(): Promise<AiInsights> {
    return getJson<AiInsights>('/ai/insights');
  }

  public getAnomalies(filter?: AnomalyFilter): Promise<Page<Anomaly>> {
    return getPage<Anomaly>('/ai/anomalies', {
      plantId: filter?.plantId === undefined || filter.plantId === 'all' ? undefined : plantNumber(filter.plantId),
      sort: filter?.sort,
      minScore: filter?.minScore !== undefined && Number.isFinite(filter.minScore) ? filter.minScore : undefined,
      limit: filter?.limit,
      offset: filter?.offset,
    });
  }

  public getAnomalySummary(): Promise<AnomalySummary> {
    return getJson<AnomalySummary>('/ai/anomalies/summary');
  }

  // Recommendations
  public getRecommendationsPage(filter?: RecommendationFilter): Promise<Page<Recommendation>> {
    return getPage<Recommendation>('/recommendations', recommendationQuery(filter));
  }

  /** Rejects with an `ApiError` (status 409, code `conflict`) when reopening would duplicate an active one. */
  public async updateRecommendationStatus(id: string, status: UserRecommendationStatus): Promise<Recommendation> {
    return sendJson<Recommendation>(`/recommendations/${encodeURIComponent(id)}`, 'PATCH', { status });
  }

  // Settings
  public getSettings(): Promise<SystemSettings> {
    return getJson<SystemSettings>('/settings');
  }

  public async updateSettings(patch: Partial<SystemSettings>): Promise<SystemSettings> {
    return sendJson<SystemSettings>('/settings', 'PUT', patch);
  }
}

export const apiClient = new ApiClient();
