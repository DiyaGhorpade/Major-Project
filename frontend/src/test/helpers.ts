import { vi } from 'vitest';

export interface MockRoute {
  status?: number;
  body?: unknown;
  headers?: Record<string, string>;
}

/** Queue of responses per `METHOD /path` (last entry repeats). A function entry is called per request. */
export type RouteTable = Record<string, MockRoute | MockRoute[] | ((init?: RequestInit) => MockRoute | Promise<MockRoute>)>;

export function jsonResponse(route: MockRoute): Response {
  return new Response(route.body === undefined ? null : JSON.stringify(route.body), {
    status: route.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(route.headers ?? {}) },
  });
}

/** Replace global fetch with a router keyed by "METHOD /api/path" (query string ignored). */
export function mockFetch(routes: RouteTable) {
  const queues = new Map<string, MockRoute[]>();
  const calls: Array<{ key: string; url: string; init?: RequestInit }> = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const method = (init?.method ?? 'GET').toUpperCase();
    const key = `${method} ${url.split('?')[0]}`;
    calls.push({ key, url, init });
    const entry = routes[key];
    if (entry === undefined) return jsonResponse({ status: 404, body: { error: { code: 'not_found', message: `no mock for ${key}`, details: null } } });
    if (typeof entry === 'function') return jsonResponse(await entry(init));
    if (Array.isArray(entry)) {
      const queue = queues.get(key) ?? [...entry];
      queues.set(key, queue);
      const next = queue.length > 1 ? queue.shift()! : queue[0];
      return jsonResponse(next);
    }
    return jsonResponse(entry);
  });
  vi.stubGlobal('fetch', fn);
  return { fn, calls };
}

export const statusBody = (over: Record<string, unknown> = {}) => ({
  runId: 1,
  state: 'idle',
  total: 0,
  processed: 0,
  failed: 0,
  percent: 0,
  currentPlantId: null,
  startedAt: null,
  finishedAt: null,
  message: null,
  mlServiceOk: true,
  modelVersion: '1.0.0',
  readyToRun: true,
  ...over,
});

export const healthBody = (over: Record<string, unknown> = {}) => ({
  status: 'ok',
  db: { ok: true, readings: 5376, plants: 16 },
  mlService: { ok: true, url: 'http://127.0.0.1:8001', detail: 'ok', status: 'ok', modelsLoaded: true, explainabilityAvailable: true },
  modelVersion: '1.0.0',
  runId: 'v1',
  datasetName: 'plant_sensor_data.csv',
  seeded: true,
  ...over,
});
