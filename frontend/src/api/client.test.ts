import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, REQUEST_TIMEOUT_MS, apiClient } from './client';
import { mockFetch } from '../test/helpers';

afterEach(() => {
  vi.unstubAllGlobals();
});

const envelope = (code: string, message: string, details: unknown = null) => ({ error: { code, message, details } });

describe('api client', () => {
  it('parses the error envelope into an ApiError', async () => {
    mockFetch({ 'GET /api/sessions': { status: 422, body: envelope('validation_error', 'Request validation failed', [{ loc: ['query', 'limit'] }]) } });
    const err = await apiClient.getSessions().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(422);
    expect(err.code).toBe('validation_error');
    expect(err.message).toBe('Request validation failed');
    expect(err.details).toEqual([{ loc: ['query', 'limit'] }]);
  });

  it('falls back to an http_<status> code for non-envelope errors', async () => {
    mockFetch({ 'GET /api/devices': { status: 502, body: 'bad gateway' } });
    const err = await apiClient.getDevices().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(502);
    expect(err.code).toBe('http_502');
  });

  it('turns a network failure into ApiError code "network"', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    const err = await apiClient.getHealth().catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.code).toBe('network');
    expect(err.isNetwork).toBe(true);
  });

  it('maps 404 to null for plant, monitoring and local SHAP lookups', async () => {
    mockFetch({
      'GET /api/plants/99': { status: 404, body: envelope('not_found', 'plant 99 not found') },
      'GET /api/monitoring/plants/99': { status: 404, body: envelope('not_found', 'plant 99 not found') },
      'GET /api/ai/shap/5': { status: 404, body: envelope('explanation_unavailable', 'none', { hint: 'Press Run' }) },
    });
    expect(await apiClient.getPlantById(99)).toBeNull();
    expect(await apiClient.getPlantMonitoring(99, 10)).toBeNull();
    expect(await apiClient.getLocalShap(5)).toBeNull();
  });

  it('does not swallow non-404 errors when a lookup returns null on 404', async () => {
    mockFetch({ 'GET /api/plants/1': { status: 500, body: envelope('internal_error', 'boom') } });
    await expect(apiClient.getPlantById(1)).rejects.toMatchObject({ status: 500, code: 'internal_error' });
  });

  it('exposes the global SHAP hint on 404', async () => {
    mockFetch({
      'GET /api/ai/shap/global': { status: 404, body: envelope('global_shap_unavailable', 'no report', { hint: 'python scripts/generate_explanations.py' }) },
    });
    const result = await apiClient.getGlobalShapResult();
    expect(result.data).toBeNull();
    expect(result.unavailable?.hint).toBe('python scripts/generate_explanations.py');
  });

  it('reads X-Total-Count for paged getters and falls back to the item count', async () => {
    mockFetch({
      'GET /api/sensors/readings': { body: [{ record_id: 'R-1' }, { record_id: 'R-2' }], headers: { 'X-Total-Count': '5376' } },
      'GET /api/data-logs': { body: [{ record_id: 'R-1' }] },
    });
    const page = await apiClient.getRecordsPage({ limit: 2 });
    expect(page.total).toBe(5376);
    expect(page.items).toHaveLength(2);
    const noHeader = await apiClient.getDataLogsPage();
    expect(noHeader.total).toBe(1);
  });

  it('builds filters, dropping "all" and empty values', async () => {
    const { calls } = mockFetch({ 'GET /api/data-logs': { body: [], headers: { 'X-Total-Count': '0' } } });
    await apiClient.getDataLogsPage({ plantId: 'plant-5', nodeId: 'all', status: 'flagged', q: '', limit: 15, offset: 30 });
    const url = new URL(calls[0].url, 'http://x');
    expect(url.searchParams.get('plantId')).toBe('5');
    expect(url.searchParams.get('status')).toBe('flagged');
    expect(url.searchParams.get('limit')).toBe('15');
    expect(url.searchParams.get('offset')).toBe('30');
    expect(url.searchParams.has('nodeId')).toBe(false);
    expect(url.searchParams.has('q')).toBe(false);
  });

  it('builds the anomaly query: sort, minScore, plant and paging; drops "all" and non-finite scores', async () => {
    const { calls } = mockFetch({
      'GET /api/ai/anomalies': { body: [{ id: 'anom-1' }], headers: { 'X-Total-Count': '102' } },
      'GET /api/ai/anomalies/summary': { body: { flagged: 102, plants: [] } },
    });
    const page = await apiClient.getAnomalies({ plantId: 'plant-14', sort: 'score', minScore: 0.05, limit: 15, offset: 30 });
    expect(page.total).toBe(102);
    const url = new URL(calls[0].url, 'http://x');
    expect(url.pathname).toBe('/api/ai/anomalies');
    expect(Object.fromEntries(url.searchParams)).toEqual({ plantId: '14', sort: 'score', minScore: '0.05', limit: '15', offset: '30' });
    await apiClient.getAnomalies({ plantId: 'all', minScore: 0 });
    expect(Object.fromEntries(new URL(calls[1].url, 'http://x').searchParams)).toEqual({ minScore: '0' });
    await apiClient.getAnomalies({ minScore: Number.NaN });
    expect(new URL(calls[2].url, 'http://x').searchParams.has('minScore')).toBe(false);
    expect((await apiClient.getAnomalySummary()).flagged).toBe(102);
    expect(calls[3].url).toContain('/api/ai/anomalies/summary');
  });

  it('sends a free-text search for "all" but still drops the "all" select sentinel', async () => {
    const { calls } = mockFetch({
      'GET /api/recommendations': { body: [], headers: { 'X-Total-Count': '0' } },
      'GET /api/data-logs': { body: [], headers: { 'X-Total-Count': '0' } },
    });
    await apiClient.getRecommendationsPage({ q: 'all' });
    await apiClient.getDataLogsPage({ nodeId: 'all', q: 'all' });
    for (const call of calls) expect(new URL(call.url, 'http://x').searchParams.get('q')).toBe('all');
    expect(new URL(calls[1].url, 'http://x').searchParams.has('nodeId')).toBe(false);
  });

  it('builds the server CSV url from the same filters without paging', () => {
    const url = new URL(apiClient.dataLogsCsvUrl({ plantId: 3, status: 'flagged', limit: 15, offset: 15 }), 'http://x');
    expect(url.pathname).toBe('/api/data-logs');
    expect(url.searchParams.get('format')).toBe('csv');
    expect(url.searchParams.get('plantId')).toBe('3');
    expect(url.searchParams.has('limit')).toBe(false);
    expect(url.searchParams.has('offset')).toBe(false);
  });

  it('surfaces a 409 conflict when updating a recommendation', async () => {
    mockFetch({
      'PATCH /api/recommendations/rec-3': { status: 409, body: envelope('conflict', 'another active recommendation exists', { activeId: 'rec-9' }) },
    });
    const err = await apiClient.updateRecommendationStatus('rec-3', 'new').catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err.status).toBe(409);
    expect(err.code).toBe('conflict');
    expect(err.details).toEqual({ activeId: 'rec-9' });
  });

  it('posts the reset flag when starting a run', async () => {
    const { calls } = mockFetch({ 'POST /api/pipeline/run': { status: 202, body: { runId: 2, state: 'queued' } } });
    const res = await apiClient.runPipeline(true);
    expect(res).toEqual({ runId: 2, state: 'queued' });
    expect(calls[0].init?.body).toBe(JSON.stringify({ reset: true }));
  });

  it('only maps the intended 404 codes to null; any other 404 is an error', async () => {
    mockFetch({
      'GET /api/plants/7': { status: 404, body: envelope('route_not_found', 'No such route') },
      'GET /api/monitoring/plants/7': { status: 404, body: 'plain not found' },
      'GET /api/ai/shap/7': { status: 404, body: envelope('not_found', 'unknown plant') },
      'GET /api/ai/shap/global': { status: 404, body: envelope('not_found', 'wrong route') },
    });
    await expect(apiClient.getPlantById(7)).rejects.toMatchObject({ status: 404, code: 'route_not_found' });
    await expect(apiClient.getPlantMonitoring(7)).rejects.toMatchObject({ status: 404, code: 'http_404' });
    await expect(apiClient.getLocalShap(7)).rejects.toMatchObject({ status: 404, code: 'not_found' });
    await expect(apiClient.getGlobalShapResult()).rejects.toMatchObject({ status: 404, code: 'not_found' });
  });

  describe('request timeout', () => {
    afterEach(() => vi.useRealTimers());

    const hangingFetch = () =>
      vi.fn((_url: RequestInfo | URL, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')));
        }),
      );

    it('aborts a hanging request and fails with ApiError code "timeout"', async () => {
      vi.useFakeTimers();
      vi.stubGlobal('fetch', hangingFetch());
      const pending = apiClient.getHealth().catch((e) => e);
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
      const err = await pending;
      expect(err).toBeInstanceOf(ApiError);
      expect(err.code).toBe('timeout');
      expect(err.isNetwork).toBe(false);
      expect(err.message).toMatch(/timed out after 10 s/);
    });

    it('also times out POST requests (a stuck run start) and does not time out fast answers', async () => {
      vi.useFakeTimers();
      vi.stubGlobal('fetch', hangingFetch());
      const run = apiClient.runPipeline().catch((e) => e);
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS + 1);
      expect((await run).code).toBe('timeout');

      mockFetch({ 'GET /api/health': { body: { status: 'ok' } } });
      expect(await apiClient.getHealth()).toEqual({ status: 'ok' });
      await vi.advanceTimersByTimeAsync(REQUEST_TIMEOUT_MS * 2); // the finished request's timer was cleared
    });
  });
});
