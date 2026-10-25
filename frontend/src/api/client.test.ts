import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, apiClient } from './client';
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
    expect(await apiClient.getGlobalShap()).toBeNull();
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
    expect(await apiClient.getRecords()).toHaveLength(2);
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
});
