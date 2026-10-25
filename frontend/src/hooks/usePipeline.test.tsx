import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { usePipelineMachine } from './usePipeline';
import { jsonResponse, mockFetch, statusBody } from '../test/helpers';
import { REQUEST_TIMEOUT_MS } from '../api/client';

const tick = (ms: number) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe('usePipelineMachine', () => {
  it('goes idle -> queued/running -> completed and bumps the data version', async () => {
    const onChanged = vi.fn();
    mockFetch({
      'POST /api/pipeline/run': { status: 202, body: { runId: 1, state: 'queued' } },
      'GET /api/pipeline/status': [
        statusBody({ state: 'idle', runId: null }),
        statusBody({ state: 'queued', total: 5376 }),
        statusBody({ state: 'running', total: 5376, processed: 256, percent: 4.8 }),
        statusBody({ state: 'running', total: 5376, processed: 512, percent: 9.5 }),
        statusBody({ state: 'running', total: 5376, processed: 1280, percent: 23.8 }),
        statusBody({ state: 'completed', total: 5376, processed: 5376, percent: 100, message: 'Completed: 5376 predicted' }),
      ].map((body) => ({ body })),
    });

    const { result } = renderHook(() => usePipelineMachine(onChanged));
    await tick(0);
    expect(result.current.state).toBe('idle');
    expect(result.current.hasCompletedRun).toBe(false);
    expect(result.current.isActive).toBe(false);

    await act(async () => {
      await result.current.run();
    });
    // The poll right after POST sees "queued".
    expect(result.current.state).toBe('queued');
    expect(result.current.isActive).toBe(true);

    await tick(700);
    expect(result.current.state).toBe('running');
    expect(result.current.processed).toBe(256);
    expect(result.current.percent).toBeCloseTo(4.8);
    expect(onChanged).toHaveBeenCalledTimes(1); // first progress

    await tick(700); // 512: progressed, but < 2 s since the last bump
    expect(result.current.processed).toBe(512);
    expect(onChanged).toHaveBeenCalledTimes(1);

    await tick(700); // 1280: throttle window passed
    expect(result.current.processed).toBe(1280);
    expect(onChanged).toHaveBeenCalledTimes(2);

    await tick(700);
    expect(result.current.state).toBe('completed');
    expect(result.current.hasCompletedRun).toBe(true);
    expect(result.current.isActive).toBe(false);
    expect(result.current.message).toBe('Completed: 5376 predicted');
    expect(onChanged).toHaveBeenCalledTimes(3); // completion always bumps
  });

  it('ends in failed with the cause when the run fails', async () => {
    const onChanged = vi.fn();
    mockFetch({
      'POST /api/pipeline/run': { status: 202, body: { runId: 3, state: 'queued' } },
      'GET /api/pipeline/status': [
        { body: statusBody({ state: 'idle', runId: null }) },
        { body: statusBody({ runId: 3, state: 'running', total: 5376, processed: 100 }) },
        { body: statusBody({ runId: 3, state: 'failed', total: 5376, processed: 100, mlServiceOk: false, message: 'ML service not reachable at :8001' }) },
      ],
    });
    const { result } = renderHook(() => usePipelineMachine(onChanged));
    await tick(0);
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.state).toBe('running');
    await tick(700);
    expect(result.current.state).toBe('failed');
    expect(result.current.message).toBe('ML service not reachable at :8001');
    expect(result.current.isActive).toBe(false);
    expect(result.current.hasCompletedRun).toBe(false);
    expect(onChanged).toHaveBeenCalled();
  });

  it('returns to idle with an error when the run request cannot reach the backend', async () => {
    mockFetch({ 'GET /api/pipeline/status': { body: statusBody({ state: 'idle', runId: null }) } });
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(0);
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await act(async () => {
      await result.current.run();
    });
    expect(result.current.state).toBe('idle');
    expect(result.current.error?.code).toBe('network');
  });

  it('picks up a run that is already in progress on mount and polls fast', async () => {
    const { calls } = mockFetch({
      'GET /api/pipeline/status': { body: statusBody({ state: 'running', total: 5376, processed: 10, percent: 0.2 }) },
    });
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(0);
    expect(result.current.state).toBe('running');
    expect(result.current.isActive).toBe(true);
    const before = calls.length;
    await tick(2100);
    expect(calls.length - before).toBeGreaterThanOrEqual(2); // ~700 ms cadence while active
  });

  it('polls slowly when nothing is running', async () => {
    const { calls } = mockFetch({ 'GET /api/pipeline/status': { body: statusBody({ state: 'completed', total: 10, processed: 10, percent: 100 }) } });
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(0);
    expect(result.current.hasCompletedRun).toBe(true);
    const before = calls.length;
    await tick(2100);
    expect(calls.length).toBe(before);
    await tick(3500);
    expect(calls.length).toBe(before + 1);
  });

  /** fetch whose handler may return a never-settling promise that rejects on abort (like a hung server). */
  const hangUntilAborted = (init?: RequestInit) =>
    new Promise<Response>((_res, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
    });

  it('keeps polling after a status request times out', async () => {
    let statusCalls = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (String(input).endsWith('/pipeline/status')) {
          statusCalls += 1;
          if (statusCalls === 1) return hangUntilAborted(init);
        }
        return Promise.resolve(jsonResponse({ body: statusBody({ state: 'idle', runId: null }) }));
      }),
    );
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(REQUEST_TIMEOUT_MS + 1);
    expect(result.current.error?.code).toBe('timeout');
    expect(statusCalls).toBe(1);

    await tick(5100); // the idle cadence reschedules even though the previous request failed
    expect(statusCalls).toBe(2);
    expect(result.current.error).toBeNull();
    expect(result.current.state).toBe('idle');
  });

  it('re-enables Run and shows the error when starting a run times out', async () => {
    let postCalls = 0;
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        if (init?.method === 'POST') {
          postCalls += 1;
          if (postCalls === 1) return hangUntilAborted(init);
          return Promise.resolve(jsonResponse({ status: 202, body: { runId: 2, state: 'queued' } }));
        }
        const body = postCalls >= 2 ? statusBody({ state: 'queued', total: 10 }) : statusBody({ state: 'idle', runId: null });
        return Promise.resolve(jsonResponse({ body }));
      }),
    );
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(0);

    let first: Promise<void> = Promise.resolve();
    act(() => {
      first = result.current.run();
    });
    expect(result.current.isActive).toBe(true); // "starting" while the POST is pending
    await tick(REQUEST_TIMEOUT_MS + 1);
    await act(async () => first);
    expect(result.current.error?.code).toBe('timeout');
    expect(result.current.isActive).toBe(false);
    expect(result.current.state).toBe('idle');

    await act(async () => {
      await result.current.run(); // Run works again
    });
    expect(postCalls).toBe(2);
    expect(result.current.error).toBeNull();
    expect(result.current.state).toBe('queued');
  });

  it('reports no stale totals while a re-run is starting', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((_input: RequestInfo | URL, init?: RequestInit) =>
        init?.method === 'POST'
          ? hangUntilAborted(init)
          : Promise.resolve(jsonResponse({ body: statusBody({ state: 'completed', total: 5376, processed: 5376, percent: 100 }) })),
      ),
    );
    const { result } = renderHook(() => usePipelineMachine(vi.fn()));
    await tick(0);
    expect(result.current.total).toBe(5376);

    act(() => {
      void result.current.run();
    });
    expect(result.current.state).toBe('queued');
    expect(result.current.total).toBe(0);
    expect(result.current.processed).toBe(0);
    await tick(REQUEST_TIMEOUT_MS + 1);
  });
});
