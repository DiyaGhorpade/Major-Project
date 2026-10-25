import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { usePipelineMachine } from './usePipeline';
import { mockFetch, statusBody } from '../test/helpers';

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
});
