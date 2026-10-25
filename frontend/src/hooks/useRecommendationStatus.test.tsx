import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useRecommendationStatus } from './useRecommendationStatus';
import { mockFetch } from '../test/helpers';

afterEach(() => vi.unstubAllGlobals());

describe('useRecommendationStatus', () => {
  it('blocks a second PATCH for the same id while the first is pending', async () => {
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    const { calls } = mockFetch({
      'PATCH /api/recommendations/rec-1': async () => {
        await gate;
        return { body: { id: 'rec-1', status: 'completed' } };
      },
    });
    const { result } = renderHook(() => useRecommendationStatus());

    let first: Promise<unknown> = Promise.resolve();
    act(() => {
      first = result.current.update('rec-1', 'completed');
    });
    expect(result.current.isPending('rec-1')).toBe(true);
    expect(result.current.isPending('rec-2')).toBe(false);

    let second: unknown = 'unset';
    await act(async () => {
      second = await result.current.update('rec-1', 'dismissed');
    });
    expect(second).toBeNull();
    expect(calls.filter((c) => c.key.startsWith('PATCH'))).toHaveLength(1);

    await act(async () => {
      release();
      await first;
    });
    expect(result.current.isPending('rec-1')).toBe(false);
    // A new change is allowed again once the first finished.
    await act(async () => {
      await result.current.update('rec-1', 'new');
    });
    expect(calls.filter((c) => c.key.startsWith('PATCH'))).toHaveLength(2);
  });

  it('surfaces the 409 and calls onFailure (rollback) for the latest request', async () => {
    mockFetch({
      'PATCH /api/recommendations/rec-9': { status: 409, body: { error: { code: 'conflict', message: 'duplicate', details: { activeId: 'rec-2' } } } },
    });
    const onFailure = vi.fn();
    const { result } = renderHook(() => useRecommendationStatus());
    await act(async () => {
      await result.current.update('rec-9', 'new', onFailure);
    });
    await waitFor(() => expect(result.current.error?.status).toBe(409));
    expect(onFailure).toHaveBeenCalledTimes(1);
    expect(result.current.isPending('rec-9')).toBe(false);
    act(() => result.current.clearError());
    expect(result.current.error).toBeNull();
  });
});
