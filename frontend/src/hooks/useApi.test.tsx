import { describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { type ReactNode, useCallback, useEffect, useState } from 'react';
import { useAsync } from './useApi';
import { DataVersionContext } from './dataVersion';
import { ApiError } from '../api/client';

describe('useAsync', () => {
  it('loads, keeps stale data while refetching on a data-version bump, and reports errors', async () => {
    const version = 0;
    const fetcher = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce('first')
      .mockImplementationOnce(() => new Promise((resolve) => setTimeout(() => resolve('second'), 150)))
      .mockRejectedValueOnce(new ApiError(503, 'service_unavailable', 'down'));

    const bumpRef = { current: () => undefined as void };
    const Wrapper = ({ children }: { children: ReactNode }) => {
      const [v, setV] = useState(version);
      const bumpDataVersion = useCallback(() => setV((n) => n + 1), []);
      useEffect(() => {
        bumpRef.current = bumpDataVersion;
      }, [bumpDataVersion]);
      return <DataVersionContext.Provider value={{ dataVersion: v, bumpDataVersion }}>{children}</DataVersionContext.Provider>;
    };
    const bump = () => bumpRef.current();

    const { result } = renderHook(() => useAsync(fetcher, ['k']), { wrapper: Wrapper });
    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.data).toBe('first'));
    expect(result.current.loading).toBe(false);

    act(() => bump());
    await waitFor(() => expect(result.current.refreshing).toBe(true));
    expect(result.current.data).toBe('first'); // stale data stays visible
    expect(result.current.loading).toBe(false);
    await waitFor(() => expect(result.current.data).toBe('second'));
    expect(result.current.refreshing).toBe(false);

    act(() => bump());
    await waitFor(() => expect(result.current.error?.code).toBe('service_unavailable'));
    expect(result.current.data).toBe('second');
  });

  it('ignores a slower, older response that arrives after a newer one', async () => {
    const resolvers: Array<(v: string) => void> = [];
    const fetcher = vi.fn<() => Promise<string>>(() => new Promise((resolve) => resolvers.push(resolve)));
    const bumpRef = { current: () => undefined as void };
    const Wrapper = ({ children }: { children: ReactNode }) => {
      const [v, setV] = useState(0);
      const bumpDataVersion = useCallback(() => setV((n) => n + 1), []);
      useEffect(() => {
        bumpRef.current = bumpDataVersion;
      }, [bumpDataVersion]);
      return <DataVersionContext.Provider value={{ dataVersion: v, bumpDataVersion }}>{children}</DataVersionContext.Provider>;
    };
    const { result } = renderHook(() => useAsync(fetcher, ['stale']), { wrapper: Wrapper });
    await waitFor(() => expect(resolvers).toHaveLength(1));
    act(() => bumpRef.current());
    await waitFor(() => expect(resolvers).toHaveLength(2));
    await act(async () => resolvers[1]('newer'));
    await waitFor(() => expect(result.current.data).toBe('newer'));
    await act(async () => resolvers[0]('older'));
    expect(result.current.data).toBe('newer');
  });

  it('does not update after unmount', async () => {
    let resolve: (v: string) => void = () => undefined;
    const fetcher = vi.fn<() => Promise<string>>(() => new Promise((r) => (resolve = r)));
    const { result, unmount } = renderHook(() => useAsync(fetcher, ['unmount']));
    await waitFor(() => expect(fetcher).toHaveBeenCalled());
    unmount();
    await act(async () => resolve('late'));
    expect(result.current.data).toBeNull();
  });
});
