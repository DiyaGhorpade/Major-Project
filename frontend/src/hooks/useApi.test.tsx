import { describe, expect, it, vi } from 'vitest';
import { act, renderHook, waitFor } from '@testing-library/react';
import { type ReactNode, useState } from 'react';
import { useAsync } from './useApi';
import { DataVersionContext } from './dataVersion';
import { ApiError } from '../api/client';

describe('useAsync', () => {
  it('loads, keeps stale data while refetching on a data-version bump, and reports errors', async () => {
    const version = 0;
    const fetcher = vi
      .fn<() => Promise<string>>()
      .mockResolvedValueOnce('first')
      .mockImplementationOnce(() => new Promise((resolve) => setTimeout(() => resolve('second'), 20)))
      .mockRejectedValueOnce(new ApiError(503, 'service_unavailable', 'down'));

    let bump: () => void = () => undefined;
    const Wrapper = ({ children }: { children: ReactNode }) => {
      const [v, setV] = useState(version);
      bump = () => setV((n) => n + 1);
      return <DataVersionContext.Provider value={{ dataVersion: v, bumpDataVersion: bump }}>{children}</DataVersionContext.Provider>;
    };

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
});
