import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError, toApiError } from '../api/client';
import { useDataVersion } from './dataVersion';

export interface AsyncState<T> {
  data: T | null;
  /** True only while there is no data yet (first load, or after the key changed). */
  loading: boolean;
  /** True while a refetch is in flight and older (stale) data is still shown. */
  refreshing: boolean;
  error: ApiError | null;
  refetch: () => Promise<void>;
}

interface Options {
  /** Skip fetching entirely (e.g. no plant selected). */
  enabled?: boolean;
}

/**
 * Generic data hook. Fetches on mount, whenever `key` changes, and whenever the global data version
 * is bumped (pipeline progress/completion, manual refresh). Stale data stays visible while refetching.
 */
export function useAsync<T>(fetcher: () => Promise<T>, key: ReadonlyArray<unknown> = [], options: Options = {}): AsyncState<T> {
  const enabled = options.enabled ?? true;
  const { dataVersion } = useDataVersion();
  const keyText = JSON.stringify(key);

  const [state, setState] = useState<{ keyText: string; data: T | null; error: ApiError | null; settledKey: string | null }>({
    keyText,
    data: null,
    error: null,
    settledKey: null,
  });
  const [inFlight, setInFlight] = useState(0);

  const fetcherRef = useRef(fetcher);
  useEffect(() => {
    fetcherRef.current = fetcher;
  });
  const requestId = useRef(0);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async () => {
    const id = ++requestId.current;
    // Start the in-flight count on a microtask so it is never a synchronous setState inside the fetching effect.
    await Promise.resolve();
    if (!mounted.current) return;
    setInFlight((n) => n + 1);
    try {
      const data = await fetcherRef.current();
      if (mounted.current && id === requestId.current) {
        setState({ keyText, data, error: null, settledKey: keyText });
      }
    } catch (err) {
      if (mounted.current && id === requestId.current) {
        setState((prev) => ({
          keyText,
          data: prev.keyText === keyText ? prev.data : null,
          error: toApiError(err),
          settledKey: keyText,
        }));
      }
    } finally {
      if (mounted.current) setInFlight((n) => n - 1);
    }
  }, [keyText]);

  useEffect(() => {
    if (!enabled) return;
    void run();
  }, [run, enabled, dataVersion]);

  const sameKey = state.keyText === keyText;
  const data = sameKey ? state.data : null;
  const error = sameKey ? state.error : null;
  const settled = sameKey && state.settledKey === keyText;

  return {
    data,
    loading: enabled && data === null && !settled,
    refreshing: inFlight > 0 && data !== null,
    error,
    refetch: run,
  };
}
