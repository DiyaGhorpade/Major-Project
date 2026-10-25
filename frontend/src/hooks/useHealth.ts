import { createContext, useContext, useEffect } from 'react';
import { useAsync } from './useApi';
import { apiClient, type ApiError } from '../api/client';
import type { HealthResponse } from '../types';

export const HEALTH_POLL_MS = 10000;

export interface HealthValue {
  health: HealthResponse | null;
  /** Set when the backend itself cannot be reached. */
  error: ApiError | null;
  /** True when the backend answered and reports the ML service as reachable with models loaded. */
  mlConnected: boolean;
  /** True when the backend answered and reports the ML service as not usable. */
  mlUnreachable: boolean;
}

export const HealthContext = createContext<HealthValue>({ health: null, error: null, mlConnected: false, mlUnreachable: false });

export const useHealth = (): HealthValue => useContext(HealthContext);

/** Polls /api/health (backend, DB and ML reachability, model version). Used once, by AppShell. */
export function useHealthPolling(): HealthValue {
  const { data, error, refetch } = useAsync(() => apiClient.getHealth(), ['health']);
  useEffect(() => {
    const id = setInterval(() => void refetch(), HEALTH_POLL_MS);
    return () => clearInterval(id);
  }, [refetch]);
  return {
    health: data,
    error,
    mlConnected: data?.mlService.ok === true,
    mlUnreachable: data !== null && !data.mlService.ok,
  };
}
