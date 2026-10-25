import { apiClient, type ReadingFilter } from '../api/client';
import type { SensorRecord } from '../types';
import { useAsync, type AsyncState } from './useApi';

export interface ReadingsState extends AsyncState<SensorRecord[]> {
  total: number;
}

/** Sensor readings (newest first by default) with the total from X-Total-Count. */
export function useReadings(filter?: ReadingFilter): ReadingsState {
  const state = useAsync(() => apiClient.getRecordsPage(filter), ['readings', filter]);
  const page = state.data;
  return { ...state, data: page ? page.items : null, total: page ? page.total : 0 };
}

/** Same filters plus `q` search for the Data Logs table. Export via `apiClient.dataLogsCsvUrl(filter)`. */
export function useDataLogs(filter?: ReadingFilter): ReadingsState {
  const state = useAsync(() => apiClient.getDataLogsPage(filter), ['dataLogs', filter]);
  const page = state.data;
  return { ...state, data: page ? page.items : null, total: page ? page.total : 0 };
}
