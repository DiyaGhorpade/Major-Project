import { useCallback, useState } from 'react';
import { apiClient, toApiError, type ApiError } from '../api/client';
import type { SystemSettings } from '../types';
import { useAsync, type AsyncState } from './useApi';

export interface SettingsState extends AsyncState<SystemSettings> {
  save: (patch: Partial<SystemSettings>) => Promise<SystemSettings | null>;
  saving: boolean;
  saveError: ApiError | null;
}

export function useSettings(): SettingsState {
  const state = useAsync(() => apiClient.getSettings(), ['settings']);
  const { refetch } = state;
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<ApiError | null>(null);

  const save = useCallback(
    async (patch: Partial<SystemSettings>) => {
      setSaving(true);
      setSaveError(null);
      try {
        const next = await apiClient.updateSettings(patch);
        await refetch();
        return next;
      } catch (err) {
        setSaveError(toApiError(err));
        return null;
      } finally {
        setSaving(false);
      }
    },
    [refetch],
  );

  return { ...state, save, saving, saveError };
}
