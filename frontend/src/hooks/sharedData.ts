import { createContext } from 'react';
import type { Overview } from '../types';
import type { AsyncState } from './useApi';
import type { PlantsState } from './usePlants';

/**
 * Data the app shell already loads (overview for the bell and the Overview page, plants for the search box and the
 * plant pages). `useOverview()` / `usePlants()` read it from here when a provider exists, so each is fetched once;
 * without a provider (a page rendered on its own) they fetch for themselves.
 */
export interface SharedData {
  overview: AsyncState<Overview> | null;
  plants: PlantsState | null;
}

export const SharedDataContext = createContext<SharedData>({ overview: null, plants: null });
