import { createContext, useContext } from 'react';

/** Counter bumped whenever backend data may have changed (run progress, completion, manual refresh). */
export interface DataVersionValue {
  dataVersion: number;
  bumpDataVersion: () => void;
}

export const DataVersionContext = createContext<DataVersionValue>({
  dataVersion: 0,
  bumpDataVersion: () => undefined,
});

export function useDataVersion(): DataVersionValue {
  return useContext(DataVersionContext);
}
