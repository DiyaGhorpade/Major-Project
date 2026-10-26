/** Validation for the Unusual Readings filters; the ranges mirror what the backend accepts. */

export const PLANT_ID_MIN = 1;
export const PLANT_ID_MAX = 16;
/** Backend range of the `minScore` query parameter (anything outside it is a 422). */
export const MIN_SCORE_MIN = -10;
export const MIN_SCORE_MAX = 10;

/** A plant id from a URL hash or a prop: the integer 1-16 as a string, otherwise null (show all plants). */
export function parseAnomalyPlantId(raw: number | string | null | undefined): string | null {
  const text = raw === null || raw === undefined ? '' : String(raw).trim();
  if (!/^\d+$/.test(text)) return null;
  const id = Number(text);
  return id >= PLANT_ID_MIN && id <= PLANT_ID_MAX ? String(id) : null;
}

/** The typed minimum score as a number clamped to the backend range; undefined when empty or not a number. */
export function parseMinScore(text: string): number | undefined {
  if (text.trim() === '') return undefined;
  const value = Number(text);
  if (!Number.isFinite(value)) return undefined;
  return Math.min(MIN_SCORE_MAX, Math.max(MIN_SCORE_MIN, value));
}
