import { IST_TIME_ZONE } from '../../lib/utils';

/** Time ranges for trend charts. Ranges are relative to the dataset's latest timestamp (the data is historical). */
export type TimeRange = '24h' | '7d' | 'all';

export const TIME_RANGE_LABELS: Record<TimeRange, string> = {
  '24h': 'Last 24 h',
  '7d': 'Last 7 days',
  all: 'Full dataset',
};

const RANGE_MS: Record<Exclude<TimeRange, 'all'>, number> = {
  '24h': 24 * 3600 * 1000,
  '7d': 7 * 24 * 3600 * 1000,
};

export function toMillis(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const ms = new Date(iso).getTime();
  return Number.isNaN(ms) ? null : ms;
}

/** Keep points within the range ending at the latest point (not at "now"). Input order is preserved. */
export function filterByRange<T extends { timestamp: string }>(points: readonly T[], range: TimeRange): T[] {
  if (range === 'all' || points.length === 0) return [...points];
  const times = points.map((p) => toMillis(p.timestamp)).filter((t): t is number => t !== null);
  if (times.length === 0) return [...points];
  const cutoff = Math.max(...times) - RANGE_MS[range];
  return points.filter((p) => {
    const t = toMillis(p.timestamp);
    return t !== null && t >= cutoff;
  });
}

/** Axis tick in IST, e.g. "18 Oct 14:00" (date included) or "14:00". */
export function formatIstTick(ms: number, withDate = true): string {
  const d = new Date(ms);
  const time = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: IST_TIME_ZONE });
  if (!withDate) return time;
  const day = d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', timeZone: IST_TIME_ZONE });
  return `${day} ${time}`;
}

/** "6d 22h", "3h 5m", "42s". Null-safe. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) return 'n/a';
  const s = Math.max(0, Math.round(seconds));
  const days = Math.floor(s / 86400);
  const hours = Math.floor((s % 86400) / 3600);
  const minutes = Math.floor((s % 3600) / 60);
  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${s % 60}s`;
  return `${s}s`;
}

/** 0.62 -> "62%". */
export function formatConfidence(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : `${Math.round(value * 100)}%`;
}

/** "1, 2, 3, 4" -> "Pots 1-4" when contiguous; otherwise the plain list. */
export function formatPlantIds(ids: readonly number[]): string {
  if (ids.length === 0) return 'no pots';
  const sorted = [...ids].sort((a, b) => a - b);
  const contiguous = sorted.every((v, i) => i === 0 || v === sorted[i - 1] + 1);
  if (sorted.length > 2 && contiguous) return `Pots ${sorted[0]}–${sorted[sorted.length - 1]}`;
  return `${sorted.length === 1 ? 'Pot' : 'Pots'} ${sorted.join(', ')}`;
}

/**
 * Describe a daily collection schedule from "HH:MM" times, e.g. "every 2 h (12 per day)".
 * Returns null when there are no times; falls back to the plain list when the spacing is uneven.
 */
export function describeCadence(times: readonly string[] | null | undefined): string | null {
  if (!times || times.length === 0) return null;
  const minutes = times
    .map((t) => {
      const [h, m] = t.split(':').map(Number);
      return h * 60 + m;
    })
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);
  if (minutes.length === 0) return null;
  if (minutes.length === 1) return `once a day (${times[0]})`;
  const gaps = minutes.map((m, i) => (i === 0 ? m + 1440 - minutes[minutes.length - 1] : m - minutes[i - 1]));
  const uniform = gaps.every((g) => g === gaps[0]);
  if (!uniform) return `${minutes.length} per day (${[...times].sort().join(', ')})`;
  const gap = gaps[0];
  const spacing = gap % 60 === 0 ? `${gap / 60} h` : `${gap} min`;
  return `every ${spacing} (${minutes.length} per day)`;
}

/**
 * The cadence the data actually shows: a session's span divided by the readings each plant has in it,
 * e.g. "every 2 h (12 per day)". Null when the session is too small to tell.
 */
export function observedCadence(
  session: { received_records: number; duration_seconds: number },
  plantCount: number,
): string | null {
  if (plantCount <= 0) return null;
  const perPlant = session.received_records / plantCount;
  if (!Number.isFinite(perPlant) || perPlant < 2 || session.duration_seconds <= 0) return null;
  const intervalMin = Math.round(session.duration_seconds / (perPlant - 1) / 60 / 5) * 5; // nearest 5 min
  if (intervalMin <= 0) return null;
  const spacing = intervalMin % 60 === 0 ? `${intervalMin / 60} h` : `${intervalMin} min`;
  return `every ${spacing} (${Math.round(1440 / intervalMin)} per day)`;
}

/** Session filter handed from the Sessions page to the Data Logs page (the router only carries a route name). */
export const SESSION_FILTER_KEY = 'plant-app.dataLogs.sessionId';

export function stashSessionFilter(sessionId: number | null): void {
  try {
    if (sessionId === null) window.sessionStorage.removeItem(SESSION_FILTER_KEY);
    else window.sessionStorage.setItem(SESSION_FILTER_KEY, String(sessionId));
  } catch {
    // storage unavailable: the Data Logs page simply opens unfiltered
  }
}

/** Read the stashed session filter without consuming it (pure, so it is safe in a state initializer under StrictMode). */
export function peekSessionFilter(): number | null {
  try {
    const raw = window.sessionStorage.getItem(SESSION_FILTER_KEY);
    const n = raw === null ? NaN : Number(raw);
    return Number.isInteger(n) && n >= 1 ? n : null;
  } catch {
    return null;
  }
}

/** Consume the stashed filter (call from an effect once it has been applied). */
export function clearSessionFilter(): void {
  stashSessionFilter(null);
}
