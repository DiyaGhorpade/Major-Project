import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export const IST_TIME_ZONE = 'Asia/Kolkata';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDateTime(isoString?: string | null): string {
  if (!isoString) return "n/a";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
      timeZone: IST_TIME_ZONE,
    });
  } catch {
    return isoString;
  }
}

export function formatRelativeTime(isoString?: string | null): string {
  if (!isoString) return "Never";
  try {
    const d = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now.getTime() - d.getTime()) / 1000);

    if (diffSec < 5) return "Just now";
    if (diffSec < 60) return `${diffSec}s ago`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHours = Math.floor(diffMin / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
  } catch {
    return "Unknown";
  }
}

export function formatPercent(value: number | null | undefined, decimals: number = 1): string {
  return value === null || value === undefined ? 'n/a' : `${value.toFixed(decimals)}%`;
}

/** A 0-1 fraction as a percentage, e.g. 0.912 -> "91.2%". Null, undefined and non-finite values give "n/a". */
export function formatFraction(value: number | null | undefined, decimals: number = 1): string {
  return value === null || value === undefined || !Number.isFinite(value) ? 'n/a' : formatPercent(value * 100, decimals);
}

export function formatTemp(value: number | null | undefined, decimals: number = 1): string {
  return value === null || value === undefined ? 'n/a' : `${value.toFixed(decimals)}°C`;
}

export function formatLux(value: number | null | undefined): string {
  return value === null || value === undefined ? 'n/a' : `${Math.round(value).toLocaleString()} lx`;
}

export function formatCount(value: number): string {
  return value.toLocaleString('en-US');
}

/** Current wall-clock time in IST, "HH:MM". */
export function formatIstClock(date: Date = new Date()): string {
  return date.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: IST_TIME_ZONE });
}

/** Render a nullable value, or "n/a" when the backend has nothing (dataset rows carry no hardware metrics). */
export function orNA(value: string | number | null | undefined, suffix: string = ''): string {
  return value === null || value === undefined || value === '' ? 'n/a' : `${value}${suffix}`;
}
