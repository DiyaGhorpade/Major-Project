import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { apiClient, toApiError, ApiError } from '../api/client';
import type { PipelineState, PipelineStatus } from '../types';

export const ACTIVE_POLL_MS = 700;
export const IDLE_POLL_MS = 5000;
export const BUMP_THROTTLE_MS = 2000;

export interface PipelineValue {
  /** Latest status from the backend (null until the first response). */
  status: PipelineStatus | null;
  state: PipelineState;
  percent: number;
  processed: number;
  total: number;
  message: string | null;
  /** True while queued or running (including the moment between click and first response). */
  isActive: boolean;
  /** True once any run has completed (drives the "Re-run" label). */
  hasCompletedRun: boolean;
  /** Set when starting a run or polling the status failed (e.g. backend unreachable). */
  error: ApiError | null;
  run: (reset?: boolean) => Promise<void>;
}

export const idlePipeline: PipelineValue = {
  status: null,
  state: 'idle',
  percent: 0,
  processed: 0,
  total: 0,
  message: null,
  isActive: false,
  hasCompletedRun: false,
  error: null,
  run: async () => undefined,
};

export const PipelineContext = createContext<PipelineValue>(idlePipeline);

/** Pipeline state shared by the whole app (provided by AppShell). */
export function usePipeline(): PipelineValue {
  return useContext(PipelineContext);
}

const isActiveState = (s: PipelineState) => s === 'queued' || s === 'running';

/**
 * The pipeline state machine: idle -> (run) -> queued/running -> completed | failed.
 * Polls /api/pipeline/status every ~700 ms while active (every few seconds otherwise) and calls
 * `onDataChanged` when progress advances (throttled to ~2 s) and on completion/failure.
 */
export function usePipelineMachine(onDataChanged: () => void): PipelineValue {
  const [status, setStatus] = useState<PipelineStatus | null>(null);
  const [starting, setStarting] = useState(false);
  const [everCompleted, setEverCompleted] = useState(false);
  // Failing to read the status and failing to start a run are tracked apart so a good status poll cannot erase a run error.
  const [pollError, setPollError] = useState<ApiError | null>(null);
  const [runError, setRunError] = useState<ApiError | null>(null);

  const onChangedRef = useRef(onDataChanged);
  useEffect(() => {
    onChangedRef.current = onDataChanged;
  });

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const alive = useRef(true);
  const prev = useRef<{ state: PipelineState; processed: number; runId: number | null } | null>(null);
  const lastBump = useRef(0);
  const pendingBump = useRef(false);
  const activeHint = useRef(false);
  const startingRef = useRef(false);
  const pollRef = useRef<() => Promise<void>>(async () => undefined);

  const schedule = useCallback((ms: number) => {
    if (timer.current) clearTimeout(timer.current);
    if (!alive.current) return;
    timer.current = setTimeout(() => void pollRef.current(), ms);
  }, []);

  const bump = useCallback(() => {
    lastBump.current = Date.now();
    pendingBump.current = false;
    onChangedRef.current();
  }, []);

  const poll = useCallback(async () => {
    // Polling must always continue (a failed or timed-out status request just tries again), so reschedule in `finally`.
    try {
      let next: PipelineStatus | null = null;
      try {
        next = await apiClient.getPipelineStatus();
        if (!alive.current) return;
        setPollError(null);
      } catch (err) {
        if (!alive.current) return;
        setPollError(toApiError(err));
        setStarting(false); // never leave the Run button locked on "Starting" when the status cannot be read
      }
      if (next) {
        const before = prev.current;
        const finished = next.state === 'completed' || next.state === 'failed';
        const progressed = before !== null && (next.processed !== before.processed || next.runId !== before.runId);
        const justFinished =
          before !== null && finished && (isActiveState(before.state) || next.runId !== before.runId);
        if (next.state === 'completed') setEverCompleted(true);
        if (justFinished) {
          bump();
        } else if (progressed && isActiveState(next.state)) {
          if (Date.now() - lastBump.current >= BUMP_THROTTLE_MS) bump();
          else pendingBump.current = true;
        } else if (pendingBump.current && Date.now() - lastBump.current >= BUMP_THROTTLE_MS) {
          bump();
        }
        prev.current = { state: next.state, processed: next.processed, runId: next.runId };
        activeHint.current = isActiveState(next.state);
        setStatus(next);
        setStarting(false);
      }
    } finally {
      if (alive.current) schedule(activeHint.current ? ACTIVE_POLL_MS : IDLE_POLL_MS);
    }
  }, [bump, schedule]);
  useEffect(() => {
    pollRef.current = poll;
  }, [poll]);

  useEffect(() => {
    alive.current = true;
    void pollRef.current();
    return () => {
      alive.current = false;
      if (timer.current) clearTimeout(timer.current);
    };
  }, []);

  const run = useCallback(
    async (reset = false) => {
      if (startingRef.current) return; // a start request is already in flight
      startingRef.current = true;
      setStarting(true);
      setRunError(null);
      activeHint.current = true;
      try {
        await apiClient.runPipeline(reset);
      } catch (err) {
        startingRef.current = false;
        if (!alive.current) return;
        activeHint.current = false;
        setStarting(false);
        setRunError(toApiError(err));
        return;
      }
      startingRef.current = false;
      if (!alive.current) return;
      // Poll right away so the UI leaves "idle" without waiting for the next tick.
      await pollRef.current();
    },
    [],
  );

  return useMemo(() => {
    const backendState: PipelineState = status?.state ?? 'idle';
    const state: PipelineState = starting && !isActiveState(backendState) ? 'queued' : backendState;
    return {
      status,
      state,
      percent: starting && !isActiveState(backendState) ? 0 : (status?.percent ?? 0),
      processed: starting && !isActiveState(backendState) ? 0 : (status?.processed ?? 0),
      // While starting, the previous run's numbers are stale: report 0 so the UI shows "Starting..." rather than "0 / <old total>".
      total: starting && !isActiveState(backendState) ? 0 : (status?.total ?? 0),
      message: status?.message ?? null,
      isActive: isActiveState(state),
      hasCompletedRun: everCompleted || backendState === 'completed',
      error: runError ?? pollError,
      run,
    };
  }, [status, starting, everCompleted, runError, pollError, run]);
}
