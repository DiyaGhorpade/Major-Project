import { afterEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { TopBar } from './TopBar';
import { PipelineContext, idlePipeline, type PipelineValue } from '../../hooks/usePipeline';
import type { Alert } from '../../types';

function renderTopBar(value: Partial<PipelineValue>, alerts: Alert[] = []) {
  const pipeline: PipelineValue = { ...idlePipeline, ...value };
  return render(
    <PipelineContext.Provider value={pipeline}>
      <TopBar currentRoute="overview" onOpenMobile={() => undefined} onRefresh={() => undefined} onNavigate={() => undefined} alerts={alerts} />
    </PipelineContext.Provider>,
  );
}

describe('TopBar Run button', () => {
  it('shows an enabled "Run" button before any run and starts the pipeline on click', () => {
    const run = vi.fn().mockResolvedValue(undefined);
    renderTopBar({ run });
    const button = screen.getByRole('button', { name: 'Run' });
    expect(button).toBeEnabled();
    fireEvent.click(button);
    expect(run).toHaveBeenCalledTimes(1);
    expect(screen.queryByTestId('run-progress')).not.toBeInTheDocument();
  });

  it('disables the button and shows progress text while running', () => {
    renderTopBar({ state: 'running', isActive: true, processed: 3120, total: 5376, percent: 58 });
    expect(screen.getByRole('button', { name: 'Run' })).toBeDisabled();
    expect(screen.getByText('Analyzing 3,120 / 5,376 readings...')).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '58');
  });

  it('shows "Re-run" once a run has completed', () => {
    renderTopBar({ state: 'completed', hasCompletedRun: true, processed: 5376, total: 5376, percent: 100 });
    expect(screen.getByRole('button', { name: 'Re-run' })).toBeEnabled();
    expect(screen.queryByTestId('run-progress')).not.toBeInTheDocument();
  });

  afterEach(() => vi.useRealTimers());

  it('shows the current time in IST (not the machine zone) and keeps it ticking', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-10-18T10:00:00Z')); // 15:30 in IST (UTC+5:30)
    renderTopBar({});
    expect(screen.getByText('IST 15:30')).toBeInTheDocument();
    act(() => {
      vi.advanceTimersByTime(60000);
    });
    expect(screen.getByText('IST 15:31')).toBeInTheDocument();
  });

  it('shows "Starting..." instead of the previous run total right after clicking Run', () => {
    renderTopBar({ state: 'queued', isActive: true, processed: 0, total: 0 });
    expect(screen.getAllByText(/Starting/).length).toBeGreaterThan(0);
    expect(screen.queryByText(/Analyzing 0 \/ 5,376/)).not.toBeInTheDocument();
  });

  it('jumps to a plant by number or name from the search box', () => {
    const onNavigate = vi.fn();
    const plants = [
      { id: 5, name: 'Pot 5: Rama Tulsi (Watering Trial)' },
      { id: 12, name: 'Pot 12: Krishna Tulsi (Light Trial)' },
    ];
    render(
      <PipelineContext.Provider value={idlePipeline}>
        <TopBar currentRoute="overview" onOpenMobile={() => undefined} onRefresh={() => undefined} onNavigate={onNavigate} plants={plants} />
      </PipelineContext.Provider>,
    );
    const input = screen.getByLabelText('Jump to a plant by number or name');
    fireEvent.change(input, { target: { value: '#12' } });
    fireEvent.submit(input.closest('form')!);
    expect(onNavigate).toHaveBeenLastCalledWith('plant-detail', 12);

    fireEvent.change(input, { target: { value: 'watering' } });
    fireEvent.submit(input.closest('form')!);
    expect(onNavigate).toHaveBeenLastCalledWith('plant-detail', 5);

    fireEvent.change(input, { target: { value: 'nothing here' } });
    fireEvent.submit(input.closest('form')!);
    expect(screen.getByRole('alert')).toHaveTextContent('No matching plant.');
    expect(onNavigate).toHaveBeenCalledTimes(2);
  });

  it('lists real alerts in the bell and hides the badge when there are none', () => {
    const alert: Alert = {
      kind: 'recommendation',
      plantId: 5,
      plantName: 'Pot 5: Rama Tulsi',
      timestamp: '2026-10-18T22:01:30+05:30',
      message: 'Check watering',
      severity: 'high',
      recordId: 'R-5365',
      recommendationId: 'rec-1',
    };
    const { unmount } = renderTopBar({}, [alert]);
    fireEvent.click(screen.getByTitle('Recent alerts'));
    expect(screen.getByText('Pot 5: Rama Tulsi')).toBeInTheDocument();
    expect(screen.getByText('Check watering')).toBeInTheDocument();
    unmount();

    renderTopBar({}, []);
    fireEvent.click(screen.getByTitle('Recent alerts'));
    expect(screen.getByText(/No alerts/)).toBeInTheDocument();
  });
});
