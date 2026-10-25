import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
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

  it('shows an IST clock instead of a hard-coded time', () => {
    renderTopBar({});
    expect(screen.getByText(/^IST \d{2}:\d{2}$/)).toBeInTheDocument();
    expect(screen.queryByText(/UTC 12:00/)).not.toBeInTheDocument();
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
