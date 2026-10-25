import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { LiveMonitoringPage } from './LiveMonitoringPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, flaggedFixture, monitoringFixture, plantsFixture, preRunMonitoring, readingsFixture, settingsFixture } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

function routes(over: Record<string, unknown> = {}) {
  const feed = [...flaggedFixture(), ...readingsFixture().slice(0, 2)];
  return {
    'GET /api/plants': { body: plantsFixture(), headers: { 'X-Total-Count': '4' } },
    'GET /api/settings': { body: { ...settingsFixture(), ...(over.settings as object) } },
    'GET /api/sensors/readings': { body: feed, headers: { 'X-Total-Count': '5376' } },
    'GET /api/monitoring/plants/5': { body: over.monitoring ?? monitoringFixture() },
    'GET /api/monitoring/plants/1': { body: over.monitoring ?? monitoringFixture() },
  };
}

describe('LiveMonitoringPage', () => {
  it('is an honest latest-readings view of the dataset, defaulting to the first pot', async () => {
    const { calls } = mockFetch(routes());
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);

    const focus = await screen.findByTestId('focus-plant');
    expect(within(focus).getByText('Pot 5: Rama Tulsi (Watering Trial)')).toBeInTheDocument();
    expect(within(focus).getByText(/Node 1/)).toBeInTheDocument();
    expect(screen.getByText("Showing the bundled dataset's most recent readings")).toBeInTheDocument();
    expect(await screen.findByText(/Auto-refresh every 5 s/)).toBeInTheDocument();
    expect(screen.queryByText(/Jetson/)).not.toBeInTheDocument();
    expect(screen.queryByText(/5000/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Port/)).not.toBeInTheDocument();

    // Model output badges with the disclaimer tips.
    expect(within(focus).getByText('Water Stress')).toBeInTheDocument();
    expect(within(focus).getAllByRole('note').length).toBeGreaterThanOrEqual(3);
    expect(within(focus).getByText('Environmental state group 1')).toBeInTheDocument();

    // Feed: a flagged reading with a missing soil value renders n/a, not 0.
    const flaggedRow = (await screen.findAllByText('Pot #16'))[0].closest('tr')!;
    expect(within(flaggedRow).getAllByText('n/a').length).toBeGreaterThan(0);
    expect(calls.some((c) => c.key === 'GET /api/sensors/readings')).toBe(true);
  });

  it('shows reading times and the refresh time in IST', async () => {
    mockFetch(routes());
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);
    const focus = await screen.findByTestId('focus-plant');
    // 2026-10-18T22:01:30+05:30 is shown as IST wall-clock time, whatever the machine zone is.
    expect(within(focus).getByText(/Reading time \(IST\): Oct 18, 22:01:30/)).toBeInTheDocument();
    expect(await screen.findByText(/last refreshed .* IST/)).toBeInTheDocument();
    const row = (await screen.findAllByText('Pot #16'))[0].closest('tr')!;
    expect(within(row).getByText('Oct 18, 18:05:15')).toBeInTheDocument();
  });

  it('polls the readings API at the interval from Settings', async () => {
    const { calls } = mockFetch(routes({ settings: { polling_interval_seconds: 1 } }));
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);
    await screen.findByTestId('focus-plant');
    await screen.findByText(/Auto-refresh every 1 s/);
    const count = () => calls.filter((c) => c.key === 'GET /api/sensors/readings').length;
    const first = count();
    await waitFor(() => expect(count()).toBeGreaterThan(first), { timeout: 4000 });
  });

  it('passes the group filter to the API', async () => {
    const { calls } = mockFetch(routes());
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);
    await screen.findByTestId('focus-plant');
    fireEvent.change(screen.getByLabelText('Group filter'), { target: { value: 'Cluster 3' } });
    await waitFor(() => expect(calls.some((c) => c.url.includes('cluster=Cluster+3'))).toBe(true));
  });

  it('before a Run the model block says not analyzed yet', async () => {
    mockFetch(routes({ monitoring: preRunMonitoring() }));
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);
    expect(await screen.findByTestId('prediction-missing')).toHaveTextContent('Not analyzed yet');
  });

  it('shows an error when the API fails', async () => {
    mockFetch({ ...routes(), 'GET /api/sensors/readings': { status: 503, body: errorEnvelope('unavailable', 'Backend busy') } });
    render(<LiveMonitoringPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('Backend busy')).toBeInTheDocument();
  });
});
