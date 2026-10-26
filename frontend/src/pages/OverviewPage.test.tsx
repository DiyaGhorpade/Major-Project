import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { OverviewPage } from './OverviewPage';
import { mockFetch } from '../test/helpers';
import {
  errorEnvelope,
  monitoringFixture,
  overviewFixture,
  plantsFixture,
  preRunOverview,
  readingsFixture,
} from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

const routes = (overview: unknown = overviewFixture()) => ({
  'GET /api/monitoring/overview': { body: overview },
  'GET /api/plants': { body: plantsFixture(), headers: { 'X-Total-Count': '4' } },
  'GET /api/sensors/readings': { body: readingsFixture().slice(0, 4), headers: { 'X-Total-Count': '5376' } },
  'GET /api/monitoring/plants/5': { body: monitoringFixture() },
});

describe('OverviewPage', () => {
  it('renders real per-group, session and integrity data with no hard-coded numbers', async () => {
    mockFetch(routes());
    render(<OverviewPage onNavigate={vi.fn()} />);

    const c1 = await screen.findByTestId('cluster-card-Cluster 1');
    expect(within(c1).getByText('Cluster 1: Healthy Control')).toBeInTheDocument();
    expect(within(c1).getByText('Pots 1–4')).toBeInTheDocument();
    expect(within(c1).getByText('4/4 rated healthy')).toBeInTheDocument();
    const c2 = screen.getByTestId('cluster-card-Cluster 2');
    expect(within(c2).getByText('Cluster 2: Water Stress')).toBeInTheDocument();
    expect(within(c2).getByText('0/4 rated healthy')).toBeInTheDocument();
    expect(within(c2).getByText(/Water Stress/, { selector: 'strong' })).toBeInTheDocument();

    expect(screen.getByText('Session #4')).toBeInTheDocument();
    expect(screen.getByText('1,344 of 1,344 readings received')).toBeInTheDocument();
    expect(screen.getByText('95.7% Accepted')).toBeInTheDocument();
    expect(screen.getByText('5,376 records logged (233 flagged)')).toBeInTheDocument();
    expect(screen.getByText('3 devices')).toBeInTheDocument();
    expect(screen.getByText(/^Last data: .*Oct.* IST$/)).toBeInTheDocument();
    expect(screen.queryByText(/heartbeat/i)).not.toBeInTheDocument();

    expect(screen.queryByText(/17%/)).not.toBeInTheDocument();
    expect(screen.queryByText(/16\/16/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Session #8/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Jetson/)).not.toBeInTheDocument();

    // Prediction-dependent widgets use the API lists.
    expect(screen.getByTestId('model-snapshot')).toHaveTextContent('102');
    expect(within(screen.getByTestId('recent-alerts')).getAllByText('Pot 16: Krishna Tulsi (Climate Trial)').length).toBeGreaterThan(0);
    expect(within(screen.getByTestId('top-recommendations')).getAllByText('Monitor and inspect').length).toBeGreaterThan(0);
    expect(screen.queryByTestId('run-empty-state')).not.toBeInTheDocument();
  });

  it('makes the time-range buttons selectable (relative to the dataset)', async () => {
    mockFetch(routes());
    render(<OverviewPage onNavigate={vi.fn()} />);
    const full = await screen.findByRole('button', { name: 'Full dataset' });
    expect(screen.getByRole('button', { name: 'Last 7 days' })).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(full);
    expect(full).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', { name: 'Last 7 days' })).toHaveAttribute('aria-pressed', 'false');
  });

  it('before a Run: real reading counts plus a Run prompt where predictions are needed', async () => {
    mockFetch(routes(preRunOverview()));
    render(<OverviewPage onNavigate={vi.fn()} />);

    expect(await screen.findByText('5,376 records logged (233 flagged)')).toBeInTheDocument();
    expect(screen.getAllByTestId('run-empty-state').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Not analyzed yet').length).toBeGreaterThan(0);
    expect(screen.queryByTestId('model-snapshot')).not.toBeInTheDocument();
    expect(within(screen.getByTestId('cluster-card-Cluster 2')).getByText('Not analyzed yet')).toBeInTheDocument();
  });

  it('shows the error message when the API fails', async () => {
    mockFetch({
      ...routes(),
      'GET /api/monitoring/overview': { status: 500, body: errorEnvelope('internal_error', 'Overview exploded') },
    });
    render(<OverviewPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('Overview exploded')).toBeInTheDocument();
  });
});
