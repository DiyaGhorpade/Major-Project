import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { AnomaliesPage } from './AnomaliesPage';
import { mockFetch, statusBody } from '../test/helpers';
import { anomaliesByScoreFixture, anomaliesFixture, anomalySummaryFixture, errorEnvelope, plantsFixture } from '../test/fixtureHelpers';

afterEach(() => vi.unstubAllGlobals());

const page = (items: unknown[], total = 102) => ({ body: items, headers: { 'X-Total-Count': String(total) } });

function backend(over: Record<string, unknown> = {}) {
  return mockFetch({
    'GET /api/ai/anomalies/summary': { body: anomalySummaryFixture() },
    'GET /api/ai/anomalies': page(anomaliesFixture()),
    'GET /api/plants': page(plantsFixture(), 16),
    'GET /api/pipeline/status': { body: statusBody() },
    ...over,
  });
}

const lastQuery = (calls: Array<{ key: string; url: string }>) => {
  const list = calls.filter((c) => c.key === 'GET /api/ai/anomalies');
  return new URL(list[list.length - 1].url, 'http://x').searchParams;
};

describe('AnomaliesPage', () => {
  it('shows the KPI row, affected plants and flagged readings from the real payloads', async () => {
    backend();
    render(<AnomaliesPage onNavigate={vi.fn()} />);

    expect(await screen.findByRole('heading', { level: 1, name: 'Unusual Readings' })).toBeInTheDocument();
    const kpi = (title: string) => screen.getByText(title).closest('[class*="shadow-card"]') as HTMLElement;
    expect(await screen.findByText(/1\.9% of 5,376 analyzed readings/)).toBeInTheDocument();
    expect(within(kpi('Flagged Readings')).getByText('102')).toBeInTheDocument();
    expect(within(kpi('Plants Affected')).getByText('14')).toBeInTheDocument();
    expect(within(kpi('Threshold')).getByText('0.000')).toBeInTheDocument();

    const plants = within(screen.getByTestId('affected-plants'));
    const rows = plants.getAllByRole('row');
    expect(rows).toHaveLength(15);
    expect(within(rows[1]).getByText('Pot 14: Krishna Tulsi (Climate Trial)')).toBeInTheDocument();
    expect(within(rows[1]).getByText('42')).toBeInTheDocument();
    expect(within(rows[1]).getByText('Cluster 4: Temperature/Humidity Stress')).toBeInTheDocument();
    expect(within(rows[1]).getByText('0.085')).toBeInTheDocument();

    const readings = within(screen.getByTestId('anomaly-readings'));
    expect(readings.getAllByRole('row')).toHaveLength(9);
    expect(readings.getByText('Showing', { exact: false })).toHaveTextContent('Showing 1 to 15 of 102 readings');
    expect(readings.getAllByText('0.040').length).toBeGreaterThan(0);
    // Neutral copy: no verdicts and no caveats.
    expect(document.body.textContent).not.toMatch(/sensor fault|synthetic|experimental|not validated/i);
  });

  it('links plants to their detail page', async () => {
    backend();
    const onNavigate = vi.fn();
    render(<AnomaliesPage onNavigate={onNavigate} />);
    const plants = within(await screen.findByTestId('affected-plants'));
    fireEvent.click(plants.getByRole('button', { name: 'Pot 14: Krishna Tulsi (Climate Trial)' }));
    expect(onNavigate).toHaveBeenCalledWith('plant-detail', 14);
  });

  it('sends plant, sort, min score and paging to the server and resets to page 1', async () => {
    const { calls } = backend();
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    await screen.findByTestId('anomaly-readings');

    fireEvent.click(screen.getByRole('button', { name: 'Next page' }));
    await waitFor(() => expect(lastQuery(calls).get('offset')).toBe('15'));

    fireEvent.change(screen.getByLabelText('Sort'), { target: { value: 'score' } });
    await waitFor(() => expect(lastQuery(calls).get('sort')).toBe('score'));
    expect(lastQuery(calls).get('offset')).toBe('0');

    fireEvent.change(screen.getByLabelText('Minimum score'), { target: { value: '0.05' } });
    await waitFor(() => expect(lastQuery(calls).get('minScore')).toBe('0.05'));

    fireEvent.change(screen.getByLabelText('Plant filter'), { target: { value: '10' } });
    await waitFor(() => expect(lastQuery(calls).get('plantId')).toBe('10'));
    expect(lastQuery(calls).get('limit')).toBe('15');

    fireEvent.click(screen.getByRole('button', { name: 'Clear filters' }));
    await waitFor(() => expect(lastQuery(calls).has('plantId')).toBe(false));
    expect(lastQuery(calls).has('minScore')).toBe(false);
  });

  it('opens filtered to the plant it was given', async () => {
    const { calls } = backend({ 'GET /api/ai/anomalies': page(anomaliesByScoreFixture().slice(0, 2), 2) });
    render(<AnomaliesPage onNavigate={vi.fn()} initialPlantId="10" />);
    await screen.findByTestId('anomaly-readings');
    expect(lastQuery(calls).get('plantId')).toBe('10');
    expect(screen.getByLabelText('Plant filter')).toHaveValue('10');
  });

  it('ignores a plant id that is not 1-16', async () => {
    const { calls } = backend();
    render(<AnomaliesPage onNavigate={vi.fn()} initialPlantId="999" />);
    await screen.findByTestId('anomaly-readings');
    expect(lastQuery(calls).has('plantId')).toBe(false);
    expect(screen.getByLabelText('Plant filter')).toHaveValue('all');
  });

  it('clamps the minimum score to the range the server accepts', async () => {
    const { calls } = backend();
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    await screen.findByTestId('anomaly-readings');
    fireEvent.change(screen.getByLabelText('Minimum score'), { target: { value: '500' } });
    await waitFor(() => expect(lastQuery(calls).get('minScore')).toBe('10'));
    fireEvent.change(screen.getByLabelText('Minimum score'), { target: { value: '-99' } });
    await waitFor(() => expect(lastQuery(calls).get('minScore')).toBe('-10'));
    expect(screen.getByLabelText('Minimum score')).toHaveAttribute('min', '-10');
    expect(screen.getByLabelText('Minimum score')).toHaveAttribute('max', '10');
  });

  it('steps back to the last page when the total shrinks below the current page', async () => {
    let shrunk = false;
    const ref: { calls: Array<{ key: string; url: string }> } = { calls: [] };
    const { calls } = backend({
      'GET /api/ai/anomalies': () => {
        const offset = Number(new URL(ref.calls[ref.calls.length - 1].url, 'http://x').searchParams.get('offset'));
        if (offset >= 30) shrunk = true; // the data changed under the user: only 20 readings are left
        return shrunk ? page(offset >= 20 ? [] : anomaliesFixture().slice(0, 3), 20) : page(anomaliesFixture(), 102);
      },
    });
    ref.calls = calls;
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    await screen.findByTestId('anomaly-readings');
    fireEvent.click(screen.getByRole('button', { name: 'Next page' }));
    await waitFor(() => expect(lastQuery(calls).get('offset')).toBe('15'));
    fireEvent.click(await screen.findByRole('button', { name: 'Next page' }));
    expect(await screen.findByText('Page 2 of 2')).toBeInTheDocument();
    expect(lastQuery(calls).get('offset')).toBe('15');
    expect(screen.getByTestId('anomaly-readings')).toHaveTextContent('Showing 16 to 20 of 20 readings');
  });

  it('shows an empty row when the filters match nothing', async () => {
    backend({ 'GET /api/ai/anomalies': page([], 0) });
    render(<AnomaliesPage onNavigate={vi.fn()} initialPlantId={3} />);
    expect(await screen.findByText('No unusual readings match these filters.')).toBeInTheDocument();
  });

  it('shows the Run prompt before any analysis', async () => {
    const empty = { ...anomalySummaryFixture(), flagged: 0, totalScored: 0, rate: null, plantsAffected: 0, latestFlaggedAt: null, plants: [] };
    backend({ 'GET /api/ai/anomalies/summary': { body: empty }, 'GET /api/ai/anomalies': page([], 0) });
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    expect(await screen.findByTestId('run-empty-state')).toHaveTextContent('Press Run to analyze the plant data.');
  });

  it('says so when an analysis found nothing unusual', async () => {
    const none = { ...anomalySummaryFixture(), flagged: 0, rate: 0, plantsAffected: 0, latestFlaggedAt: null, plants: [] };
    backend({ 'GET /api/ai/anomalies/summary': { body: none }, 'GET /api/ai/anomalies': page([], 0) });
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('No unusual readings')).toBeInTheDocument();
  });

  it('shows an error state with retry when the summary fails', async () => {
    backend({ 'GET /api/ai/anomalies/summary': { status: 500, body: errorEnvelope('internal_error', 'boom') } });
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('boom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
  });

  it('shows an error state when only the readings list fails', async () => {
    backend({ 'GET /api/ai/anomalies': { status: 500, body: errorEnvelope('internal_error', 'list broke') } });
    render(<AnomaliesPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('list broke')).toBeInTheDocument();
    expect(screen.getByTestId('affected-plants')).toBeInTheDocument();
  });
});
