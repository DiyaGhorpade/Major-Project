import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { PlantDetailPage } from './PlantDetailPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, monitoringFixture, preRunMonitoring } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

const monitoringRoute = (body: unknown) => ({ 'GET /api/monitoring/plants/5': { body } });

describe('PlantDetailPage', () => {
  it('renders ML output, history, anomaly/env panels and keeps dataset labels separate', async () => {
    mockFetch(monitoringRoute(monitoringFixture()));
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);

    expect(await screen.findByRole('heading', { level: 1, name: 'Pot 5: Rama Tulsi (Watering Trial)' })).toBeInTheDocument();
    const ml = screen.getByTestId('ml-panel');
    expect(within(ml).getAllByText('High stress').length).toBeGreaterThan(0);
    expect(within(ml).getAllByText('62%').length).toBeGreaterThan(0);
    expect(within(ml).getByText(/Stress type guess \(heuristic\)/)).toBeInTheDocument();
    expect(within(ml).getByText('Water Stress')).toBeInTheDocument();
    expect(within(screen.getByTestId('anomaly-panel')).getByText('Normal reading')).toBeInTheDocument();
    expect(within(screen.getByTestId('anomaly-panel')).getAllByRole('note').length).toBeGreaterThan(0);
    expect(within(screen.getByTestId('envstate-panel')).getByText('Environmental state group 1')).toBeInTheDocument();

    // Prediction history table: newest first, capped to 12 rows until expanded.
    const history = screen.getByTestId('prediction-history');
    expect(within(history).getAllByRole('row').length).toBe(13);
    fireEvent.click(within(history).getByRole('button', { name: /Show all 30 predictions/ }));
    expect(within(history).getAllByRole('row').length).toBe(31);

    // Dataset labels are a separate, explicitly labelled block (not inside the ML panel).
    const labels = screen.getByTestId('dataset-labels');
    expect(labels).toHaveTextContent('Dataset labels (synthetic ground truth, not model output)');
    expect(labels).toHaveTextContent('high');
    expect(labels).toHaveTextContent('water');
    expect(ml).not.toContainElement(labels);

    expect(screen.getByText('Check watering')).toBeInTheDocument();
    expect(screen.getByText(/Advisory heuristic\./, { selector: 'p' })).toBeInTheDocument();
  });

  it('shows a neutral Night state, not "Shaded / Dim", for a night-time light reading', async () => {
    mockFetch(monitoringRoute(monitoringFixture())); // latest reading is 22:01 IST; daytime hours come from the API
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    await screen.findByRole('heading', { level: 1, name: 'Pot 5: Rama Tulsi (Watering Trial)' });
    expect(screen.getByText('Night')).toBeInTheDocument();
    expect(screen.queryByText('Shaded / Dim')).not.toBeInTheDocument();
  });

  it('explains a 409 conflict when reopening a recommendation', async () => {
    const m = monitoringFixture();
    const rec = m.recommendations[0];
    const { calls } = mockFetch({
      ...monitoringRoute(m),
      [`PATCH /api/recommendations/${rec.id}`]: {
        status: 409,
        body: errorEnvelope('conflict', 'An active recommendation already exists'),
      },
    });
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    await screen.findByText(rec.title);

    fireEvent.change(screen.getByDisplayValue('New'), { target: { value: 'in_progress' } });
    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent('another active recommendation for the same plant and category already exists');
    expect(calls.some((c) => c.key.startsWith('PATCH '))).toBe(true);
    fireEvent.click(within(alert).getByRole('button', { name: 'Dismiss' }));
    await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument());
  });

  it('before a Run shows the Run prompt instead of made-up predictions', async () => {
    mockFetch(monitoringRoute(preRunMonitoring()));
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    expect(await screen.findByTestId('run-empty-state')).toBeInTheDocument();
    expect(screen.queryByTestId('ml-panel')).not.toBeInTheDocument();
    expect(screen.queryByTestId('prediction-history')).not.toBeInTheDocument();
    expect(screen.queryByText(/is healthy!/)).not.toBeInTheDocument();
    // Raw sensor data and the dataset labels are still real.
    expect(screen.getByTestId('dataset-labels')).toBeInTheDocument();
  });

  it('renders n/a for missing sensor values', async () => {
    const m = monitoringFixture();
    m.plant.latestReading!.soil = null;
    m.plant.samplingPoint = null;
    mockFetch(monitoringRoute(m));
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    await screen.findByRole('heading', { level: 1, name: /Pot 5/ });
    expect(screen.getByText('Point n/a')).toBeInTheDocument();
    expect(screen.getByText(/this reading has no value for this sensor/)).toBeInTheDocument();
  });

  it('handles an unknown plant and an API failure', async () => {
    mockFetch({ 'GET /api/monitoring/plants/5': { status: 404, body: errorEnvelope('not_found', 'no such plant') } });
    const { unmount } = render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    expect(await screen.findByText('Plant Not Found')).toBeInTheDocument();
    unmount();

    mockFetch({ 'GET /api/monitoring/plants/5': { status: 500, body: errorEnvelope('internal_error', 'Boom') } });
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    expect(await screen.findByText('Boom')).toBeInTheDocument();
  });

  it('disables the status control and sends one PATCH while a change is pending', async () => {
    const m = monitoringFixture();
    const rec = m.recommendations[0];
    let release: () => void = () => undefined;
    const gate = new Promise<void>((r) => (release = r));
    const { calls } = mockFetch({
      ...monitoringRoute(m),
      [`PATCH /api/recommendations/${rec.id}`]: async () => {
        await gate;
        return { body: { ...rec, status: 'in_progress' } };
      },
    });
    render(<PlantDetailPage plantId={5} onNavigate={vi.fn()} />);
    await screen.findByText(rec.title);

    const select = screen.getByLabelText(`Status of ${rec.title}`);
    fireEvent.change(select, { target: { value: 'in_progress' } });
    await waitFor(() => expect(select).toBeDisabled());
    fireEvent.change(select, { target: { value: 'completed' } });
    expect(calls.filter((c) => c.key.startsWith('PATCH '))).toHaveLength(1);
    release();
    await waitFor(() => expect(select).toBeEnabled());
  });
});
