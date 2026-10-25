import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { PlantsPage } from './PlantsPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, plantsFixture, preRunPlants } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

const route = (plants: unknown) => ({ 'GET /api/plants': { body: plants, headers: { 'X-Total-Count': '4' } } });

describe('PlantsPage', () => {
  it('shows group, ML status, heuristic guess with confidence, env state and anomaly tips', async () => {
    mockFetch(route(plantsFixture()));
    render(<PlantsPage onNavigate={vi.fn()} />);

    const card = await screen.findByTestId('plant-card-5');
    expect(within(card).getByText('Cluster 2: Water Stress')).toBeInTheDocument();
    expect(within(card).getByText('Severe stress')).toBeInTheDocument();
    expect(within(card).getByText('High stress')).toBeInTheDocument();
    expect(within(card).getByText('62%')).toBeInTheDocument();
    expect(within(card).getByText('Water Stress')).toBeInTheDocument();
    expect(within(card).getByText('Environmental state group 1')).toBeInTheDocument();
    expect(within(card).getByText('Normal reading')).toBeInTheDocument();
    expect(within(card).getByRole('note', { name: 'Environmental state group. Not a stress level.' })).toBeInTheDocument();
    expect(within(card).getByRole('note', { name: 'Unusual reading, check the sensor. Not a plant-health verdict.' })).toBeInTheDocument();
    expect(within(card).getByRole('note', { name: /Heuristic guess of the stress type/ })).toBeInTheDocument();
    expect(within(card).getByText('Node 1')).toBeInTheDocument();
    expect(screen.queryByText(/Pots 1–12|Pots 13–16/)).not.toBeInTheDocument();
  });

  it('filters by group and node using options built from the data', async () => {
    mockFetch(route(plantsFixture()));
    render(<PlantsPage onNavigate={vi.fn()} />);
    await screen.findByTestId('plant-card-5');
    expect(screen.getByRole('option', { name: 'Node 1 (4 pots)' })).toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue(/All experimental groups/), { target: { value: 'Cluster 1' } });
    expect(screen.queryByTestId('plant-card-5')).not.toBeInTheDocument();
    expect(screen.getByTestId('plant-card-1')).toBeInTheDocument();
  });

  it('before a Run shows Not analyzed yet and a null-safe latest reading', async () => {
    const plants = preRunPlants();
    plants[0].latestReading = null;
    mockFetch(route(plants));
    render(<PlantsPage onNavigate={vi.fn()} />);

    const card = await screen.findByTestId('plant-card-5');
    expect(within(card).getAllByText(/Not analyzed yet/).length).toBeGreaterThan(0);
    expect(within(card).getByText('No reading stored.')).toBeInTheDocument();
    expect(within(card).getByText('No reading')).toBeInTheDocument();
    expect(screen.getAllByText(/Press Run to analyze the bundled plant dataset\./).length).toBeGreaterThan(0);
  });

  it('renders n/a in list view for missing values', async () => {
    const plants = plantsFixture();
    plants[0].latestReading!.soil = null;
    plants[0].samplingPoint = null;
    mockFetch(route(plants));
    render(<PlantsPage onNavigate={vi.fn()} />);
    await screen.findByTestId('plant-card-5');
    fireEvent.click(screen.getByTitle('Dense List View'));
    const row = screen.getByText('Pot 5: Rama Tulsi (Watering Trial)').closest('tr')!;
    expect(within(row).getAllByText('n/a').length).toBeGreaterThan(0);
    expect(within(row).getByText(/Point n\/a/)).toBeInTheDocument();
  });

  it('shows the API error', async () => {
    mockFetch({ 'GET /api/plants': { status: 500, body: errorEnvelope('internal_error', 'Plants unavailable') } });
    render(<PlantsPage onNavigate={vi.fn()} />);
    expect(await screen.findByText('Plants unavailable')).toBeInTheDocument();
  });
});
