import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, within } from '@testing-library/react';
import { ShapAnalysisPage } from './ShapAnalysisPage';
import { ShapWaterfallChart } from '../components/charts/ShapWaterfallChart';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, globalShapFixture, localShapFixture, plantsFixture } from '../test/fixtureHelpers';

afterEach(() => vi.unstubAllGlobals());

const HINT = 'cd money_plant_ml_pipeline && python scripts/generate_explanations.py --input data/raw/plant_sensor_data.csv --run-id v1.0.0_20261021T145244Z';

const plantsBody = () => ({ body: plantsFixture(), headers: { 'X-Total-Count': '16' } });

describe('ShapWaterfallChart', () => {
  it('renders 5 real features with a null base value (starts at 0, log-odds)', () => {
    const explanation = localShapFixture();
    expect(explanation.features).toHaveLength(5);
    expect(explanation.base_value).toBeNull();
    render(<ShapWaterfallChart explanation={explanation} />);

    expect(screen.getAllByTestId('shap-feature-row')).toHaveLength(5);
    expect(screen.getByText('Soil moisture, 12-reading average')).toBeInTheDocument();
    expect(screen.getByText('Reading: 8.31 %')).toBeInTheDocument();
    expect(screen.getByText('+2.31 log-odds')).toBeInTheDocument();
    expect(screen.getByText('-0.66 log-odds')).toBeInTheDocument();
    expect(screen.getByTestId('shap-probability')).toHaveTextContent('62%');
    expect(screen.getByText(/Model rating: high \(model probability 62%\)/)).toBeInTheDocument();
    expect(screen.getByTestId('shap-baseline-note')).toHaveTextContent(
      'Contributions relative to model baseline (not provided by model service). The waterfall starts at 0.',
    );
    // Layman summary and the explanation note are shown verbatim.
    expect(screen.getByText(/the model rated stress 'high'/)).toBeInTheDocument();
    expect(screen.getByText(/For the reading at 18 Oct 2026, 22:01 IST/)).toBeInTheDocument();
    expect(screen.queryByText(/\d{4}-\d{2}-\d{2}T\d{2}:/)).not.toBeInTheDocument();
    expect(screen.getByText(/Units: log-odds contribution/)).toBeInTheDocument();
  });

  it('handles any number of features (3 here), not just 4 or 5', () => {
    const explanation = localShapFixture();
    explanation.features = explanation.features.slice(0, 3);
    render(<ShapWaterfallChart explanation={explanation} />);
    expect(screen.getAllByTestId('shap-feature-row')).toHaveLength(3);
    expect(screen.getByText(/Top 3 contributions/)).toBeInTheDocument();
  });
});

describe('ShapAnalysisPage', () => {
  it('loads the local explanation for the selected plant', async () => {
    const { calls } = mockFetch({
      'GET /api/plants': plantsBody(),
      'GET /api/ai/shap/5': { body: localShapFixture() },
      'GET /api/ai/shap/global': { body: globalShapFixture() },
    });
    render(<ShapAnalysisPage initialPlantId={5} onNavigate={vi.fn()} />);
    expect(await screen.findAllByTestId('shap-feature-row')).toHaveLength(5);
    expect(screen.queryByText(/TreeSHAP|Jetson/)).not.toBeInTheDocument();
    expect(calls.some((c) => c.key === 'GET /api/ai/shap/5')).toBe(true);

    // All 16 plants are selectable.
    const select = (await screen.findByRole('combobox')) as HTMLSelectElement;
    await screen.findByRole('option', { name: /^Pot 16: / });
    expect(select.options).toHaveLength(16);
    // The plant name already starts with "Pot N:", so the option must not repeat the prefix.
    expect(select.options[2].textContent).toBe('Pot 3: Rama Tulsi (Control Group) (Cluster 1)');
    expect(Array.from(select.options).some((o) => /Pot #\d+: Pot/.test(o.textContent ?? ''))).toBe(false);
  });

  it('shows the Run empty state when no local explanation exists yet (404)', async () => {
    mockFetch({
      'GET /api/plants': plantsBody(),
      'GET /api/ai/shap/5': { status: 404, body: errorEnvelope('explanation_unavailable', 'no explanation yet') },
      'GET /api/ai/shap/global': { body: globalShapFixture() },
    });
    render(<ShapAnalysisPage initialPlantId={5} onNavigate={vi.fn()} />);
    expect(await screen.findByTestId('run-empty-state')).toHaveTextContent('Press Run to analyze the bundled plant dataset.');
  });

  it('global view: top-15 list, per-class view, labelled sample beeswarm', async () => {
    mockFetch({
      'GET /api/plants': plantsBody(),
      'GET /api/ai/shap/5': { body: localShapFixture() },
      'GET /api/ai/shap/global': { body: globalShapFixture() },
    });
    render(<ShapAnalysisPage initialPlantId={5} onNavigate={vi.fn()} />);
    await screen.findAllByTestId('shap-feature-row');
    fireEvent.click(screen.getByRole('button', { name: /Global Feature Impact/ }));

    const list = await screen.findByTestId('shap-global-features');
    expect(within(list).getAllByRole('listitem')).toHaveLength(15);
    expect(within(list).getByText('0.462')).toBeInTheDocument();
    expect(screen.getByTestId('beeswarm-approximate')).toHaveTextContent('sample of 50 test rows, top-5 features each');
    expect(screen.getByTestId('shap-beeswarm')).toBeInTheDocument();

    // Per-class view switches the list to that class's own top 15.
    fireEvent.change(screen.getByLabelText('Class view:'), { target: { value: 'healthy' } });
    const healthyTop = globalShapFixture().perClassTop15.healthy[0];
    expect(within(screen.getByTestId('shap-global-features')).getByText(healthyTop.mean_abs_shap.toFixed(3))).toBeInTheDocument();
    expect(screen.getByText(/for class 'healthy'/)).toBeInTheDocument();
  });

  it('global view: a missing report (404) shows the command hint', async () => {
    mockFetch({
      'GET /api/plants': plantsBody(),
      'GET /api/ai/shap/5': { body: localShapFixture() },
      'GET /api/ai/shap/global': {
        status: 404,
        body: errorEnvelope('global_shap_unavailable', 'Global SHAP report not found', { hint: HINT }),
      },
    });
    render(<ShapAnalysisPage initialPlantId={5} onNavigate={vi.fn()} />);
    await screen.findAllByTestId('shap-feature-row');
    fireEvent.click(screen.getByRole('button', { name: /Global Feature Impact/ }));
    const box = await screen.findByTestId('global-shap-unavailable');
    expect(box).toHaveTextContent('The global SHAP report has not been generated yet.');
    expect(box).toHaveTextContent(HINT);
  });
});
