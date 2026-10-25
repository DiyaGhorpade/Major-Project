import { afterEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { SettingsPage } from './SettingsPage';
import { mockFetch } from '../test/helpers';
import { errorEnvelope, settingsFixture } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

describe('SettingsPage', () => {
  it('renders the stored values and derives schedule text from them (no demo/API-URL fields)', async () => {
    mockFetch({ 'GET /api/settings': { body: settingsFixture() } });
    render(<SettingsPage />);

    expect(await screen.findByLabelText(/Refresh interval/)).toHaveValue(5);
    expect(screen.getByLabelText(/Dry Air/)).toHaveValue(3200);
    expect(screen.getByLabelText(/Wet Water/)).toHaveValue(1450);
    expect(screen.getByLabelText(/retention/)).toHaveValue(30);
    expect(screen.getByLabelText(/collection times/)).toHaveValue('00:00, 02:00, 04:00, 06:00, 08:00, 10:00, 12:00, 14:00, 16:00, 18:00, 20:00, 22:00');
    expect(screen.getByText('Currently every 2 h (12 per day).')).toBeInTheDocument();
    expect(screen.getByText('backend/data/app.db')).toBeInTheDocument();
    expect(screen.queryByText(/08:00 UTC/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Jetson/)).not.toBeInTheDocument();
    expect(screen.queryByText(/API/)).not.toBeInTheDocument();
    expect(screen.queryByText(/demo/i)).not.toBeInTheDocument();
  });

  it('saves only changed fields via PUT and confirms', async () => {
    let putBody: unknown = null;
    const stored = settingsFixture();
    mockFetch({
      'GET /api/settings': () => ({ body: stored }),
      'PUT /api/settings': (init) => {
        putBody = JSON.parse(String(init?.body));
        Object.assign(stored, putBody);
        return { body: stored };
      },
    });
    render(<SettingsPage />);
    fireEvent.change(await screen.findByLabelText(/Refresh interval/), { target: { value: '12' } });
    fireEvent.click(screen.getByRole('button', { name: 'Not Installed' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration Changes' }));

    expect(await screen.findByText('Settings Saved')).toBeInTheDocument();
    expect(putBody).toEqual({ polling_interval_seconds: 12, ph_sensor_installed: true });
    await waitFor(() => expect(screen.getByLabelText(/Refresh interval/)).toHaveValue(12));
  });

  it('shows 422 validation messages next to the field', async () => {
    mockFetch({
      'GET /api/settings': { body: settingsFixture() },
      'PUT /api/settings': {
        status: 422,
        body: errorEnvelope('validation_error', 'Request validation failed', [
          { loc: ['body', 'soil_adc_dry'], message: 'Input should be less than or equal to 1000000', type: 'less_than_equal' },
        ]),
      },
    });
    render(<SettingsPage />);
    fireEvent.change(await screen.findByLabelText(/Dry Air/), { target: { value: '99999999' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration Changes' }));

    expect(await screen.findByText('Some settings are invalid')).toBeInTheDocument();
    expect(screen.getByText('Input should be less than or equal to 1000000')).toBeInTheDocument();
    expect(screen.queryByText('Settings Saved')).not.toBeInTheDocument();
  });

  it('rejects non-integers on the client without calling the API', async () => {
    const { calls } = mockFetch({ 'GET /api/settings': { body: settingsFixture() } });
    render(<SettingsPage />);
    fireEvent.change(await screen.findByLabelText(/Refresh interval/), { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration Changes' }));
    expect(await screen.findByText('Enter a whole number.')).toBeInTheDocument();
    expect(calls.some((c) => c.key === 'PUT /api/settings')).toBe(false);
  });

  it('shows an error when settings cannot be loaded', async () => {
    mockFetch({ 'GET /api/settings': { status: 500, body: errorEnvelope('internal_error', 'Settings down') } });
    render(<SettingsPage />);
    expect(await screen.findByText('Settings down')).toBeInTheDocument();
  });
});
