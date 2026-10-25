import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { DevicesNetworkPage } from './DevicesNetworkPage';
import { mockFetch } from '../test/helpers';
import { devicesFixture, errorEnvelope } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

const route = (devices: unknown) => ({ 'GET /api/devices': { body: devices, headers: { 'X-Total-Count': '3' } } });

describe('DevicesNetworkPage', () => {
  it('renders null hardware fields as n/a, explains unknown status and shows real counts', async () => {
    mockFetch(route(devicesFixture()));
    const { container } = render(<DevicesNetworkPage />);

    const node1 = await screen.findByTestId('device-esp32-node-1');
    expect(within(node1).getAllByText('n/a').length).toBeGreaterThanOrEqual(3); // ip, mac, rssi, uptime
    expect(within(node1).getByText('Status unknown: no live heartbeat (historical dataset).')).toBeInTheDocument();
    expect(within(node1).getByText('2,688')).toBeInTheDocument();
    expect(within(node1).getByText('12')).toBeInTheDocument(); // channels from data
    expect(screen.getByTestId('plant-ids-esp32-node-1')).toHaveTextContent('1, 3, 5, 7, 9, 11, 13, 15');
    expect(screen.getByTestId('plant-ids-esp32-node-2')).toHaveTextContent('2, 4, 6, 8, 10, 12, 14, 16');
    expect(within(node1).getByText(/Last data: .*Oct.* IST/)).toBeInTheDocument();

    const gw = screen.getByTestId('device-jetson-01');
    expect(within(gw).getByText('Jetson Gateway')).toBeInTheDocument(); // the device's own name from the API
    expect(within(gw).getByText('The bundled dataset carries no hardware metrics for this device.')).toBeInTheDocument();

    // No fake wording or fake usage bars.
    expect(screen.queryByText(/3 Configured Nodes/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Port 5000/)).not.toBeInTheDocument();
    expect(container.querySelector('[style*="width: 40%"], [style*="width: 20%"]')).toBeNull();
    expect(container.querySelector('.bg-\\[\\#79B86A\\].h-full')).toBeNull();
  });

  it('draws usage bars only when the API supplies the numbers', async () => {
    const devices = devicesFixture();
    const gw = devices.find((d) => d.type === 'jetson')!;
    gw.hardware_details.ram_used_gb = 2;
    gw.hardware_details.ram_total_gb = 8;
    mockFetch(route(devices));
    const { container } = render(<DevicesNetworkPage />);
    await screen.findByTestId('device-jetson-01');
    expect(screen.getByText('2 / 8 GB')).toBeInTheDocument();
    expect(container.querySelector('[style*="width: 25%"]')).not.toBeNull();
  });

  it('shows empty and error states', async () => {
    mockFetch(route([]));
    const { unmount } = render(<DevicesNetworkPage />);
    expect(await screen.findByText('No devices found in the data.')).toBeInTheDocument();
    unmount();

    mockFetch({ 'GET /api/devices': { status: 500, body: errorEnvelope('internal_error', 'Devices down') } });
    render(<DevicesNetworkPage />);
    expect(await screen.findByText('Devices down')).toBeInTheDocument();
  });
});
