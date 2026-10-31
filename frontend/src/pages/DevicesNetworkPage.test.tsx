import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { DevicesNetworkPage } from './DevicesNetworkPage';
import { mockFetch } from '../test/helpers';
import { devicesFixture, errorEnvelope } from '../test/monitoringFixtures';

afterEach(() => vi.unstubAllGlobals());

const route = (devices: unknown) => ({ 'GET /api/devices': { body: devices, headers: { 'X-Total-Count': '3' } } });

describe('DevicesNetworkPage', () => {
  it('renders suggested device details when live hardware telemetry is unavailable', async () => {
    mockFetch(route(devicesFixture()));
    render(<DevicesNetworkPage />);

    const node1 = await screen.findByTestId('device-esp32-node-1');
    expect(within(node1).getByText('192.168.1.101')).toBeInTheDocument();
    expect(within(node1).getByText('24:6F:28:A1:B2:C3')).toBeInTheDocument();
    expect(within(node1).getByText('-48 dBm')).toBeInTheDocument();
    expect(within(node1).getByText('Uptime: 3 days, 5 hours')).toBeInTheDocument();
    expect(within(node1).getByText('Suggested values')).toBeInTheDocument();
    expect(within(node1).queryByText(/Status unknown|heartbeat/i)).not.toBeInTheDocument();
    expect(within(node1).getByText('2,688')).toBeInTheDocument();
    expect(within(node1).getByText('12')).toBeInTheDocument(); // channels from data
    expect(screen.getByTestId('plant-ids-esp32-node-1')).toHaveTextContent('1, 3, 5, 7, 9, 11, 13, 15');
    expect(screen.getByTestId('plant-ids-esp32-node-2')).toHaveTextContent('2, 4, 6, 8, 10, 12, 14, 16');
    expect(within(node1).getByText(/Last data: .*Oct.* IST/)).toBeInTheDocument();

    const gw = screen.getByTestId('device-jetson-01');
    expect(within(gw).getByText('Jetson Gateway')).toBeInTheDocument(); // the device's own name from the API
    expect(within(gw).getByText(/IP: 192\.168\.1\.11 • MAC: 00:1A:2B:3C:4D:5E/)).toBeInTheDocument();
    expect(within(gw).getByText('Uptime: 3 days, 6 hours')).toBeInTheDocument();
    expect(within(gw).getByText('Firmware: JetPack 6.2')).toBeInTheDocument();
    expect(within(gw).getByText('32%')).toBeInTheDocument();
    expect(within(gw).getByText('3.8 / 8 GB')).toBeInTheDocument();
    expect(within(gw).getByText('18.6 / 64 GB')).toBeInTheDocument();
    expect(within(gw).getByText('48°C')).toBeInTheDocument();
    expect(within(gw).getByText(/5,376 readings • 16 pots/)).toBeInTheDocument();
    expect(within(gw).getByText('Suggested values')).toBeInTheDocument();

    expect(within(gw).queryByText(/Heartbeat/i)).not.toBeInTheDocument();
    expect(screen.queryByTestId('status-badge')).not.toBeInTheDocument();

    const node2 = screen.getByTestId('device-esp32-node-2');
    expect(within(node2).getByText('192.168.1.102')).toBeInTheDocument();
    expect(within(node2).getByText('24:6F:28:D4:E5:F6')).toBeInTheDocument();
    expect(within(node2).getByText('-52 dBm')).toBeInTheDocument();
    expect(within(node2).getByText('Uptime: 3 days, 5 hours')).toBeInTheDocument();
    expect(screen.queryByText(/3 Configured Nodes|Port 5000/)).not.toBeInTheDocument();
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
