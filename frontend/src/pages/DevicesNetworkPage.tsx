import React from 'react';
import { Cpu, Wifi, RefreshCw } from 'lucide-react';
import { useDevices } from '../hooks/useData';
import { ErrorState } from '../components/shared/ErrorState';
import { formatDateTime, formatCount, orNA } from '../lib/utils';
import type { HardwareDetails } from '../types';

const SUGGESTED_DEVICE_VALUES: Record<string, { ipAddress: string; macAddress: string; hardware: Partial<HardwareDetails> }> = {
  'jetson-01': {
    ipAddress: '192.168.1.11',
    macAddress: '00:1A:2B:3C:4D:5E',
    hardware: {
      cpu_usage: 32,
      ram_used_gb: 3.8,
      ram_total_gb: 8,
      disk_used_gb: 18.6,
      disk_total_gb: 64,
      temperature_c: 48,
      uptime: '3 days, 6 hours',
      firmware_version: 'JetPack 6.2',
    },
  },
  'esp32-node-1': {
    ipAddress: '192.168.1.101',
    macAddress: '24:6F:28:A1:B2:C3',
    hardware: { rssi_dbm: -48, uptime: '3 days, 5 hours' },
  },
  'esp32-node-2': {
    ipAddress: '192.168.1.102',
    macAddress: '24:6F:28:D4:E5:F6',
    hardware: { rssi_dbm: -52, uptime: '3 days, 5 hours' },
  },
};

/** Percentage 0-100 for a "used / total" pair, or null when either side is missing. */
function usedPct(used: number | null, total: number | null): number | null {
  if (used === null || total === null || total <= 0) return null;
  return Math.max(0, Math.min(100, (used / total) * 100));
}

const Bar: React.FC<{ pct: number | null }> = ({ pct }) =>
  pct === null ? null : (
    <div className="w-full bg-[#173F32] h-1.5 rounded-full mt-2 overflow-hidden">
      <div className="bg-[#79B86A] h-full rounded-full" style={{ width: `${pct}%` }} />
    </div>
  );

export const DevicesNetworkPage: React.FC = () => {
  const devicesQuery = useDevices();
  const devices = devicesQuery.data ?? [];

  const gateways = devices.filter((d) => d.type === 'jetson');
  const espNodes = devices.filter((d) => d.type === 'esp32');

  return (
    <div className="space-y-6">
      {devicesQuery.error && <ErrorState message={devicesQuery.error.message} onRetry={() => void devicesQuery.refetch()} />}

      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">Edge Ingestion Hardware</span>
            {devices.length > 0 && (
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
                {devices.length} devices in the data ({espNodes.length} sensor nodes)
              </span>
            )}
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">Devices &amp; Sensor Nodes</h1>
          <p className="text-xs text-[#6B786E] mt-1">
            Suggested values are shown when live device telemetry is unavailable.
          </p>
        </div>

        <button
          onClick={() => void devicesQuery.refetch()}
          className="px-3.5 py-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-semibold flex items-center gap-2 transition-colors self-start md:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${devicesQuery.refreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {devicesQuery.loading && <div className="p-6 text-center text-xs text-stone-500">Loading devices...</div>}
      {!devicesQuery.loading && !devicesQuery.error && devices.length === 0 && (
        <div className="rounded-card border border-[#DCE4D9] bg-white p-8 text-center text-xs text-stone-500">No devices found in the data.</div>
      )}

      {/* Gateway */}
      {gateways.map((gw) => {
        const hw = gw.hardware_details;
        const suggested = SUGGESTED_DEVICE_VALUES[gw.id];
        const cpuUsage = hw.cpu_usage ?? suggested?.hardware.cpu_usage ?? null;
        const ramUsed = hw.ram_used_gb ?? suggested?.hardware.ram_used_gb ?? null;
        const ramTotal = hw.ram_total_gb ?? suggested?.hardware.ram_total_gb ?? null;
        const diskUsed = hw.disk_used_gb ?? suggested?.hardware.disk_used_gb ?? null;
        const diskTotal = hw.disk_total_gb ?? suggested?.hardware.disk_total_gb ?? null;
        const temperature = hw.temperature_c ?? suggested?.hardware.temperature_c ?? null;
        const uptime = hw.uptime ?? suggested?.hardware.uptime ?? null;
        const firmware = hw.firmware_version ?? suggested?.hardware.firmware_version ?? null;
        const ipAddress = gw.ip_address ?? suggested?.ipAddress ?? null;
        const macAddress = gw.mac_address ?? suggested?.macAddress ?? null;
        const usingSuggestedValues = [
          gw.ip_address, gw.mac_address, hw.cpu_usage, hw.ram_used_gb,
          hw.ram_total_gb, hw.disk_used_gb, hw.disk_total_gb,
          hw.temperature_c, hw.uptime, hw.firmware_version,
        ].some((value) => value === null);
        const noMetrics = [cpuUsage, ramUsed, diskUsed, temperature].every((v) => v === null);
        return (
          <div key={gw.id} data-testid={`device-${gw.id}`} className="bg-gradient-to-br from-[#102F26] to-[#173F32] rounded-card p-6 text-white shadow-xl">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2E6754] pb-4 mb-5">
              <div className="flex items-center gap-3.5">
                <div className="w-12 h-12 rounded-xl bg-[#2E8B57] text-white flex items-center justify-center shadow-lg">
                  <Cpu className="w-6 h-6 text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs uppercase font-bold tracking-wider text-[#79B86A]">Gateway</span>
                    {usingSuggestedValues && <span className="text-[10px] text-[#DDE8D8]/70">Suggested values</span>}
                  </div>
                  <h2 className="text-xl font-bold tracking-tight">{gw.name}</h2>
                  <div className="text-xs text-[#DDE8D8]/80 font-mono mt-0.5">
                    IP: {orNA(ipAddress)} • MAC: {orNA(macAddress)}
                  </div>
                </div>
              </div>

              <div className="text-right text-xs text-[#DDE8D8]/80 font-mono space-y-0.5">
                <div>Last data: {formatDateTime(gw.last_transmission)} IST</div>
                <div>Uptime: {orNA(uptime)}</div>
                <div>Firmware: {orNA(firmware)}</div>
              </div>
            </div>

            <p className="text-xs text-[#DDE8D8]/80 mb-4">
              {gw.simpleRole} • {formatCount(gw.readingCount)} readings • {gw.plantIds.length} pots
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
              <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
                <span className="text-[#DDE8D8]/70 block mb-1">CPU Load</span>
                <div className="text-xl font-bold font-mono text-white">{orNA(cpuUsage, '%')}</div>
                <Bar pct={cpuUsage} />
              </div>
              <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
                <span className="text-[#DDE8D8]/70 block mb-1">System Memory</span>
                <div className="text-xl font-bold font-mono text-white">
                  {ramUsed === null || ramTotal === null ? 'n/a' : `${ramUsed} / ${ramTotal} GB`}
                </div>
                <Bar pct={usedPct(ramUsed, ramTotal)} />
              </div>
              <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
                <span className="text-[#DDE8D8]/70 block mb-1">Storage</span>
                <div className="text-xl font-bold font-mono text-white">
                  {diskUsed === null || diskTotal === null ? 'n/a' : `${diskUsed} / ${diskTotal} GB`}
                </div>
                <Bar pct={usedPct(diskUsed, diskTotal)} />
              </div>
              <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
                <span className="text-[#DDE8D8]/70 block mb-1">Temperature</span>
                <div className="text-xl font-bold font-mono text-white">{orNA(temperature, '°C')}</div>
              </div>
            </div>
            {noMetrics && <p className="mt-3 text-[11px] text-[#DDE8D8]/70">No hardware metrics are available for this device.</p>}
          </div>
        );
      })}

      {/* Sensor nodes */}
      {espNodes.length > 0 && (
        <div className="space-y-4">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">Sensor Nodes</h2>
            <p className="text-xs text-[#6B786E]">Each node reads the plants listed on its card.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {espNodes.map((node) => {
              const suggested = SUGGESTED_DEVICE_VALUES[node.id];
              const ipAddress = node.ip_address ?? suggested?.ipAddress ?? null;
              const macAddress = node.mac_address ?? suggested?.macAddress ?? null;
              const rssi = node.hardware_details.rssi_dbm ?? suggested?.hardware.rssi_dbm ?? null;
              const uptime = node.hardware_details.uptime ?? suggested?.hardware.uptime ?? null;
              const usingSuggestedValues = [
                node.ip_address, node.mac_address, node.hardware_details.rssi_dbm,
                node.hardware_details.uptime,
              ].some((value) => value === null);
              return (
              <div
                key={node.id}
                data-testid={`device-${node.id}`}
                className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-start justify-between gap-3 mb-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-xl bg-[#EDF4EB] text-[#2E8B57] flex items-center justify-center font-bold">
                        <Wifi className="w-5 h-5" />
                      </div>
                      <div>
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">{node.id.toUpperCase()}</span>
                          {usingSuggestedValues && <span className="text-[10px] text-[#6B786E]">Suggested values</span>}
                        </div>
                        <h3 className="text-sm font-bold text-[#1C2B23]">{node.name}</h3>
                      </div>
                    </div>
                  </div>

                  <p className="text-xs text-[#6B786E] mb-4 bg-stone-50 p-2.5 rounded-lg border border-stone-200">{node.simpleRole}</p>

                  <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">IP Address</span>
                      <span className="font-mono text-[#1C2B23] font-medium">{orNA(ipAddress)}</span>
                    </div>
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">MAC Address</span>
                      <span className="font-mono text-stone-600">{orNA(macAddress)}</span>
                    </div>
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">Wi-Fi RSSI Signal</span>
                      <span className="font-semibold text-[#1C2B23]">{orNA(rssi, ' dBm')}</span>
                    </div>
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">Sampling Channels</span>
                      <span className="font-bold text-[#1C2B23]">{orNA(node.hardware_details.channels)}</span>
                    </div>
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">Readings Stored</span>
                      <span className="font-bold text-[#1C2B23]">{formatCount(node.readingCount)}</span>
                    </div>
                    <div>
                      <span className="text-stone-400 block text-[10px] uppercase font-semibold">Pots</span>
                      <span className="font-mono text-[#1C2B23]" data-testid={`plant-ids-${node.id}`}>
                        {node.plantIds.length > 0 ? node.plantIds.join(', ') : 'n/a'}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="pt-3 border-t border-[#DCE4D9] flex flex-wrap items-center justify-between gap-2 text-[11px] text-[#6B786E] font-mono">
                  <span>Last data: {formatDateTime(node.last_transmission)} IST</span>
                  <span>Uptime: {orNA(uptime)}</span>
                </div>
              </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};
