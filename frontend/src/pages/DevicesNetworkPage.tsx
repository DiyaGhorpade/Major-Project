import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Wifi,
  RefreshCw,
} from 'lucide-react';
import { useDevices } from '../hooks/useData';
import { StatusBadge } from '../components/shared/StatusBadge';
import { formatRelativeTime, orNA } from '../lib/utils';

export const DevicesNetworkPage: React.FC = () => {
  const devicesQuery = useDevices();
  const devices = devicesQuery.data ?? [];
  const loading = devicesQuery.loading;

  const jetson = devices.find((d) => d.type === 'jetson') || devices[0];
  const espNodes = devices.filter((d) => d.type === 'esp32');

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              Edge Ingestion Hardware Infrastructure
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              3 Configured Nodes
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            Devices & Local Area Telemetry Network
          </h1>
          <p className="text-xs text-[#6B786E] mt-1">
            Gateway and ESP32 sensor nodes derived from the bundled dataset. The dataset carries no live heartbeat or hardware metrics, so those show n/a.
          </p>
        </div>

        <button
          onClick={() => void devicesQuery.refetch()}
          className="px-3.5 py-2 rounded-lg bg-stone-100 hover:bg-stone-200 text-stone-700 text-xs font-semibold flex items-center gap-2 transition-colors self-start md:self-auto"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Jetson Orin Nano Dedicated Card */}
      {jetson && (
        <div className="bg-gradient-to-br from-[#102F26] to-[#173F32] rounded-card p-6 text-white shadow-xl">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#2E6754] pb-4 mb-5">
            <div className="flex items-center gap-3.5">
              <div className="w-12 h-12 rounded-xl bg-[#2E8B57] text-white flex items-center justify-center shadow-lg">
                <Cpu className="w-6 h-6 text-white" />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-0.5">
                  <span className="text-xs uppercase font-bold tracking-wider text-[#79B86A]">
                    Primary Edge Ingestion Host
                  </span>
                  <StatusBadge status={jetson.status} size="sm" />
                </div>
                <h2 className="text-xl font-bold tracking-tight">{jetson.name}</h2>
                <div className="text-xs text-[#DDE8D8]/80 font-mono mt-0.5">
                  IP: {orNA(jetson.ip_address)} • MAC: {orNA(jetson.mac_address)}
                </div>
              </div>
            </div>

            <div className="text-right text-xs text-[#DDE8D8]/80 font-mono">
              <div>Uptime: {orNA(jetson.hardware_details.uptime)}</div>
              <div>Firmware: {orNA(jetson.hardware_details.firmware_version)}</div>
            </div>
          </div>

          {/* Jetson Core Telemetry Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
            <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
              <span className="text-[#DDE8D8]/70 block mb-1">CPU Load</span>
              <div className="text-xl font-bold font-mono text-white">
                {orNA(jetson.hardware_details.cpu_usage, '%')}
              </div>
              <div className="w-full bg-[#173F32] h-1.5 rounded-full mt-2 overflow-hidden">
                <div
                  className="bg-[#79B86A] h-full rounded-full"
                  style={{ width: `${jetson.hardware_details.cpu_usage ?? 0}%` }}
                />
              </div>
            </div>

            <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
              <span className="text-[#DDE8D8]/70 block mb-1">System Memory</span>
              <div className="text-xl font-bold font-mono text-white">
                {jetson.hardware_details.ram_used_gb === null || jetson.hardware_details.ram_total_gb === null ? 'n/a' : `${jetson.hardware_details.ram_used_gb} / ${jetson.hardware_details.ram_total_gb} GB`}
              </div>
              <div className="w-full bg-[#173F32] h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-[#79B86A] h-full rounded-full" style={{ width: '40%' }} />
              </div>
            </div>

            <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
              <span className="text-[#DDE8D8]/70 block mb-1">NVMe Storage</span>
              <div className="text-xl font-bold font-mono text-white">
                {jetson.hardware_details.disk_used_gb === null || jetson.hardware_details.disk_total_gb === null ? 'n/a' : `${jetson.hardware_details.disk_used_gb} / ${jetson.hardware_details.disk_total_gb} GB`}
              </div>
              <div className="w-full bg-[#173F32] h-1.5 rounded-full mt-2 overflow-hidden">
                <div className="bg-[#79B86A] h-full rounded-full" style={{ width: '20%' }} />
              </div>
            </div>

            <div className="bg-[#102F26]/70 p-3.5 rounded-xl border border-[#2E6754]">
              <span className="text-[#DDE8D8]/70 block mb-1">Die Temperature</span>
              <div className="text-xl font-bold font-mono text-white">
                {orNA(jetson.hardware_details.temperature_c, '°C')}
              </div>
              
            </div>
          </div>
        </div>
      )}

      {/* ESP32 Sensor Nodes Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-[#1C2B23]">
              Configured ESP32 Sequential Sensor Nodes
            </h2>
            <p className="text-xs text-[#6B786E]">
              Microcontroller nodes sampling analog soil probes and I2C/SPI environmental sensors.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {espNodes.map((node) => (
            <div
              key={node.id}
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
                        <span className="text-[10px] font-bold uppercase tracking-wider text-[#426B4F]">
                          {node.id.toUpperCase()}
                        </span>
                        <StatusBadge status={node.status} size="sm" />
                      </div>
                      <h3 className="text-sm font-bold text-[#1C2B23]">{node.name}</h3>
                    </div>
                  </div>
                </div>

                <p className="text-xs text-[#6B786E] mb-4 bg-stone-50 p-2.5 rounded-lg border border-stone-200">
                  {node.role}
                </p>

                <div className="grid grid-cols-2 gap-3 text-xs mb-4">
                  <div>
                    <span className="text-stone-400 block text-[10px] uppercase font-semibold">IP Address</span>
                    <span className="font-mono text-[#1C2B23] font-medium">{orNA(node.ip_address)}</span>
                  </div>
                  <div>
                    <span className="text-stone-400 block text-[10px] uppercase font-semibold">MAC Address</span>
                    <span className="font-mono text-stone-600">{orNA(node.mac_address)}</span>
                  </div>
                  <div>
                    <span className="text-stone-400 block text-[10px] uppercase font-semibold">Wi-Fi RSSI Signal</span>
                    <span className="font-semibold text-[#2E8B57]">
                      {orNA(node.hardware_details.rssi_dbm, ' dBm')}
                    </span>
                  </div>
                  <div>
                    <span className="text-stone-400 block text-[10px] uppercase font-semibold">Sampling Channels</span>
                    <span className="font-bold text-[#1C2B23]">
                      {orNA(node.hardware_details.channels, ' Channels')}
                    </span>
                  </div>
                </div>
              </div>

              <div className="pt-3 border-t border-[#DCE4D9] flex items-center justify-between text-[11px] text-[#6B786E] font-mono">
                <span>Last Tx: {formatRelativeTime(node.last_transmission)}</span>
                <span>Uptime: {orNA(node.hardware_details.uptime)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
