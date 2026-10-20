import React, { useState, useEffect } from 'react';
import {
  Settings,
  Server,
  Cpu,
  Clock,
  Database,
  Sliders,
  Check,
  AlertCircle,
  HelpCircle,
} from 'lucide-react';
import { apiClient } from '../api/client';
import { SystemSettings } from '../types';

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SystemSettings | null>(null);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSettings();
  }, []);

  const loadSettings = async () => {
    setLoading(true);
    try {
      const s = await apiClient.getSettings();
      setSettings(s);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!settings) return;
    await apiClient.updateSettings(settings);
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  if (!settings) return null;

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">
              System Configuration
            </span>
            <span className="text-xs px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-semibold border border-[#DDE8D8]">
              Jetson Local State
            </span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">
            Calibration, Hardware Mapping & Telemetry Settings
          </h1>
          <p className="text-xs text-[#6B786E] mt-0.5">
            Parameters governing sequential sampling schedules, ADC calibration curves, and API endpoints.
          </p>
        </div>

        {savedSuccess && (
          <div className="flex items-center gap-1.5 text-xs font-bold text-[#2E8B57] bg-[#EDF4EB] px-3 py-1.5 rounded-lg border border-[#DDE8D8] animate-fadeIn">
            <Check className="w-4 h-4" />
            <span>Settings Saved</span>
          </div>
        )}
      </div>

      <form onSubmit={handleSave} className="space-y-6">
        {/* Backend Endpoint Settings */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Server className="w-4 h-4 text-[#2E8B57]" />
            <span>Ingestion Receiver API Endpoint</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Jetson Base Receiver URL:
              </label>
              <input
                type="text"
                value={settings.api_base_url}
                onChange={(e) => setSettings({ ...settings, api_base_url: e.target.value })}
                className="w-full bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg px-3 py-2 font-mono text-xs outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
                placeholder="http://localhost:5000"
              />
              <span className="text-[10px] text-[#6B786E] block mt-1">
                Flask receiver daemon listening on port 5000 on Jetson Orin Nano.
              </span>
            </div>

            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Live Polling Frequency (seconds):
              </label>
              <select
                value={settings.polling_interval_seconds}
                onChange={(e) =>
                  setSettings({ ...settings, polling_interval_seconds: Number(e.target.value) })
                }
                className="w-full bg-white border border-[#DCE4D9] rounded-lg px-3 py-2 text-xs outline-none focus:border-[#2E8B57] text-[#1C2B23] cursor-pointer"
              >
                <option value={2}>2 seconds (High frequency)</option>
                <option value={3}>3 seconds (Recommended)</option>
                <option value={5}>5 seconds</option>
                <option value={10}>10 seconds</option>
                <option value={30}>30 seconds</option>
              </select>
            </div>
          </div>
        </div>

        {/* Hardware & Calibration Parameters */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#2E8B57]" />
            <span>Sensor Hardware & Soil ADC Calibration Points</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Dry Air Calibration Value (ADC Counts):
              </label>
              <input
                type="number"
                value={settings.soil_adc_dry}
                onChange={(e) =>
                  setSettings({ ...settings, soil_adc_dry: Number(e.target.value) })
                }
                className="w-full bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg px-3 py-2 font-mono text-xs outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
              />
              <span className="text-[10px] text-[#6B786E] block mt-1">
                Sensor exposed to ambient air corresponds to 0% moisture.
              </span>
            </div>

            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Wet Water Immersion Value (ADC Counts):
              </label>
              <input
                type="number"
                value={settings.soil_adc_wet}
                onChange={(e) =>
                  setSettings({ ...settings, soil_adc_wet: Number(e.target.value) })
                }
                className="w-full bg-[#F6F7F0] border border-[#DCE4D9] rounded-lg px-3 py-2 font-mono text-xs outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]"
              />
              <span className="text-[10px] text-[#6B786E] block mt-1">
                Submerged in cup of water corresponds to 100% saturation.
              </span>
            </div>
          </div>

          {/* Optional pH Sensor Configuration (Explicitly unconfigured by default per prompt) */}
          <div className="pt-4 border-t border-[#DCE4D9] flex items-center justify-between text-xs">
            <div>
              <span className="font-semibold text-[#1C2B23] block">
                Substrate pH Channel (Optional Sensor Probe)
              </span>
              <span className="text-[#6B786E] text-[11px]">
                Currently marked as not installed / unconfigured. Enable only if physical analog pH probe is connected.
              </span>
            </div>

            <button
              type="button"
              onClick={() =>
                setSettings({ ...settings, ph_sensor_installed: !settings.ph_sensor_installed })
              }
              className={`px-3 py-1.5 rounded-lg font-bold text-xs transition-colors border ${
                settings.ph_sensor_installed
                  ? 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]'
                  : 'bg-stone-100 text-stone-600 border-stone-200'
              }`}
            >
              {settings.ph_sensor_installed ? 'Installed & Active' : 'Not Installed (Default)'}
            </button>
          </div>
        </div>

        {/* Collection Schedule & Storage Retention */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#2E8B57]" />
            <span>Sampling Schedule & Storage Retention</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Scheduled Daily Passes (3 readings per plant):
              </label>
              <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 font-mono text-xs text-[#1C2B23]">
                08:00 UTC (Morning) • 13:00 UTC (Midday) • 18:00 UTC (Evening)
              </div>
            </div>

            <div>
              <label className="block font-semibold text-[#1C2B23] mb-1">
                Local Telemetry Retention:
              </label>
              <div className="p-3 bg-stone-50 rounded-lg border border-stone-200 font-mono text-xs text-[#1C2B23]">
                30 Days ({settings.storage_path})
              </div>
            </div>
          </div>
        </div>

        {/* Save button */}
        <div className="flex justify-end">
          <button
            type="submit"
            className="px-6 py-2.5 bg-[#2E8B57] hover:bg-[#236e44] text-white text-xs font-bold rounded-lg transition-colors shadow-sm"
          >
            Save Configuration Changes
          </button>
        </div>
      </form>
    </div>
  );
};
