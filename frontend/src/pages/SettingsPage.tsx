import React, { useEffect, useState } from 'react';
import { Server, Clock, Sliders, Check } from 'lucide-react';
import { useSettings } from '../hooks/useSettings';
import { ErrorState } from '../components/shared/ErrorState';
import { describeCadence } from '../components/monitoring/format';
import type { ApiError } from '../api/client';
import type { SystemSettings } from '../types';

/** Editable form state (numbers kept as text while typing). */
interface Draft {
  polling_interval_seconds: string;
  soil_adc_dry: string;
  soil_adc_wet: string;
  log_retention_days: string;
  collection_times: string;
  ph_sensor_installed: boolean;
}

function toDraft(s: SystemSettings): Draft {
  return {
    polling_interval_seconds: String(s.polling_interval_seconds),
    soil_adc_dry: String(s.soil_adc_dry),
    soil_adc_wet: String(s.soil_adc_wet),
    log_retention_days: String(s.log_retention_days),
    collection_times: s.collection_times.join(', '),
    ph_sensor_installed: s.ph_sensor_installed,
  };
}

const parseTimes = (text: string): string[] => text.split(/[\s,]+/).filter(Boolean);

const INT_FIELDS = ['polling_interval_seconds', 'soil_adc_dry', 'soil_adc_wet', 'log_retention_days'] as const;

/** Only the changed fields, plus client-side errors for values that are not whole numbers. */
function buildPatch(draft: Draft, current: SystemSettings): { patch: Partial<SystemSettings>; errors: Record<string, string> } {
  const patch: Partial<SystemSettings> = {};
  const errors: Record<string, string> = {};
  for (const key of INT_FIELDS) {
    const text = draft[key].trim();
    const n = Number(text);
    if (text === '' || !Number.isInteger(n)) errors[key] = 'Enter a whole number.';
    else if (n !== current[key]) patch[key] = n;
  }
  const times = parseTimes(draft.collection_times);
  if (times.join(',') !== current.collection_times.join(',')) patch.collection_times = times;
  if (draft.ph_sensor_installed !== current.ph_sensor_installed) patch.ph_sensor_installed = draft.ph_sensor_installed;
  return { patch, errors };
}

/** Map a 422 `validation_error` envelope (details: [{loc: [..., field], message}]) to field -> message. */
function fieldErrorsFrom(error: ApiError | null): Record<string, string> {
  const out: Record<string, string> = {};
  if (!error || !Array.isArray(error.details)) return out;
  for (const d of error.details as Array<{ loc?: unknown[]; message?: string }>) {
    const field = Array.isArray(d.loc) && d.loc.length > 0 ? String(d.loc[d.loc.length - 1]) : null;
    if (field && d.message) out[field] = d.message;
  }
  return out;
}

const inputCls =
  'w-full bg-[#F6F7F0] border rounded-lg px-3 py-2 font-mono text-xs outline-none focus:border-[#2E8B57] focus:bg-white text-[#1C2B23]';

const FieldError: React.FC<{ message?: string }> = ({ message }) =>
  message ? (
    <span role="alert" className="text-[11px] text-red-700 block mt-1">
      {message}
    </span>
  ) : null;

export const SettingsPage: React.FC = () => {
  const settingsQuery = useSettings();
  const [draft, setDraft] = useState<Draft | null>(null);
  const [savedSuccess, setSavedSuccess] = useState(false);
  const [clientErrors, setClientErrors] = useState<Record<string, string>>({});
  const [noChanges, setNoChanges] = useState(false);
  const current = settingsQuery.data;
  // Hide the "Settings Saved" notice after a moment; the timer is cleared if the page unmounts first.
  useEffect(() => {
    if (!savedSuccess) return;
    const id = setTimeout(() => setSavedSuccess(false), 2500);
    return () => clearTimeout(id);
  }, [savedSuccess]);
  const form = draft ?? (current ? toDraft(current) : null);

  const edit = (patch: Partial<Draft>) => {
    if (!form) return;
    setDraft({ ...form, ...patch });
    setNoChanges(false);
  };

  const handleSave = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form || !current) return;
    const { patch, errors } = buildPatch(form, current);
    setClientErrors(errors);
    if (Object.keys(errors).length > 0) return;
    if (Object.keys(patch).length === 0) {
      setNoChanges(true);
      return;
    }
    const saved = await settingsQuery.save(patch);
    if (saved) {
      setDraft(null);
      setSavedSuccess(true);
    }
  };

  if (settingsQuery.error && !current) {
    return <ErrorState message={settingsQuery.error.message} onRetry={() => void settingsQuery.refetch()} />;
  }

  if (!form || !current) return <div className="p-6 text-center text-xs text-stone-500">Loading settings...</div>;

  const serverErrors = fieldErrorsFrom(settingsQuery.saveError);
  const errorFor = (k: string) => clientErrors[k] ?? serverErrors[k];
  const cadence = describeCadence(parseTimes(form.collection_times));

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Header */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-bold uppercase tracking-wider text-[#426B4F]">System Configuration</span>
          </div>
          <h1 className="text-lg font-bold text-[#1C2B23]">Calibration &amp; Telemetry Settings</h1>
          <p className="text-xs text-[#6B786E] mt-0.5">Stored by the backend. Changes are validated on save.</p>
        </div>

        {savedSuccess && (
          <div role="status" className="flex items-center gap-1.5 text-xs font-bold text-[#2E8B57] bg-[#EDF4EB] px-3 py-1.5 rounded-lg border border-[#DDE8D8] animate-fadeIn">
            <Check className="w-4 h-4" />
            <span>Settings Saved</span>
          </div>
        )}
      </div>

      {settingsQuery.saveError && (
        <ErrorState
          title={settingsQuery.saveError.status === 422 ? 'Some settings are invalid' : 'Could not save settings'}
          message={settingsQuery.saveError.message}
        />
      )}

      <form onSubmit={handleSave} className="space-y-6">
        {/* Polling */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Server className="w-4 h-4 text-[#2E8B57]" />
            <span>Latest Readings Refresh</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label htmlFor="polling" className="block font-semibold text-[#1C2B23] mb-1">
                Refresh interval on Latest Readings (seconds):
              </label>
              <input
                id="polling"
                type="number"
                min={1}
                max={3600}
                value={form.polling_interval_seconds}
                onChange={(e) => edit({ polling_interval_seconds: e.target.value })}
                className={`${inputCls} ${errorFor('polling_interval_seconds') ? 'border-red-400' : 'border-[#DCE4D9]'}`}
              />
              <FieldError message={errorFor('polling_interval_seconds')} />
              <span className="text-[10px] text-[#6B786E] block mt-1">1 to 3600 seconds.</span>
            </div>
          </div>
        </div>

        {/* Calibration */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Sliders className="w-4 h-4 text-[#2E8B57]" />
            <span>Sensor Hardware &amp; Soil ADC Calibration Points</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label htmlFor="adc-dry" className="block font-semibold text-[#1C2B23] mb-1">
                Dry Air Calibration Value (ADC Counts):
              </label>
              <input
                id="adc-dry"
                type="number"
                value={form.soil_adc_dry}
                onChange={(e) => edit({ soil_adc_dry: e.target.value })}
                className={`${inputCls} ${errorFor('soil_adc_dry') ? 'border-red-400' : 'border-[#DCE4D9]'}`}
              />
              <FieldError message={errorFor('soil_adc_dry')} />
              <span className="text-[10px] text-[#6B786E] block mt-1">Sensor exposed to ambient air corresponds to 0% moisture.</span>
            </div>

            <div>
              <label htmlFor="adc-wet" className="block font-semibold text-[#1C2B23] mb-1">
                Wet Water Immersion Value (ADC Counts):
              </label>
              <input
                id="adc-wet"
                type="number"
                value={form.soil_adc_wet}
                onChange={(e) => edit({ soil_adc_wet: e.target.value })}
                className={`${inputCls} ${errorFor('soil_adc_wet') ? 'border-red-400' : 'border-[#DCE4D9]'}`}
              />
              <FieldError message={errorFor('soil_adc_wet')} />
              <span className="text-[10px] text-[#6B786E] block mt-1">Submerged in water corresponds to 100% saturation.</span>
            </div>
          </div>

          <div className="pt-4 border-t border-[#DCE4D9] flex items-center justify-between text-xs">
            <div>
              <span className="font-semibold text-[#1C2B23] block">Substrate pH Channel (Optional Sensor Probe)</span>
              <span className="text-[#6B786E] text-[11px]">Enable only if a physical analog pH probe is connected.</span>
            </div>

            <button
              type="button"
              aria-pressed={form.ph_sensor_installed}
              onClick={() => edit({ ph_sensor_installed: !form.ph_sensor_installed })}
              className={`px-3 py-1.5 rounded-lg font-bold text-xs transition-colors border ${
                form.ph_sensor_installed ? 'bg-[#EDF4EB] text-[#2E8B57] border-[#DDE8D8]' : 'bg-stone-100 text-stone-600 border-stone-200'
              }`}
            >
              {form.ph_sensor_installed ? 'Installed & Active' : 'Not Installed'}
            </button>
          </div>
        </div>

        {/* Schedule and retention */}
        <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card space-y-4">
          <h3 className="text-sm font-bold text-[#1C2B23] flex items-center gap-2">
            <Clock className="w-4 h-4 text-[#2E8B57]" />
            <span>Sampling Schedule &amp; Storage Retention</span>
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div>
              <label htmlFor="times" className="block font-semibold text-[#1C2B23] mb-1">
                Daily collection times (HH:MM, IST, comma separated):
              </label>
              <input
                id="times"
                type="text"
                value={form.collection_times}
                onChange={(e) => edit({ collection_times: e.target.value })}
                className={`${inputCls} ${errorFor('collection_times') ? 'border-red-400' : 'border-[#DCE4D9]'}`}
              />
              <FieldError message={errorFor('collection_times')} />
              <span className="text-[10px] text-[#6B786E] block mt-1">{cadence ? `Currently ${cadence}.` : 'No collection times set.'}</span>
            </div>

            <div>
              <label htmlFor="retention" className="block font-semibold text-[#1C2B23] mb-1">
                Local telemetry retention (days):
              </label>
              <input
                id="retention"
                type="number"
                min={1}
                value={form.log_retention_days}
                onChange={(e) => edit({ log_retention_days: e.target.value })}
                className={`${inputCls} ${errorFor('log_retention_days') ? 'border-red-400' : 'border-[#DCE4D9]'}`}
              />
              <FieldError message={errorFor('log_retention_days')} />
              <span className="text-[10px] text-[#6B786E] block mt-1">
                Storage: <span className="font-mono">{current.storage_path}</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3">
          {noChanges && <span className="text-xs text-stone-500">No changes to save.</span>}
          <button
            type="submit"
            disabled={settingsQuery.saving}
            className="px-6 py-2.5 bg-[#2E8B57] hover:bg-[#236e44] disabled:opacity-60 text-white text-xs font-bold rounded-lg transition-colors shadow-sm"
          >
            {settingsQuery.saving ? 'Saving...' : 'Save Configuration Changes'}
          </button>
        </div>
      </form>
    </div>
  );
};
