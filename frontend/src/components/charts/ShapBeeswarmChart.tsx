import React, { useMemo, useState } from 'react';
import { AlertCircle } from 'lucide-react';
import { GlobalShapData } from '../../types';

interface ShapBeeswarmChartProps {
  data: GlobalShapData;
}

const ALL = 'all';
const TOP_N = 15;

interface BarRow {
  name: string;
  displayName: string;
  value: number;
  interpretation?: string;
  rank: number;
}

function pointColor(norm: number | null): string {
  if (norm === null) return '#a8a29e';
  if (norm < 0.35) return '#3b82f6';
  if (norm > 0.65) return '#ef4444';
  return '#8b5cf6';
}

/** Deterministic vertical offset (0..1) so overlapping points on one row stay readable. */
function jitter(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i += 1) h = (h * 31 + id.charCodeAt(i)) % 997;
  return h / 997;
}

export const ShapBeeswarmChart: React.FC<ShapBeeswarmChartProps> = ({ data }) => {
  const classes = Object.keys(data.perClassTop15 ?? {});
  const [selected, setSelected] = useState<string>(ALL);
  const active = selected !== ALL && classes.includes(selected) ? selected : ALL;

  const displayNames = useMemo(() => {
    const m = new Map<string, string>();
    for (const f of data.features) m.set(f.name, f.displayName);
    for (const list of Object.values(data.perClassTop15 ?? {})) for (const f of list) m.set(f.name, f.displayName);
    return m;
  }, [data]);

  const bars: BarRow[] = useMemo(() => {
    if (active === ALL) {
      return [...data.features]
        .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
        .slice(0, TOP_N)
        .map((f, i) => ({ name: f.name, displayName: f.displayName, value: f.mean_abs_shap, interpretation: f.interpretation, rank: f.importance_rank ?? i + 1 }));
    }
    return [...(data.perClassTop15[active] ?? [])]
      .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
      .slice(0, TOP_N)
      .map((f, i) => ({ name: f.name, displayName: f.displayName, value: f.mean_abs_shap, rank: i + 1 }));
  }, [data, active]);
  const barMax = Math.max(0, ...bars.map((b) => b.value));

  // Beeswarm: only the sampled points; optionally only rows predicted as the selected class.
  const points = useMemo(
    () => data.beeswarm_points.filter((p) => active === ALL || p.predicted_class === active),
    [data, active],
  );
  const axisMax = Math.max(0.1, ...points.map((p) => Math.abs(p.shap_value)));
  const axisLabel = axisMax.toFixed(2);
  const swarmRows = useMemo(() => {
    const byFeature = new Map<string, typeof points>();
    for (const p of points) byFeature.set(p.feature, [...(byFeature.get(p.feature) ?? []), p]);
    const meanAbs = new Map(data.features.map((f) => [f.name, f.mean_abs_shap]));
    return [...byFeature.entries()]
      .map(([name, pts]) => ({
        name,
        displayName: displayNames.get(name) ?? name,
        pts,
        // Sort by the global importance when known, else by the sample's own mean |SHAP|.
        order: meanAbs.get(name) ?? pts.reduce((n, p) => n + Math.abs(p.shap_value), 0) / pts.length,
      }))
      .sort((a, b) => b.order - a.order);
  }, [points, data.features, displayNames]);

  return (
    <div className="space-y-6">
      {/* Class selector */}
      {classes.length > 0 && (
        <div className="bg-white rounded-card border border-[#DCE4D9] p-4 shadow-card flex flex-wrap items-center gap-3 text-xs">
          <label htmlFor="shap-class-select" className="font-bold text-[#173F32]">
            Class view:
          </label>
          <select
            id="shap-class-select"
            value={active}
            onChange={(e) => setSelected(e.target.value)}
            className="text-xs font-bold bg-[#EDF4EB] border border-[#DDE8D8] rounded-lg px-3 py-1.5 text-[#102F26] outline-none cursor-pointer"
          >
            <option value={ALL}>All classes (mean over classes)</option>
            {classes.map((c) => (
              <option key={c} value={c}>
                Class: {c}
              </option>
            ))}
          </select>
          <span className="text-[#6B786E]">The class view changes the top-{TOP_N} list and filters the sampled beeswarm points by predicted class.</span>
        </div>
      )}

      {/* Global Feature Importance */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Top {bars.length} features by mean |SHAP|{active === ALL ? '' : ` for class '${active}'`}
            </h3>
            <p className="text-xs text-[#6B786E]">
              Explained {data.sample_count.toLocaleString()} test rows on model {data.model_version ?? 'n/a'}. Units: {data.units}.
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
            Sample N = {data.sample_count.toLocaleString()}
          </span>
        </div>

        <ul className="space-y-2 text-xs" data-testid="shap-global-features">
          {bars.map((b) => (
            <li key={b.name} className="flex items-center gap-3" title={b.interpretation}>
              <span className="w-6 text-right font-mono text-[#6B786E]">#{b.rank}</span>
              <span className="w-64 shrink-0 font-semibold text-[#1C2B23] truncate">{b.displayName}</span>
              <div className="flex-1 bg-[#EDF4EB] h-2.5 rounded-full overflow-hidden">
                <div className="bg-[#2E8B57] h-full" style={{ width: `${barMax > 0 ? (b.value / barMax) * 100 : 0}%` }} />
              </div>
              <span className="w-16 text-right font-mono font-semibold text-[#173F32]">{b.value.toFixed(3)}</span>
            </li>
          ))}
          {bars.length === 0 && <li className="text-[#6B786E]">No features reported for this view.</li>}
        </ul>
      </div>

      {/* Beeswarm (sampled) */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-2">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">SHAP summary (beeswarm) of a sample</h3>
            <p className="text-xs text-[#6B786E]">
              Each point is one feature of one sampled row. Position is its impact on the predicted class; colour is the feature value, scaled from low to high.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#6B786E]">Feature value:</span>
            <span className="text-blue-600 font-medium">Low</span>
            <div className="w-20 h-2.5 rounded bg-gradient-to-r from-blue-500 via-purple-500 to-red-500" />
            <span className="text-red-600 font-medium">High</span>
          </div>
        </div>
        <p
          className="mb-4 inline-block text-[11px] font-semibold px-2.5 py-1 rounded bg-amber-50 text-amber-900 border border-amber-200"
          data-testid="beeswarm-approximate"
        >
          Approximate: {data.approximate.beeswarm}
        </p>

        <div className="space-y-4 my-4" data-testid="shap-beeswarm">
          {swarmRows.map((row) => (
            <div key={row.name} className="border-b border-[#DCE4D9]/60 pb-3">
              <div className="flex items-center justify-between text-xs mb-1">
                <span className="font-semibold text-[#1C2B23]">{row.displayName}</span>
                <span className="text-[#6B786E] text-[11px] font-mono">{row.pts.length} sampled points</span>
              </div>
              <div className="relative h-10 bg-stone-50 rounded-lg border border-stone-200 overflow-hidden">
                <div className="absolute left-1/2 top-0 bottom-0 w-px bg-stone-400 z-10" />
                {row.pts.map((pt) => {
                  const xPercent = Math.max(2, Math.min(98, 50 + (pt.shap_value / axisMax) * 48));
                  const yPercent = 12 + jitter(pt.id) * 62;
                  return (
                    <div
                      key={pt.id}
                      className="absolute w-3 h-3 rounded-full border border-white/80 shadow-sm opacity-85 hover:scale-150 hover:z-20 transition-transform cursor-pointer -translate-x-1/2"
                      style={{ left: `${xPercent}%`, top: `${yPercent}%`, backgroundColor: pointColor(pt.feature_value_norm) }}
                      title={`${row.displayName} SHAP: ${pt.shap_value > 0 ? '+' : ''}${pt.shap_value.toFixed(2)} log-odds (value norm: ${
                        pt.feature_value_norm === null ? 'n/a' : pt.feature_value_norm.toFixed(2)
                      }${pt.predicted_class ? `, predicted: ${pt.predicted_class}` : ''})`}
                    />
                  );
                })}
              </div>
            </div>
          ))}
          {swarmRows.length === 0 && <p className="text-xs text-[#6B786E]">No sampled points for this class.</p>}
        </div>
        {swarmRows.length > 0 && (
          <div className="flex justify-between text-[10px] text-stone-400">
            <span>-{axisLabel} log-odds (away from predicted class)</span>
            <span className="font-semibold text-stone-500">0 (no impact)</span>
            <span>+{axisLabel} log-odds (towards predicted class)</span>
          </div>
        )}

        {/* Global Summary */}
        <div className="mt-6 border-t border-[#DCE4D9] pt-4">
          <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-2">Global feature interpretation</h4>
          <div className="divide-y divide-[#DCE4D9]/60 text-xs">
            {[...data.features]
              .sort((a, b) => b.mean_abs_shap - a.mean_abs_shap)
              .slice(0, TOP_N)
              .map((f) => (
                <div key={f.name} className="py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <span className="min-w-5 h-5 px-1 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-bold flex items-center justify-center text-[10px]">
                      #{f.importance_rank}
                    </span>
                    <span className="font-semibold text-[#1C2B23]">{f.displayName}</span>
                  </div>
                  <span className="text-[#6B786E] max-w-md">{f.laymanTip}</span>
                </div>
              ))}
          </div>
        </div>

        <div className="mt-6 p-4 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="font-semibold">Explanatory note: </strong>
            {data.note ??
              'SHAP explains how features influence the model’s prediction. It does not establish causality or independently identify the biological root cause.'}
          </p>
        </div>
      </div>
    </div>
  );
};
