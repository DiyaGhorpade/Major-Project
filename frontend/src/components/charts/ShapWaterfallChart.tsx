import React from 'react';
import { ArrowUp, ArrowDown, AlertCircle, Scale } from 'lucide-react';
import { LocalShapExplanation } from '../../types';

interface ShapWaterfallChartProps {
  explanation: LocalShapExplanation;
}

export const BASELINE_NOT_PROVIDED = 'Contributions relative to model baseline (not provided by model service)';

const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;

export const ShapWaterfallChart: React.FC<ShapWaterfallChartProps> = ({ explanation }) => {
  const hasBase = explanation.base_value !== null;
  // Without a base value the waterfall starts at 0 and every position is relative to the model baseline.
  const start = explanation.base_value ?? 0;
  const probability = explanation.predicted_probability;
  const probabilityPct = probability === null ? 'n/a' : `${(probability * 100).toFixed(0)}%`;
  const target = explanation.target_class;

  const rows = explanation.features.reduce<Array<{ f: (typeof explanation.features)[number]; from: number; to: number }>>((acc, f) => {
    const from = acc.length > 0 ? acc[acc.length - 1].to : start;
    acc.push({ f, from, to: from + f.shap_value });
    return acc;
  }, []);
  const end = rows.length > 0 ? rows[rows.length - 1].to : start;

  // Data-driven axis covering the start, every running total and zero (log-odds units).
  const points = [0, start, ...rows.map((r) => r.to)];
  const lo = Math.min(...points);
  const hi = Math.max(...points);
  const span = hi - lo || 1;
  const pos = (v: number) => ((v - lo) / span) * 100;

  return (
    <div className="space-y-6">
      {/* Layman friendly summary */}
      <div className="bg-gradient-to-r from-[#EDF4EB] to-[#DDE8D8]/60 rounded-card border border-[#DDE8D8] p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-[#2E8B57] text-white shrink-0 mt-0.5 shadow-sm">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#2E8B57]">Plain-language summary of this model rating</span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white text-[#173F32] font-semibold border border-[#DDE8D8]">
                Pot #{explanation.plant_id}
              </span>
            </div>
            <h3 className="text-base font-bold text-[#102F26] mb-1">
              Model rating: {target} (model probability {probabilityPct})
            </h3>
            <p className="text-xs text-[#1C2B23] leading-relaxed">{explanation.laymanSummary}</p>
          </div>
        </div>
      </div>

      {/* Target class and probability */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-[#6B786E]">Explained class (target_class)</span>
            <div className="text-xl font-bold text-[#1C2B23] flex flex-wrap items-center gap-2">
              <span>{target}</span>
              <span className="text-xs font-normal px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
                Model {explanation.model_version ?? 'n/a'}
              </span>
              <span className="text-xs font-normal px-2.5 py-0.5 rounded-full bg-stone-100 text-stone-600 border border-stone-200 font-mono">
                {explanation.record_id}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-6 text-right">
            <div>
              <span className="text-xs text-[#6B786E] block">Baseline (log-odds)</span>
              <span className="font-mono text-sm font-semibold text-stone-600">{hasBase ? start.toFixed(2) : 'n/a'}</span>
            </div>
            <div>
              <span className="text-xs text-[#6B786E] block">Predicted probability</span>
              <span className="font-mono text-2xl font-bold text-[#1C2B23]" data-testid="shap-probability">
                {probabilityPct}
              </span>
            </div>
          </div>
        </div>

        <div className="space-y-1.5">
          <div className="w-full bg-[#EDF4EB] h-3 rounded-full overflow-hidden flex">
            <div
              className="bg-[#2E8B57] h-full transition-all duration-500 rounded-l-full"
              style={{ width: `${Math.min(100, Math.max(0, (probability ?? 0) * 100))}%` }}
            />
          </div>
          <div className="flex justify-between text-[11px] text-[#6B786E]">
            <span>0%</span>
            <span className="font-semibold text-[#102F26]">
              {explanation.base_value !== null && explanation.model_output !== null
                ? `Net shift: ${signed(explanation.model_output - explanation.base_value)} (log-odds)`
                : BASELINE_NOT_PROVIDED}
            </span>
            <span>100%</span>
          </div>
        </div>
      </div>

      {/* Waterfall */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">Sensor influence on the model score (waterfall)</h3>
            <p className="text-xs text-[#6B786E]">
              Top {explanation.features.length} contributions to the score for '{target}'. Units: log-odds (not probability points).
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-semibold">
            <span className="flex items-center gap-1.5 text-[#C98227]">
              <span className="w-3 h-3 rounded bg-[#C98227]" />
              <span>Pushes towards '{target}' (+)</span>
            </span>
            <span className="flex items-center gap-1.5 text-[#3977B8]">
              <span className="w-3 h-3 rounded bg-[#3977B8]" />
              <span>Pushes away from '{target}' (-)</span>
            </span>
          </div>
        </div>

        <p className="text-[11px] text-[#6B786E] mb-3" data-testid="shap-baseline-note">
          {hasBase ? `Starts at the model baseline ${start.toFixed(2)}.` : BASELINE_NOT_PROVIDED + '. The waterfall starts at 0.'}
        </p>

        <div className="space-y-3 my-4" data-testid="shap-waterfall-rows">
          {rows.map(({ f, from, to }) => {
            const up = f.shap_value >= 0;
            const left = pos(Math.min(from, to));
            const width = Math.max(0.8, (Math.abs(f.shap_value) / span) * 100);
            return (
              <div
                key={f.name}
                data-testid="shap-feature-row"
                className="p-3.5 rounded-xl border border-[#DCE4D9] bg-stone-50/60 hover:bg-[#EDF4EB]/30 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-[#1C2B23] text-sm">{f.displayName}</span>
                    <span className="px-2.5 py-0.5 rounded bg-white text-stone-800 font-mono text-[11px] border border-stone-200 shadow-sm font-bold">
                      Reading: {f.observed_value === null ? 'n/a' : `${Number(f.observed_value.toFixed(2)).toLocaleString()}${f.unit ? ` ${f.unit}` : ''}`}
                    </span>
                  </div>
                  <div className="font-mono font-bold text-xs">
                    {up ? (
                      <span className="text-[#C98227] flex items-center">
                        <ArrowUp className="w-3.5 h-3.5 mr-0.5" /> {signed(f.shap_value)} log-odds
                      </span>
                    ) : (
                      <span className="text-[#3977B8] flex items-center">
                        <ArrowDown className="w-3.5 h-3.5 mr-0.5" /> {signed(f.shap_value)} log-odds
                      </span>
                    )}
                  </div>
                </div>

                {/* Waterfall step on a shared, data-driven axis */}
                <div className="relative w-full bg-stone-200 h-3 rounded-full overflow-hidden my-2" role="presentation">
                  <div className="absolute top-0 bottom-0 w-px bg-stone-500 z-10" style={{ left: `${pos(0)}%` }} />
                  <div
                    className={`absolute top-0 bottom-0 ${up ? 'bg-[#C98227]' : 'bg-[#3977B8]'}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] font-mono text-stone-400">
                  <span>from {signed(from)}</span>
                  <span>to {signed(to)}</span>
                </div>

                <div className="text-[11px] text-[#1C2B23] flex items-start gap-1.5 mt-1">
                  <span className="font-bold text-[#2E8B57]">Plain English:</span>
                  <span>{f.laymanExplanation}</span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap justify-between gap-2 text-[11px] text-[#6B786E] font-mono">
          <span>Axis: {signed(lo)} to {signed(hi)} log-odds</span>
          <span>Sum of the {rows.length} shown contributions: {signed(end - start)}</span>
        </div>

        {explanation.explanation_note && (
          <div className="mt-6 p-4 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-start gap-3">
            <AlertCircle className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
            <p className="leading-relaxed">
              <strong className="font-semibold">Interpretability note: </strong>
              {explanation.explanation_note}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};
