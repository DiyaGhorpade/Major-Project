import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  CartesianGrid,
} from 'recharts';
import { LocalShapExplanation } from '../../types';
import { ArrowUp, ArrowDown, HelpCircle, AlertCircle, Sparkles, Scale } from 'lucide-react';

interface ShapWaterfallChartProps {
  explanation: LocalShapExplanation;
}

export const ShapWaterfallChart: React.FC<ShapWaterfallChartProps> = ({ explanation }) => {
  // The ML service gives no base value, so contributions start at 0 (relative to the model baseline).
  let cumulative = explanation.base_value ?? 0;
  const probability = explanation.predicted_probability;
  const probabilityPct = probability === null ? 'n/a' : `${(probability * 100).toFixed(0)}%`;

  const waterfallData = explanation.features.map((f) => {
    const startVal = cumulative;
    cumulative += f.shap_value;
    const endVal = cumulative;

    return {
      feature: f.displayName,
      observed: f.observed_value === null ? 'n/a' : `${f.observed_value} ${f.unit}`,
      shap: f.shap_value,
      isPositive: f.shap_value >= 0,
      startVal,
      endVal,
      description: f.description,
      laymanExplanation: f.laymanExplanation,
    };
  });

  return (
    <div className="space-y-6">
      {/* Layman Friendly "How the AI Made this Decision" Callout */}
      <div className="bg-gradient-to-r from-[#EDF4EB] to-[#DDE8D8]/60 rounded-card border border-[#DDE8D8] p-5 shadow-sm">
        <div className="flex items-start gap-3">
          <div className="p-2.5 rounded-xl bg-[#2E8B57] text-white shrink-0 mt-0.5 shadow-sm">
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider text-[#2E8B57]">
                Layman's Guide to this AI Diagnosis
              </span>
              <span className="text-xs px-2 py-0.5 rounded-full bg-white text-[#173F32] font-semibold border border-[#DDE8D8]">
                Pot #{explanation.plant_id}
              </span>
            </div>
            <h3 className="text-base font-bold text-[#102F26] mb-1">
              AI Decision: {explanation.target_class} ({probabilityPct} Certainty)
            </h3>
            <p className="text-xs text-[#1C2B23] leading-relaxed">
              {explanation.laymanSummary}
            </p>
          </div>
        </div>
      </div>

      {/* Probability Summary Bar */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <span className="text-xs font-semibold uppercase tracking-wider text-[#6B786E]">
              Evaluated Stress Category
            </span>
            <div className="text-xl font-bold text-[#1C2B23] flex items-center gap-2">
              <span>{explanation.target_class}</span>
              <span className="text-xs font-normal px-2.5 py-0.5 rounded-full bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
                Model {explanation.model_version ?? 'n/a'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-6 text-right">
            <div>
              <span className="text-xs text-[#6B786E] block">Baseline Start E[f(x)]</span>
              <span className="font-mono text-sm font-semibold text-stone-600">
                {explanation.base_value === null ? 'n/a' : `${(explanation.base_value * 100).toFixed(0)}%`}
              </span>
            </div>
            <div>
              <span className="text-xs text-[#6B786E] block">Final AI Output f(x)</span>
              <span className="font-mono text-2xl font-bold text-[#1C2B23]">
                {probabilityPct}
              </span>
            </div>
          </div>
        </div>

        {/* Visual Probability meter */}
        <div className="space-y-1.5">
          <div className="w-full bg-[#EDF4EB] h-3 rounded-full overflow-hidden flex">
            <div
              className="bg-[#2E8B57] h-full transition-all duration-500 rounded-l-full"
              style={{ width: `${Math.min(100, (probability ?? 0) * 100)}%` }}
            />
          </div>
          <div className="flex justify-between text-[11px] text-[#6B786E]">
            <span>0% (No Likelihood)</span>
            <span className="font-semibold text-[#102F26]">
              {explanation.base_value !== null && explanation.model_output !== null
                ? `Net AI Shift: +${((explanation.model_output - explanation.base_value) * 100).toFixed(0)}%`
                : 'Contributions are relative to the model baseline (not provided)'}
            </span>
            <span>100% (Absolute Certainty)</span>
          </div>
        </div>
      </div>

      {/* Waterfall Feature Contributions with Simple Explanations */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Step-by-Step Sensor Influence (Waterfall Decomposition)
            </h3>
            <p className="text-xs text-[#6B786E]">
              See exactly how each sensor pushed the AI's final score up or down.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-semibold">
            <span className="flex items-center gap-1.5 text-[#C94C4C]">
              <span className="w-3 h-3 rounded bg-[#C94C4C]" />
              <span>Pushes toward stress (+)</span>
            </span>
            <span className="flex items-center gap-1.5 text-[#2E8B57]">
              <span className="w-3 h-3 rounded bg-[#2E8B57]" />
              <span>Rules out or lowers stress (-)</span>
            </span>
          </div>
        </div>

        {/* Breakdown rows */}
        <div className="space-y-3 my-4">
          {waterfallData.map((item, idx) => {
            const shiftPercent = Math.abs(item.shap * 100).toFixed(0);
            return (
              <div
                key={idx}
                className="p-3.5 rounded-xl border border-[#DCE4D9] bg-stone-50/60 hover:bg-[#EDF4EB]/30 transition-colors"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-[#1C2B23] text-sm">{item.feature}</span>
                    <span className="px-2.5 py-0.5 rounded bg-white text-stone-800 font-mono text-[11px] border border-stone-200 shadow-sm font-bold">
                      Reading: {item.observed}
                    </span>
                  </div>
                  <div className="font-mono font-bold text-xs">
                    {item.isPositive ? (
                      <span className="text-[#C94C4C] flex items-center">
                        <ArrowUp className="w-3.5 h-3.5 mr-0.5" /> +{item.shap.toFixed(2)} (+{shiftPercent}% impact)
                      </span>
                    ) : (
                      <span className="text-[#2E8B57] flex items-center">
                        <ArrowDown className="w-3.5 h-3.5 mr-0.5" /> {item.shap.toFixed(2)} (-{shiftPercent}% impact)
                      </span>
                    )}
                  </div>
                </div>

                {/* Progress bar visual for this step */}
                <div className="w-full bg-stone-200 h-2 rounded-full overflow-hidden flex my-2">
                  <div
                    className={item.isPositive ? 'bg-[#C94C4C]' : 'bg-[#2E8B57]'}
                    style={{
                      width: `${Math.min(100, Math.max(10, Math.abs(item.shap) * 120))}%`,
                    }}
                  />
                </div>

                {/* Simple Layman Explanation Callout */}
                <div className="text-[11px] text-[#1C2B23] flex items-start gap-1.5 mt-1">
                  <span className="font-bold text-[#2E8B57]">Plain English:</span>
                  <span>{item.laymanExplanation}</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Required Mandatory Disclaimer */}
        <div className="mt-6 p-4 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="font-semibold">Interpretability Standard: </strong>
            {explanation.explanation_note}
          </p>
        </div>
      </div>
    </div>
  );
};
