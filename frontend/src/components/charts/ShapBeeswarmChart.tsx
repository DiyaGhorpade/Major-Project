import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from 'recharts';
import { GlobalShapData } from '../../types';
import { AlertCircle, Info } from 'lucide-react';

interface ShapBeeswarmChartProps {
  data: GlobalShapData;
}

export const ShapBeeswarmChart: React.FC<ShapBeeswarmChartProps> = ({ data }) => {
  // Sort features by mean absolute SHAP descending
  const sortedFeatures = [...data.features].sort((a, b) => b.mean_abs_shap - a.mean_abs_shap);

  // Data-driven SHAP axis (log-odds units), not a hard-coded range.
  const axisMax = Math.max(0.1, ...data.beeswarm_points.map((p) => Math.abs(p.shap_value)));
  const axisLabel = axisMax.toFixed(2);

  const barData = sortedFeatures.map((f) => ({
    name: f.displayName,
    importance: f.mean_abs_shap,
    rank: f.importance_rank,
    interpretation: f.interpretation,
  }));

  return (
    <div className="space-y-6">
      {/* Global Feature Importance Bar Chart */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              Global Feature Importance — Mean |SHAP Value|
            </h3>
            <p className="text-xs text-[#6B786E]">
              Evaluated across {data.sample_count.toLocaleString()} validation samples on model {data.model_version ?? 'n/a'}.
            </p>
          </div>
          <span className="text-xs font-semibold px-2.5 py-1 rounded bg-[#EDF4EB] text-[#2E8B57] border border-[#DDE8D8]">
            Sample N = {data.sample_count}
          </span>
        </div>

        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              layout="vertical"
              data={barData}
              margin={{ top: 10, right: 30, left: 40, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#EDF4EB" />
              <XAxis
                type="number"
                stroke="#6B786E"
                fontSize={11}
                tickLine={false}
                axisLine={{ stroke: '#DCE4D9' }}
                unit=" SHAP"
              />
              <YAxis
                type="category"
                dataKey="name"
                stroke="#1C2B23"
                fontSize={12}
                tickLine={false}
                axisLine={{ stroke: '#DCE4D9' }}
                width={140}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const item = payload[0].payload;
                    return (
                      <div className="bg-[#102F26] text-white p-3 rounded-lg shadow-xl text-xs space-y-1">
                        <div className="font-semibold text-[#DDE8D8]">{item.name}</div>
                        <div className="font-mono text-emerald-400 font-bold">
                          Mean |SHAP|: {item.importance.toFixed(3)}
                        </div>
                        <div className="text-[11px] text-stone-300 max-w-xs">{item.interpretation}</div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {barData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      index === 0
                        ? '#2E8B57'
                        : index === 1
                        ? '#426B4F'
                        : index === 2
                        ? '#C98227'
                        : '#3977B8'
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Beeswarm Distribution Plot Simulation */}
      <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#1C2B23]">
              SHAP Summary / Beeswarm Distribution Plot
            </h3>
            <p className="text-xs text-[#6B786E]">
              Each point represents an individual sample. Position shows impact on model prediction (SHAP), color indicates measured feature value.
            </p>
          </div>
          {/* Color legend */}
          <div className="flex items-center gap-2 text-xs">
            <span className="text-[#6B786E]">Feature Value:</span>
            <div className="flex items-center gap-1">
              <span className="text-blue-600 font-medium">Low</span>
              <div className="w-20 h-2.5 rounded bg-gradient-to-r from-blue-500 via-purple-500 to-red-500" />
              <span className="text-red-600 font-medium">High</span>
            </div>
          </div>
        </div>

        {/* Beeswarm rows for each feature */}
        <div className="space-y-4 my-4">
          {sortedFeatures.map((feat) => {
            const points = data.beeswarm_points.filter((p) => p.feature === feat.name);
            return (
              <div key={feat.name} className="border-b border-[#DCE4D9]/60 pb-3">
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className="font-semibold text-[#1C2B23]">{feat.displayName}</span>
                  <span className="text-[#6B786E] text-[11px] font-mono">
                    Mean impact: {feat.mean_abs_shap.toFixed(3)}
                  </span>
                </div>

                {/* Strip chart container representing the -max to +max SHAP axis */}
                <div className="relative h-10 bg-stone-50 rounded-lg border border-stone-200 overflow-hidden flex items-center">
                  {/* Zero centerline */}
                  <div className="absolute left-1/2 top-0 bottom-0 w-px bg-stone-400 z-10" />

                  {/* Scatter points */}
                  {points.map((pt) => {
                    // Map SHAP (-max to +max) to percentage (0% to 100%)
                    const xPercent = Math.max(5, Math.min(95, 50 + (pt.shap_value / axisMax) * 45));
                    
                    // Color based on feature_value_norm (0=blue, 0.5=purple, 1=red)
                    const color =
                      pt.feature_value_norm === null
                        ? '#a8a29e'
                        : pt.feature_value_norm < 0.35
                        ? '#3b82f6'
                        : pt.feature_value_norm > 0.65
                        ? '#ef4444'
                        : '#8b5cf6';

                    return (
                      <div
                        key={pt.id}
                        className="absolute w-3.5 h-3.5 rounded-full border border-white/80 shadow-sm opacity-85 hover:scale-150 hover:z-20 transition-transform cursor-pointer"
                        style={{
                          left: `${xPercent}%`,
                          backgroundColor: color,
                        }}
                        title={`${feat.displayName} SHAP: ${pt.shap_value > 0 ? '+' : ''}${pt.shap_value.toFixed(2)} (Value Norm: ${pt.feature_value_norm === null ? 'n/a' : pt.feature_value_norm.toFixed(2)})`}
                      />
                    );
                  })}
                </div>

                <div className="flex justify-between text-[10px] text-stone-400 mt-1">
                  <span>-{axisLabel} (Decreases target class)</span>
                  <span className="font-semibold text-stone-500">0.0 (No impact)</span>
                  <span>+{axisLabel} (Increases target class)</span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Global Summary Table */}
        <div className="mt-6 border-t border-[#DCE4D9] pt-4">
          <h4 className="text-xs font-bold text-[#1C2B23] uppercase tracking-wider mb-2">
            Global Feature Interpretation Summary
          </h4>
          <div className="divide-y divide-[#DCE4D9]/60 text-xs">
            {sortedFeatures.map((f, i) => (
              <div key={i} className="py-2.5 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-full bg-[#EDF4EB] text-[#2E8B57] font-bold flex items-center justify-center text-[10px]">
                    #{f.importance_rank}
                  </span>
                  <span className="font-semibold text-[#1C2B23]">{f.displayName}</span>
                </div>
                <span className="text-[#6B786E] max-w-md">{f.interpretation}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Required Mandatory Disclaimer */}
        <div className="mt-6 p-4 rounded-xl bg-[#EDF4EB] border border-[#DDE8D8] text-xs text-[#173F32] flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-[#2E8B57] shrink-0 mt-0.5" />
          <p className="leading-relaxed">
            <strong className="font-semibold">Explanatory Note: </strong>
            SHAP explains how features influence the model’s prediction. It does not establish causality or independently identify the biological root cause.
          </p>
        </div>
      </div>
    </div>
  );
};
