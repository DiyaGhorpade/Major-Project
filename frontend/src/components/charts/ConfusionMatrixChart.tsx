import React from 'react';
import { ModelMetrics } from '../../types';

interface ConfusionMatrixChartProps {
  metrics: ModelMetrics;
}

export const ConfusionMatrixChart: React.FC<ConfusionMatrixChartProps> = ({ metrics }) => {
  const { labels, matrix } = metrics.confusion_matrix;

  // Find max value in diagonal vs off-diagonal for shading
  let maxVal = 0;
  matrix.forEach((row) => {
    row.forEach((val) => {
      if (val > maxVal) maxVal = val;
    });
  });

  return (
    <div className="bg-white rounded-card border border-[#DCE4D9] p-5 shadow-card">
      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div>
          <h3 className="text-sm font-bold text-[#1C2B23]">
            Hold-out Confusion Matrix ({labels.length} classes)
          </h3>
          <p className="text-xs text-[#6B786E]">
            Rows are the reference label; columns are the model's prediction.
          </p>
        </div>
        <div className="text-xs font-mono text-stone-500">
          N = {metrics.total_samples.toLocaleString()} samples
        </div>
      </div>

      <div className="overflow-x-auto my-2">
        <table className="w-full text-xs text-center border-collapse">
          <thead>
            <tr>
              <th className="p-2 text-left text-[#6B786E] font-medium">True \ Pred</th>
              {labels.map((lbl, i) => (
                <th key={i} className="p-2 font-semibold text-[#1C2B23] border-b border-[#DCE4D9]">
                  {lbl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, rIdx) => {
              const rowTotal = row.reduce((a, b) => a + b, 0);
              return (
                <tr key={rIdx}>
                  <td className="p-2 text-left font-semibold text-[#1C2B23] border-r border-[#DCE4D9] whitespace-nowrap">
                    {labels[rIdx]}
                  </td>
                  {row.map((val, cIdx) => {
                    const isDiagonal = rIdx === cIdx;
                    // Compute opacity
                    const opacity = maxVal > 0 ? Math.max(0.08, Math.min(1, val / maxVal)) : 0.08;
                    const cellBg = isDiagonal
                      ? `rgba(46, 139, 87, ${opacity})`
                      : val > 0
                      ? `rgba(201, 76, 76, ${Math.max(0.12, opacity * 2)})`
                      : 'transparent';

                    const textColor =
                      isDiagonal && opacity > 0.4
                        ? 'text-white'
                        : val > 0 && !isDiagonal
                        ? 'text-[#C94C4C] font-bold'
                        : 'text-[#1C2B23]';

                    return (
                      <td
                        key={cIdx}
                        className="p-3 border border-[#DCE4D9]/60 font-mono text-xs transition-colors relative group"
                        style={{ backgroundColor: cellBg }}
                      >
                        <span className={textColor} data-testid={`cm-${rIdx}-${cIdx}`}>{val}</span>
                        {/* Hover tooltip */}
                        <div className="absolute hidden group-hover:block z-20 bottom-full left-1/2 -translate-x-1/2 mb-1 px-2.5 py-1 bg-stone-900 text-white rounded text-[10px] whitespace-nowrap shadow-lg">
                          True: {labels[rIdx]} | Pred: {labels[cIdx]} ({rowTotal > 0 ? ((val / rowTotal) * 100).toFixed(1) : '0.0'}% of true {labels[rIdx]})
                        </div>
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex flex-wrap items-center justify-between text-[11px] text-[#6B786E] pt-3 border-t border-[#DCE4D9] mt-3">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-[#2E8B57]" /> Correct Predictions (Diagonal)
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded bg-red-400" /> Misclassifications (Off-diagonal)
          </span>
        </div>
        <span>Evaluation split: {metrics.split}</span>
      </div>
    </div>
  );
};
