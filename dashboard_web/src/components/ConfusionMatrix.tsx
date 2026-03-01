import { useMemo } from "react";
import type { ConfusionEntry } from "../types";
import { useChartColors, useTheme } from "../ThemeContext";

interface Props {
  confusionTimeline: ConfusionEntry[];
  classNames?: string[];
}

export function ConfusionMatrix({ confusionTimeline, classNames }: Props) {
  const cc = useChartColors();
  const { theme } = useTheme();
  const isDark = theme === "dark";

  // Find the latest entry that has a non-empty matrix
  const latest = useMemo(() => {
    for (let i = confusionTimeline.length - 1; i >= 0; i--) {
      const entry = confusionTimeline[i];
      if (entry.matrix && entry.matrix.length > 0 && entry.matrix[0]?.length > 0) {
        return entry;
      }
    }
    return null;
  }, [confusionTimeline]);

  const { matrix, labels, maxVal } = useMemo(() => {
    if (!latest) return { matrix: [] as number[][], labels: [] as number[], maxVal: 1 };
    const mx = latest.matrix;
    const flat = mx.flat();
    return {
      matrix: mx,
      labels: latest.labels,
      maxVal: Math.max(1, ...flat),
    };
  }, [latest]);

  if (!latest) {
    return <p className="muted">No confusion matrix data yet.</p>;
  }

  const getLabel = (idx: number) => {
    if (classNames && classNames[idx]) return classNames[idx];
    if (labels[idx] !== undefined) return String(labels[idx]);
    return String(idx);
  };

  return (
    <div className="cm-wrapper">
      <div className="cm-grid" style={{ gridTemplateColumns: `48px repeat(${matrix[0]?.length ?? 0}, minmax(28px, 1fr))` }}>
        {/* header row */}
        <div className="cm-corner">T\P</div>
        {(matrix[0] ?? []).map((_, ci) => (
          <div key={`h${ci}`} className="cm-header">{getLabel(ci)}</div>
        ))}

        {/* data rows */}
        {matrix.map((row, ri) => (
          <>
            <div key={`r${ri}`} className="cm-row-label">{getLabel(ri)}</div>
            {row.map((val, ci) => {
              const intensity = val / maxVal;
              // Diagonal (correct predictions) = green, off-diagonal = blue
              const bg =
                ri === ci
                  ? `rgba(22, 163, 74, ${0.15 + 0.85 * intensity})`
                  : `rgba(37, 99, 235, ${0.10 + 0.70 * intensity})`;
              // Always use high-contrast text: light text on high-intensity cells,
              // theme-appropriate readable text on low-intensity cells.
              const textColor =
                intensity > 0.4
                  ? cc.matrixTextLight
                  : isDark
                    ? "#e2e8f0"   // light gray on dark background
                    : "#1e293b";  // dark gray on light background
              // Add text shadow for readability on all intensity levels
              const textShadow =
                intensity > 0.4
                  ? "0 1px 2px rgba(0,0,0,0.5)"
                  : isDark
                    ? "0 1px 2px rgba(0,0,0,0.8)"
                    : "none";
              return (
                <div
                  key={`${ri}-${ci}`}
                  className="cm-cell"
                  style={{
                    backgroundColor: bg,
                    color: textColor,
                    textShadow,
                    border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
                  }}
                  title={`True: ${getLabel(ri)}, Pred: ${getLabel(ci)}, Count: ${val}`}
                >
                  {val}
                </div>
              );
            })}
          </>
        ))}
      </div>
      <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
        Rows = true label, Columns = predicted label.
        Branch: {latest.branch}, epoch {latest.epoch}.
      </p>
    </div>
  );
}
