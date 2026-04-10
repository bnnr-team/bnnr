import { useMemo } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricPoint } from "../types";
import { useChartColors } from "../ThemeContext";
import { primaryMetricValue, secondaryMetricValue, type RunTask } from "../taskMetrics";

function pct(v: number) {
  return Math.max(0, Math.min(100, v * 100));
}

/* branch -> color: cycle through palette */
const BRANCH_COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
  "#ea580c", "#0284c7", "#9333ea", "#059669", "#e11d48",
];

interface Props {
  timeline: MetricPoint[];
  selectedPath?: string[];
  task?: RunTask;
}

export function MetricsCharts({ timeline, selectedPath, task = "classification" }: Props) {
  const cc = useChartColors();
  const isMultilabel = task === "multilabel";
  const isDetection = task === "detection";

  /* ---- Determine trunk branch names (raw augmentation names) ---- */
  const trunkBranchNames = useMemo(() => {
    const set = new Set<string>(["baseline"]);
    for (const p of selectedPath ?? []) {
      const parts = p.split(":");
      if (parts.length > 1) set.add(parts.slice(1).join(":"));
      else set.add(p);
    }
    return set;
  }, [selectedPath]);

  /* ---- Trunk-only training progress (first chart) ---- */
  const trunkRows = useMemo(() => {
    return timeline
      .filter((p) => trunkBranchNames.has(p.branch))
      .map((p, i) => ({
        ...p,
        idx: i,
        accuracy_pct: pct(primaryMetricValue(p, task)),
        f1_pct: pct(secondaryMetricValue(p, task)),
        label: `it${p.iteration}/e${p.epoch}`,
      }));
  }, [timeline, trunkBranchNames, task]);

  /* ---- Decision markers for trunk chart ---- */
  const decisionMarkers = useMemo(() => {
    const markers: { label: string; branch: string }[] = [];
    let prevBranch = "";
    for (const r of trunkRows) {
      if (r.branch !== prevBranch && prevBranch !== "") {
        markers.push({ label: r.label, branch: r.branch });
      }
      prevBranch = r.branch;
    }
    return markers;
  }, [trunkRows]);

  /* ==================================================================
   * "Accuracy by Branch" chart — unique branch keys per iteration
   * so lines DON'T connect across iterations.
   * 
   * Each iteration's candidates get keys like "I1:church_noise",
   * "I2:church_noise" — these are DIFFERENT lines even if the
   * augmentation name is the same, because they start from
   * different model states (the winner of the previous iteration).
   * ================================================================ */

  /** Build unique branch key: baseline stays "baseline", others get "I{iter}:{name}" */
  const makeBranchKey = (p: MetricPoint) =>
    p.branch === "baseline" ? "baseline" : `I${p.iteration}:${p.branch}`;

  /* Unique branch keys in order of appearance */
  const branchKeys = useMemo(() => {
    const seen = new Set<string>();
    const result: string[] = [];
    timeline.forEach((p) => {
      const key = makeBranchKey(p);
      if (!seen.has(key)) {
        seen.add(key);
        result.push(key);
      }
    });
    return result;
  }, [timeline]);

  /* Color map: same augmentation name gets same color across iterations */
  const branchColorMap = useMemo(() => {
    const nameColorIdx: Record<string, number> = {};
    let colorIdx = 0;
    const map: Record<string, string> = {};
    for (const key of branchKeys) {
      // Extract raw name: "I1:church_noise" -> "church_noise", "baseline" -> "baseline"
      const rawName = key.includes(":") ? key.split(":").slice(1).join(":") : key;
      if (!(rawName in nameColorIdx)) {
        nameColorIdx[rawName] = colorIdx++;
      }
      map[key] = BRANCH_COLORS[nameColorIdx[rawName] % BRANCH_COLORS.length];
    }
    return map;
  }, [branchKeys]);

  /* Determine which branch keys are on the trunk (selected path) */
  const trunkKeys = useMemo(() => {
    const set = new Set<string>(["baseline"]);
    for (const p of selectedPath ?? []) {
      // selectedPath entries like "iter_1:church_noise"
      const parts = p.split(":");
      if (parts.length > 1) {
        const rawName = parts.slice(1).join(":");
        // Extract iteration number from "iter_1"
        const iterMatch = parts[0].match(/(\d+)/);
        if (iterMatch) {
          set.add(`I${iterMatch[1]}:${rawName}`);
        }
      }
    }
    return set;
  }, [selectedPath]);

  /* ---- Best-epoch marker: ONE star total — the global best metric ---- */
  const bestEpochKey = useMemo(() => {
    let bestBranchKey = "";
    let bestLabel = "";
    let bestVal = -1;
    timeline.forEach((p) => {
      const val = primaryMetricValue(p, task);
      if (val > bestVal) {
        bestVal = val;
        bestBranchKey = makeBranchKey(p);
        // Reconstruct the label that will be used in the chart
        bestLabel = p.iteration === 0 ? `B:e${p.epoch}` : `I${p.iteration}:e${p.epoch}`;
      }
    });
    return { branchKey: bestBranchKey, label: bestLabel };
  }, [timeline, task]);

  /**
   * Build unified rows for "Accuracy by Branch" chart.
   * Each iteration's candidates are separate lines (unique keys).
   * Bridge points connect each new iteration's candidates from
   * the trunk endpoint of the previous iteration.
   */
  const { unifiedRows, phaseBoundaries } = useMemo(() => {
    if (timeline.length === 0) return { unifiedRows: [], phaseBoundaries: [] as number[] };

    const iterSet = new Set<number>();
    timeline.forEach((p) => iterSet.add(p.iteration));
    const iterations = Array.from(iterSet).sort((a, b) => a - b);

    const allRows: Record<string, number | string | null>[] = [];
    const boundaries: number[] = [];
    let trunkEndAcc: number | null = null;

    for (let ii = 0; ii < iterations.length; ii++) {
      const iter = iterations[ii];
      const iterPoints = timeline.filter((p) => p.iteration === iter);

      // Branch keys for THIS iteration
      const iterBranchKeys = new Set<string>();
      iterPoints.forEach((p) => iterBranchKeys.add(makeBranchKey(p)));

      const epochSet = new Set<number>();
      iterPoints.forEach((p) => epochSet.add(p.epoch));
      const epochs = Array.from(epochSet).sort((a, b) => a - b);

      // Bridge: connect new iteration's branches from previous trunk endpoint
      if (ii > 0 && trunkEndAcc !== null && allRows.length > 0) {
        const lastRow = allRows[allRows.length - 1];
        for (const bk of iterBranchKeys) {
          if (lastRow[bk] === null || lastRow[bk] === undefined) {
            lastRow[bk] = trunkEndAcc;
          }
        }
        // Mark boundary at the last row of previous iteration
        boundaries.push(allRows.length - 1);
      }

      for (const epoch of epochs) {
        const epochPoints = iterPoints.filter((p) => p.epoch === epoch);
        const lbl = iter === 0 ? `B:e${epoch}` : `I${iter}:e${epoch}`;
        const row: Record<string, number | string | null> = { label: lbl };
        // Initialize ALL branch keys to null
        for (const bk of branchKeys) {
          row[bk] = null;
        }
        // Fill in values for branches with data at this epoch
        for (const p of epochPoints) {
          const metricVal = primaryMetricValue(p, task);
          row[makeBranchKey(p)] = pct(metricVal);
        }
        allRows.push(row);
      }

      // Trunk endpoint: use the selected (trunk) branch's primary metric
      const lastEpoch = epochs[epochs.length - 1];
      const lastEpochPoints = iterPoints.filter((p) => p.epoch === lastEpoch);
      // Prefer the trunk branch's value
      const trunkPoint = lastEpochPoints.find((p) => trunkKeys.has(makeBranchKey(p)) || trunkBranchNames.has(p.branch));
      if (trunkPoint) {
        const metricVal = primaryMetricValue(trunkPoint, task);
        trunkEndAcc = pct(metricVal);
      } else if (lastEpochPoints.length > 0) {
        trunkEndAcc = Math.max(...lastEpochPoints.map((p) =>
          pct(primaryMetricValue(p, task))));
      }
    }

    return { unifiedRows: allRows, phaseBoundaries: boundaries };
  }, [timeline, branchKeys, trunkKeys, trunkBranchNames, task]);

  if (trunkRows.length === 0) {
    return <p className="muted">No metrics recorded yet.</p>;
  }

  return (
    <div>
      {/* Training Progress — trunk only (baseline + selected branches) */}
      <h4>Training Progress (Selected Path)</h4>
      <p className="muted" style={{ fontSize: 11, marginTop: -4, marginBottom: 8 }}>
        Only the selected augmentation path. Vertical lines mark augmentation decisions.
      </p>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={trunkRows}>
            <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
            <XAxis dataKey="label" tick={{ fontSize: 11, fill: cc.text }} interval="preserveStartEnd" />
            <YAxis yAxisId="pct" orientation="left" unit="%" domain={[0, 100]} tick={{ fontSize: 11, fill: cc.text }} />
            <YAxis yAxisId="loss" orientation="right" tick={{ fontSize: 11, fill: cc.text }} />
            <Tooltip contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }} />
            <Legend />
            {decisionMarkers.map((m) => (
              <ReferenceLine
                key={m.label}
                yAxisId="pct"
                x={m.label}
                stroke={cc.referenceLine}
                strokeDasharray="4 4"
                strokeWidth={1.5}
                label={{ value: m.branch, position: "top", fontSize: 9, fill: cc.refLabelFill }}
              />
            ))}
            <Line yAxisId="pct" dataKey="accuracy_pct" stroke="#16a34a" dot={false} name={isDetection ? "mAP@50 (%)" : isMultilabel ? "F1 samples (%)" : "accuracy (%)"} strokeWidth={2} />
            <Line yAxisId="pct" dataKey="f1_pct" stroke="#2563eb" dot={false} name={isDetection ? "mAP@50:95 (%)" : "f1 (%)"} strokeWidth={2} />
            <Line yAxisId="loss" dataKey="loss" stroke="#dc2626" dot={false} name="loss" strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Per-branch accuracy comparison */}
      {branchKeys.length > 1 && unifiedRows.length > 0 && (
        <>
          <h4 style={{ marginTop: 16 }}>{isDetection ? "mAP@50 by Branch" : isMultilabel ? "F1 (samples) by Branch" : "Accuracy by Branch"}</h4>
          <p className="muted" style={{ fontSize: 11, marginTop: -4, marginBottom: 8 }}>
            Each iteration&apos;s candidates are separate lines. All candidates in an iteration start from the previous winner.
          </p>
          <div className="chart-box chart-box--branches">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={unifiedRows}>
                <CartesianGrid strokeDasharray="3 3" stroke={cc.grid} />
                <XAxis dataKey="label" tick={{ fontSize: 10, fill: cc.text }} interval="preserveStartEnd" />
                <YAxis unit="%" domain={[0, 100]} tick={{ fontSize: 11, fill: cc.text }} />
                <Tooltip contentStyle={{ background: cc.tooltipBg, border: `1px solid ${cc.tooltipBorder}`, borderRadius: 8 }} />
                <Legend
                  wrapperStyle={{ maxHeight: 64, overflowY: 'auto', overflowX: 'hidden', fontSize: 11, lineHeight: '18px' }}
                  iconSize={10}
                />
                {phaseBoundaries.map((idx) => (
                  <ReferenceLine
                    key={`boundary-${idx}`}
                    x={unifiedRows[idx]?.label as string}
                    stroke={cc.text}
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                  />
                ))}
                {branchKeys.map((bk) => {
                  const isTrunk = trunkKeys.has(bk) || bk === "baseline";
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const renderDot = (bk === bestEpochKey.branchKey) ? (props: any) => {
                    const { cx, cy, payload } = props;
                    if (!cx || !cy || !payload) return <g />;
                    if (payload.label === bestEpochKey.label) {
                      return (
                        <svg key={`star-${bk}`} x={cx - 8} y={cy - 8} width={16} height={16} viewBox="0 0 24 24">
                          <polygon
                            points="12,2 15.09,8.26 22,9.27 17,14.14 18.18,21.02 12,17.77 5.82,21.02 7,14.14 2,9.27 8.91,8.26"
                            fill="#f0a069"
                            stroke="#8b4a25"
                            strokeWidth="1.5"
                          />
                        </svg>
                      );
                    }
                    return <g />;
                  } : false;
                  return (
                    <Line
                      key={bk}
                      dataKey={bk}
                      name={bk}
                      stroke={branchColorMap[bk]}
                      dot={renderDot}
                      strokeWidth={isTrunk ? 2.5 : 1.5}
                      strokeDasharray={isTrunk ? undefined : "4 2"}
                      connectNulls={false}
                    />
                  );
                })}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
