import { useMemo, useState } from "react";
import { useChartColors } from "../ThemeContext";

const COLORS = [
  "#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#0891b2",
  "#d97706", "#4f46e5", "#0f766e", "#be185d", "#65a30d",
  "#1d4ed8", "#15803d", "#b91c1c", "#6d28d9", "#0e7490",
];

interface Props {
  matrix: number[][];
  classNames: string[];
}

type ChordArc = {
  classIdx: number;
  startAngle: number;
  endAngle: number;
  total: number;
};

type ChordRibbon = {
  sourceIdx: number;
  targetIdx: number;
  sourceStart: number;
  sourceEnd: number;
  targetStart: number;
  targetEnd: number;
  value: number;
};

type ConfusedPair = {
  classA: number;
  classB: number;
  nameA: string;
  nameB: string;
  count: number;
};

/**
 * ConfusionChord — custom SVG chord diagram showing misclassification flow
 * between classes. Arcs are sized by total misclassifications involving
 * that class; ribbons connect confused pairs.
 */
export function ConfusionChord({ matrix, classNames }: Props) {
  const cc = useChartColors();
  const [hovered, setHovered] = useState<{ src: number; tgt: number } | null>(null);

  const n = matrix.length;

  // Build off-diagonal flow matrix (only misclassifications)
  const flow = useMemo(() => {
    const f: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i !== j) f[i][j] = matrix[i][j];
      }
    }
    return f;
  }, [matrix, n]);

  // Total off-diagonal per class
  const classTotals = useMemo(() => {
    return Array.from({ length: n }, (_, i) => {
      let total = 0;
      for (let j = 0; j < n; j++) {
        total += flow[i][j] + flow[j][i];
      }
      // Divide by 2 because we double-count symmetric pairs
      // Actually, flow[i][j] and flow[j][i] are different (A→B vs B→A)
      // We want total involvement = sum of outgoing + incoming for each class
      // but for arc sizing, use outgoing only to avoid double-counting in chords
      let outgoing = 0;
      for (let j = 0; j < n; j++) outgoing += flow[i][j];
      return outgoing;
    });
  }, [flow, n]);

  const grandTotal = useMemo(() => classTotals.reduce((s, v) => s + v, 0), [classTotals]);

  // Top confused pairs (bidirectional: A↔B = flow[A][B] + flow[B][A])
  const topPairs = useMemo(() => {
    const pairs: ConfusedPair[] = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const count = flow[i][j] + flow[j][i];
        if (count > 0) {
          pairs.push({
            classA: i,
            classB: j,
            nameA: classNames[i] ?? `${i}`,
            nameB: classNames[j] ?? `${j}`,
            count,
          });
        }
      }
    }
    pairs.sort((a, b) => b.count - a.count);
    return pairs.slice(0, 7);
  }, [flow, n, classNames]);

  // Compute arcs
  const GAP = 0.02; // radians gap between arcs
  const arcs = useMemo<ChordArc[]>(() => {
    if (grandTotal === 0) return [];
    const totalGap = GAP * n;
    const available = 2 * Math.PI - totalGap;
    let angle = 0;
    return Array.from({ length: n }, (_, i) => {
      const span = (classTotals[i] / grandTotal) * available;
      const arc: ChordArc = {
        classIdx: i,
        startAngle: angle,
        endAngle: angle + span,
        total: classTotals[i],
      };
      angle += span + GAP;
      return arc;
    });
  }, [classTotals, grandTotal, n]);

  // Compute ribbons
  const ribbons = useMemo<ChordRibbon[]>(() => {
    if (arcs.length === 0 || grandTotal === 0) return [];
    // For each arc, track consumed angle
    const consumed = arcs.map((a) => a.startAngle);
    const result: ChordRibbon[] = [];
    for (let i = 0; i < n; i++) {
      const arcI = arcs[i];
      const arcSpan = arcI.endAngle - arcI.startAngle;
      for (let j = 0; j < n; j++) {
        if (i === j || flow[i][j] === 0) continue;
        const fraction = classTotals[i] > 0 ? flow[i][j] / classTotals[i] : 0;
        const ribbonSpan = fraction * arcSpan;
        const sourceStart = consumed[i];
        const sourceEnd = consumed[i] + ribbonSpan;
        consumed[i] = sourceEnd;
        // Find corresponding target sub-arc
        const arcJ = arcs[j];
        const arcJSpan = arcJ.endAngle - arcJ.startAngle;
        // Target: how much of j's incoming is from i
        let jIncoming = 0;
        for (let k = 0; k < n; k++) jIncoming += flow[k][j];
        const tFraction = jIncoming > 0 ? flow[i][j] / jIncoming : 0;
        // We need a separate consumed tracker for target side
        // Simplify: allocate from end of arc backwards for targets
        const tSpan = tFraction * arcJSpan;
        // For simplicity, we'll just place target ribbons sequentially
        // This is an approximation but looks good visually
        result.push({
          sourceIdx: i,
          targetIdx: j,
          sourceStart,
          sourceEnd,
          targetStart: 0, // will be set below
          targetEnd: 0,
          value: flow[i][j],
        });
      }
    }
    // Now compute target positions — group by target, allocate sequentially
    const targetConsumed = arcs.map((a) => a.endAngle); // start from end, go backwards
    for (const r of result) {
      const arcJ = arcs[r.targetIdx];
      const arcJSpan = arcJ.endAngle - arcJ.startAngle;
      let jIncoming = 0;
      for (let k = 0; k < n; k++) jIncoming += flow[k][r.targetIdx];
      const tFraction = jIncoming > 0 ? r.value / jIncoming : 0;
      const tSpan = tFraction * arcJSpan;
      targetConsumed[r.targetIdx] -= tSpan;
      r.targetStart = targetConsumed[r.targetIdx];
      r.targetEnd = targetConsumed[r.targetIdx] + tSpan;
    }
    return result;
  }, [arcs, flow, n, grandTotal, classTotals]);

  if (grandTotal === 0) {
    return <p className="muted">No misclassifications to display.</p>;
  }

  const SIZE = 340;
  const CX = SIZE / 2;
  const CY = SIZE / 2;
  const R = SIZE / 2 - 40;
  const ARC_WIDTH = 14;

  function polarToXY(angle: number, r: number): [number, number] {
    // Rotate -90° so 0 is at top
    const a = angle - Math.PI / 2;
    return [CX + r * Math.cos(a), CY + r * Math.sin(a)];
  }

  function arcPath(startAngle: number, endAngle: number, innerR: number, outerR: number): string {
    const [x1, y1] = polarToXY(startAngle, outerR);
    const [x2, y2] = polarToXY(endAngle, outerR);
    const [x3, y3] = polarToXY(endAngle, innerR);
    const [x4, y4] = polarToXY(startAngle, innerR);
    const large = endAngle - startAngle > Math.PI ? 1 : 0;
    return [
      `M ${x1} ${y1}`,
      `A ${outerR} ${outerR} 0 ${large} 1 ${x2} ${y2}`,
      `L ${x3} ${y3}`,
      `A ${innerR} ${innerR} 0 ${large} 0 ${x4} ${y4}`,
      "Z",
    ].join(" ");
  }

  function ribbonPath(r: ChordRibbon): string {
    const [sx1, sy1] = polarToXY(r.sourceStart, R - ARC_WIDTH);
    const [sx2, sy2] = polarToXY(r.sourceEnd, R - ARC_WIDTH);
    const [tx1, ty1] = polarToXY(r.targetStart, R - ARC_WIDTH);
    const [tx2, ty2] = polarToXY(r.targetEnd, R - ARC_WIDTH);
    return [
      `M ${sx1} ${sy1}`,
      `Q ${CX} ${CY} ${tx1} ${ty1}`,
      `A ${R - ARC_WIDTH} ${R - ARC_WIDTH} 0 0 1 ${tx2} ${ty2}`,
      `Q ${CX} ${CY} ${sx2} ${sy2}`,
      `A ${R - ARC_WIDTH} ${R - ARC_WIDTH} 0 0 1 ${sx1} ${sy1}`,
      "Z",
    ].join(" ");
  }

  const getName = (idx: number) => classNames[idx] ?? `${idx}`;

  return (
    <div className="chord-layout">
      <div className="chord-svg-wrap">
        <svg viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ width: "100%", maxWidth: SIZE, height: "auto" }}>
          {/* Ribbons */}
          {ribbons.map((r, i) => {
            const isHovered =
              hovered &&
              ((hovered.src === r.sourceIdx && hovered.tgt === r.targetIdx) ||
                (hovered.src === r.targetIdx && hovered.tgt === r.sourceIdx));
            const anyHovered = hovered !== null;
            return (
              <path
                key={`ribbon-${i}`}
                d={ribbonPath(r)}
                fill={COLORS[r.sourceIdx % COLORS.length]}
                opacity={anyHovered ? (isHovered ? 0.65 : 0.08) : 0.35}
                stroke={isHovered ? COLORS[r.sourceIdx % COLORS.length] : "none"}
                strokeWidth={isHovered ? 1 : 0}
                onMouseEnter={() => setHovered({ src: r.sourceIdx, tgt: r.targetIdx })}
                onMouseLeave={() => setHovered(null)}
                style={{ transition: "opacity 0.15s" }}
              >
                <title>
                  {getName(r.sourceIdx)} → {getName(r.targetIdx)}: {r.value}
                </title>
              </path>
            );
          })}
          {/* Arcs */}
          {arcs.map((arc) => {
            if (arc.endAngle - arc.startAngle < 0.001) return null;
            const isHovered =
              hovered && (hovered.src === arc.classIdx || hovered.tgt === arc.classIdx);
            return (
              <path
                key={`arc-${arc.classIdx}`}
                d={arcPath(arc.startAngle, arc.endAngle, R - ARC_WIDTH, R)}
                fill={COLORS[arc.classIdx % COLORS.length]}
                opacity={hovered ? (isHovered ? 1 : 0.4) : 0.85}
                stroke={cc.arcStroke}
                strokeWidth={1}
                style={{ transition: "opacity 0.15s" }}
              >
                <title>
                  {getName(arc.classIdx)}: {arc.total} misclassifications (outgoing)
                </title>
              </path>
            );
          })}
          {/* Labels */}
          {arcs.map((arc) => {
            const midAngle = (arc.startAngle + arc.endAngle) / 2;
            const [lx, ly] = polarToXY(midAngle, R + 16);
            const span = arc.endAngle - arc.startAngle;
            if (span < 0.15) return null; // skip label for tiny arcs
            return (
              <text
                key={`label-${arc.classIdx}`}
                x={lx}
                y={ly}
                textAnchor="middle"
                dominantBaseline="central"
                fontSize={10}
                fontWeight={600}
                fill={cc.chordLabelFill}
              >
                {getName(arc.classIdx)}
              </text>
            );
          })}
        </svg>
        {hovered && (
          <div className="chord-tooltip">
            <strong>{getName(hovered.src)}</strong> ↔ <strong>{getName(hovered.tgt)}</strong>
            <br />
            {flow[hovered.src][hovered.tgt]} + {flow[hovered.tgt][hovered.src]} ={" "}
            {flow[hovered.src][hovered.tgt] + flow[hovered.tgt][hovered.src]} mis.
          </div>
        )}
      </div>
      <div className="chord-pairs-list">
        <h4>Top Confused Pairs</h4>
        {topPairs.map((p, i) => (
          <div
            key={`${p.classA}-${p.classB}`}
            className="chord-pair-row"
            onMouseEnter={() => setHovered({ src: p.classA, tgt: p.classB })}
            onMouseLeave={() => setHovered(null)}
          >
            <span className="chord-pair-rank">#{i + 1}</span>
            <span
              className="chord-pair-dot"
              style={{ background: COLORS[p.classA % COLORS.length] }}
            />
            <span className="chord-pair-name">{p.nameA}</span>
            <span className="chord-pair-arrow">↔</span>
            <span
              className="chord-pair-dot"
              style={{ background: COLORS[p.classB % COLORS.length] }}
            />
            <span className="chord-pair-name">{p.nameB}</span>
            <span className="chord-pair-count">{p.count}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
