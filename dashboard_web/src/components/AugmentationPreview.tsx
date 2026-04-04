import { useEffect, useMemo, useRef, useState } from "react";
import type { ProbeItem, SamplePoint, StatePayload } from "../types";
import { resolveArtifact } from "../hooks/useRunState";

interface Props {
  state: StatePayload;
  activeRun: string;
  offline: boolean;
}

/** Human-readable label for a timeline entry */
function chipLabel(entry: SamplePoint): string {
  if (entry.branch === "baseline") return `baseline e${entry.epoch}`;
  // Post-baseline entries: show branch name (iteration context)
  return entry.branch;
}

/** Unique key for a timeline entry (handles duplicate epochs) */
function chipKey(entry: SamplePoint, idx: number): string {
  return `${entry.iteration}_${entry.epoch}_${entry.branch}_${idx}`;
}

type XaiPanel = "gt" | "saliency" | "pred";

function panelToIndex(panel: XaiPanel): number {
  if (panel === "gt") return 0;
  if (panel === "saliency") return 1;
  return 2;
}

function ImageRegionCanvas({
  src,
  outSize = 512,
  className,
  alt,
  panel,
  crop,
  imageSize,
}: {
  src: string;
  outSize?: number;
  className?: string;
  alt?: string;
  panel?: XaiPanel;
  crop?: [number, number, number, number];
  imageSize?: [number, number];
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !src) return;
    const img = new Image();
    img.onload = () => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      canvas.width = outSize;
      canvas.height = outSize;
      ctx.clearRect(0, 0, outSize, outSize);
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      let sx = 0;
      let sy = 0;
      let sw = img.width;
      let sh = img.height;

      if (panel !== undefined) {
        // XAI artifacts are 3 horizontal panels: GT | Saliency | Pred+Saliency
        const panelW = img.width / 3;
        sx = panelToIndex(panel) * panelW;
        sw = panelW;
      }

      if (crop && imageSize) {
        const [ih, iw] = imageSize;
        const [x1, y1, x2, y2] = crop;
        const localX1 = (x1 / Math.max(1, iw)) * sw;
        const localY1 = (y1 / Math.max(1, ih)) * sh;
        const localX2 = (x2 / Math.max(1, iw)) * sw;
        const localY2 = (y2 / Math.max(1, ih)) * sh;
        sx += localX1;
        sy += localY1;
        sw = Math.max(1, localX2 - localX1);
        sh = Math.max(1, localY2 - localY1);
      }
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, outSize, outSize);
    };
    img.src = src;
  }, [src, outSize, panel, crop, imageSize]);

  return <canvas ref={canvasRef} className={className ?? "preview-img"} aria-label={alt ?? "image-region"} />;
}

function BoxOverlayPreview({
  src,
  alt,
  box,
  imageSize,
  panel,
}: {
  src: string;
  alt: string;
  box: [number, number, number, number];
  imageSize: [number, number];
  panel?: XaiPanel;
}) {
  const [h, w] = imageSize;
  const [x1, y1, x2, y2] = box;
  const left = `${Math.max(0, (x1 / Math.max(1, w)) * 100)}%`;
  const top = `${Math.max(0, (y1 / Math.max(1, h)) * 100)}%`;
  const width = `${Math.max(0, ((x2 - x1) / Math.max(1, w)) * 100)}%`;
  const height = `${Math.max(0, ((y2 - y1) / Math.max(1, h)) * 100)}%`;

  return (
    <div className="bbox-overlay-wrap">
      <ImageRegionCanvas
        src={src}
        alt={alt}
        className="preview-img"
        panel={panel}
      />
      <div className="bbox-overlay-rect" style={{ left, top, width, height }} />
    </div>
  );
}

export function AugmentationPreview({ state, activeRun, offline }: Props) {
  const probes = state.probe_set ?? [];
  const [selectedProbe, setSelectedProbe] = useState<string>("");
  const [selectedIdx, setSelectedIdx] = useState(0);

  // auto-select first probe
  useEffect(() => {
    if (!selectedProbe && probes.length > 0) {
      setSelectedProbe(probes[0].sample_id);
    }
  }, [probes, selectedProbe]);

  const timeline: SamplePoint[] = useMemo(
    () => state.sample_timelines?.[selectedProbe] ?? [],
    [state.sample_timelines, selectedProbe],
  );

  // auto-set to latest entry when timeline changes
  useEffect(() => {
    if (timeline.length > 0) {
      setSelectedIdx(timeline.length - 1);
    }
  }, [timeline]);

  const current = useMemo(
    () => timeline[selectedIdx] ?? timeline[timeline.length - 1] ?? null,
    [timeline, selectedIdx],
  );

  // group probes by class for quick selection
  const probesByClass = useMemo(() => {
    const map = new Map<number, ProbeItem[]>();
    probes.forEach((p) => {
      const arr = map.get(p.class_id) ?? [];
      arr.push(p);
      map.set(p.class_id, arr);
    });
    return map;
  }, [probes]);

  // Determine phases: baseline epochs vs decision branches
  const phases = useMemo(() => {
    const result: { label: string; startIdx: number; endIdx: number; type: "baseline" | "branch" }[] = [];
    let i = 0;
    while (i < timeline.length) {
      const entry = timeline[i];
      if (entry.branch === "baseline") {
        const start = i;
        while (i < timeline.length && timeline[i].branch === "baseline") i++;
        result.push({
          label: `Baseline (e1–e${timeline[i - 1].epoch})`,
          startIdx: start,
          endIdx: i - 1,
          type: "baseline",
        });
      } else {
        result.push({
          label: `${entry.branch}`,
          startIdx: i,
          endIdx: i,
          type: "branch",
        });
        i++;
      }
    }
    return result;
  }, [timeline]);

  const isBaseline = current?.branch === "baseline";
  const originalSrc = current ? resolveArtifact(current.artifacts?.original, activeRun, offline) : "";

  if (probes.length === 0) {
    return <p className="muted">No probe samples available.</p>;
  }

  return (
    <div>
      {/* Probe selector + position label */}
      <div className="preview-controls">
        <label>
          Probe sample:
          <select
            value={selectedProbe}
            onChange={(e) => setSelectedProbe(e.target.value)}
          >
            {Array.from(probesByClass.entries())
              .sort(([a], [b]) => a - b)
              .map(([cls, items]) => (
                <optgroup key={cls} label={`Class ${cls}`}>
                  {items.map((p) => (
                    <option key={p.sample_id} value={p.sample_id}>
                      {p.sample_id}
                    </option>
                  ))}
                </optgroup>
              ))}
          </select>
        </label>

        {timeline.length > 1 && (
          <label>
            Step: {selectedIdx + 1}/{timeline.length}
            <input
              type="range"
              min={0}
              max={timeline.length - 1}
              value={selectedIdx}
              onChange={(e) => setSelectedIdx(Number(e.target.value))}
              style={{ touchAction: "pan-x", width: "100%", minWidth: 120 }}
            />
          </label>
        )}
      </div>

      {/* Current prediction info */}
      {current && (
        <div className="preview-info">
          <span className="preview-phase-label">
            {isBaseline
              ? `Baseline · Epoch ${current.epoch}`
              : `After Decision ${current.iteration} · ${current.branch}`}
          </span>
          <span className="preview-separator">|</span>
          <span>Pred: <strong>{current.predicted_class}</strong></span>
          <span>True: <strong>{current.true_class}</strong></span>
          <span>Conf: <strong>{(current.confidence * 100).toFixed(1)}%</strong></span>
          <span
            className={`tooltip-badge ${current.predicted_class === current.true_class ? "badge-selected" : "badge-evaluated"}`}
          >
            {current.predicted_class === current.true_class ? "CORRECT" : "WRONG"}
          </span>
        </div>
      )}

      {/* Image grid at 512x512 — Original + XAI (augmentation preview moved to Branch Tree) */}
      {current && (
        <div className="preview-grid preview-grid--2col">
          <figure>
            <figcaption>Original</figcaption>
            {originalSrc ? (
              <img
                src={originalSrc}
                alt="original"
                className="preview-img"
              />
            ) : (
              <div className="empty-box">
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>🖼️</div>
                  <div>Original preview is not available yet</div>
                </div>
              </div>
            )}
          </figure>
          <figure>
            <figcaption>XAI Heatmap</figcaption>
            {current.artifacts?.xai ? (
              <img
                src={resolveArtifact(current.artifacts.xai, activeRun, offline)}
                alt="xai"
                className="preview-img"
              />
            ) : (
              <div className="empty-box">
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>🔍</div>
                  <div>No XAI heatmap for this step</div>
                  <div style={{ fontSize: 11, marginTop: 4, opacity: 0.7 }}>
                    XAI is generated every {String(state.run?.xai_every ?? 5)} epochs
                  </div>
                </div>
              </div>
            )}
          </figure>
        </div>
      )}

      {/* Phase-based timeline strip */}
      {phases.length > 0 && (
        <div className="preview-phases">
          {phases.map((phase, pi) => (
            <div key={pi} className="phase-group">
              <div className="phase-label">{phase.type === "baseline" ? "BASELINE" : `DECISION ${timeline[phase.startIdx].iteration}`}</div>
              <div className="phase-chips">
                {phase.type === "baseline" ? (
                  // Show individual epoch chips for baseline
                  Array.from({ length: phase.endIdx - phase.startIdx + 1 }, (_, j) => {
                    const idx = phase.startIdx + j;
                    const entry = timeline[idx];
                    return (
                      <button
                        key={chipKey(entry, idx)}
                        className={`timeline-chip ${idx === selectedIdx ? "active" : ""} ${
                          entry.predicted_class === entry.true_class ? "correct" : "wrong"
                        }`}
                        onClick={() => setSelectedIdx(idx)}
                        title={`Baseline epoch ${entry.epoch}: pred=${entry.predicted_class} conf=${(entry.confidence * 100).toFixed(0)}%`}
                      >
                        e{entry.epoch}
                      </button>
                    );
                  })
                ) : (
                  // Show branch name chip for post-decision entries
                  <button
                    className={`timeline-chip branch-chip ${phase.startIdx === selectedIdx ? "active" : ""} ${
                      timeline[phase.startIdx].predicted_class === timeline[phase.startIdx].true_class ? "correct" : "wrong"
                    }`}
                    onClick={() => setSelectedIdx(phase.startIdx)}
                    title={`After selecting ${timeline[phase.startIdx].branch}: pred=${timeline[phase.startIdx].predicted_class} conf=${(timeline[phase.startIdx].confidence * 100).toFixed(0)}%`}
                  >
                    {timeline[phase.startIdx].branch}
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
