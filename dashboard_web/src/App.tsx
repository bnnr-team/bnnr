import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { TabId } from "./types";
import { useRunState } from "./hooks/useRunState";
import { useTheme } from "./ThemeContext";
import { BranchGraph } from "./components/BranchGraph";
import { MetricsCharts } from "./components/MetricsCharts";
import { PerClassMetrics } from "./components/PerClassMetrics";
import { DecisionLedger } from "./components/DecisionLedger";
import { CandidateRadar } from "./components/CandidateRadar";
import { AugmentationPreview } from "./components/AugmentationPreview";
import { ConfusionMatrix } from "./components/ConfusionMatrix";
import { AugmentationImpact } from "./components/AugmentationImpact";
import { PerClassDeltaHeatmap } from "./components/PerClassDeltaHeatmap";
import { AugmentationStack } from "./components/AugmentationStack";
import { LossLandscape } from "./components/LossLandscape";
import { AccuracyGainRate } from "./components/AccuracyGainRate";
import { DatasetInsight } from "./components/DatasetInsight";
import { XAIInsights } from "./components/XAIInsights";
import { XAIQualityTrend } from "./components/XAIQualityTrend";

/* ---- Inline SVG icons (matching lucide style from website) ---- */
const svgProps = { width: 16, height: 16, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };

const IconOverview = <svg {...svgProps}><path d="M21.21 15.89A10 10 0 1 1 8 2.83"/><path d="M22 12A10 10 0 0 0 12 2v10z"/></svg>;
const IconTree = <svg {...svgProps}><line x1="6" y1="3" x2="6" y2="15"/><circle cx="18" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><path d="M18 9a9 9 0 0 1-9 9"/></svg>;
const IconMetrics = <svg {...svgProps}><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>;
const IconSamples = <svg {...svgProps}><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>;
const IconAnalysis = <svg {...svgProps}><path d="M12 2a8 8 0 1 0 8 8h-8z"/><path d="M20.2 10A8 8 0 0 0 14 3.8V10z"/></svg>;
const IconInsight = <svg {...svgProps}><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>;

const TABS: { id: TabId; label: string; icon: ReactNode }[] = [
  { id: "overview", label: "Overview", icon: IconOverview },
  { id: "tree", label: "Branch Tree", icon: IconTree },
  { id: "metrics", label: "Metrics", icon: IconMetrics },
  { id: "samples", label: "Samples & XAI", icon: IconSamples },
  { id: "analysis", label: "Analysis", icon: IconAnalysis },
  { id: "insight", label: "Dataset Insight", icon: IconInsight },
];

export function App() {
  const { runs, activeRun, setActiveRun, state, offline, loading, dashboardMode } = useRunState();
  const { theme, toggleTheme } = useTheme();
  const [tab, setTab] = useState<TabId>("overview");

  const decisions = useMemo(() => state?.decision_history ?? [], [state?.decision_history]);
  const selectedPath = state?.selected_path ?? [];
  const classNames = useMemo(
    () => (state?.run?.class_names as string[] | undefined) ?? [],
    [state?.run],
  );
  const runTask = (state?.task ?? "classification") as "classification" | "multilabel";
  const isMultilabel = runTask === "multilabel";

  /* KPI summary — best from ALL observations, current from last entry */
  const kpi = useMemo(() => {
    if (!state) return null;
    const tl = state.metrics_timeline;
    if (tl.length === 0) return null;
    const last = tl[tl.length - 1];
    const branchCount = Object.keys(state.branches).length;
    const decisionCount = decisions.length;

    const ml = runTask === "multilabel";

    // Helper to extract primary metric per task
    const primaryMetric = (r: typeof last) =>
      ml ? (r.f1_samples ?? r.f1_macro ?? 0) : (r.accuracy ?? 0);
    const secondaryMetric = (r: typeof last) => r.f1_macro ?? 0;

    // Best = max across ALL metrics (any branch, any epoch)
    const bestAccuracy = Math.max(...tl.map(primaryMetric));
    const bestF1 = Math.max(...tl.map(secondaryMetric));

    // Current = last point on the selected path (trunk)
    const selectedPathBranches = new Set<string>(["baseline"]);
    for (const p of state.selected_path ?? []) {
      const parts = p.split(":");
      if (parts.length > 1) selectedPathBranches.add(parts.slice(1).join(":"));
      else selectedPathBranches.add(p);
    }
    const trunkEntries = tl.filter((r) => selectedPathBranches.has(r.branch));
    const currentAccuracy = trunkEntries.length > 0
      ? primaryMetric(trunkEntries[trunkEntries.length - 1])
      : primaryMetric(last);

    // Baseline = final baseline primary metric (last baseline entry)
    const baselineEntries = tl.filter((r) => r.branch === "baseline");
    const baselineAcc = baselineEntries.length > 0
      ? primaryMetric(baselineEntries[baselineEntries.length - 1])
      : 0;

    // Only count non-baseline entries for "best beyond baseline"
    const hasNonBaseline = tl.some((r) => r.branch !== "baseline");

    // Gain vs baseline — only show after baseline completes and candidates exist
    const gain = hasNonBaseline ? bestAccuracy - baselineAcc : 0;

    return {
      bestAccuracy, bestF1, currentAccuracy, baselineAcc, gain,
      last, branchCount, decisionCount, totalEpochs: tl.length,
    };
  }, [state, decisions, runTask]);

  /* ---- Training progress ---- */
  const progress = useMemo(() => {
    if (!state) return null;
    const cfg = (state.run as Record<string, unknown>)?.config as Record<string, unknown> | undefined;
    const maxIter = Number(cfg?.max_iterations ?? 0);
    const mEpochs = Number(cfg?.m_epochs ?? 1);
    if (maxIter === 0) return null;

    // Current iteration = max iteration from metrics timeline
    const tl = state.metrics_timeline;
    const currentIter = tl.length > 0 ? Math.max(...tl.map((p) => p.iteration)) : 0;
    const currentEpoch = tl.length > 0
      ? Math.max(...tl.filter((p) => p.iteration === currentIter).map((p) => p.epoch))
      : 0;

    // Total phases = baseline (iter 0) + maxIter augmentation rounds
    const totalPhases = maxIter + 1;
    // Each phase has mEpochs epochs (baseline may differ but we approximate)
    const completedPhaseEpochs = currentIter * mEpochs + currentEpoch;
    const totalEpochs = totalPhases * mEpochs;
    const pct = Math.min(100, Math.round((completedPhaseEpochs / totalEpochs) * 100));

    const phaseLabel = currentIter === 0
      ? `Baseline (epoch ${currentEpoch}/${mEpochs})`
      : `Iteration ${currentIter}/${maxIter} (epoch ${currentEpoch}/${mEpochs})`;

    const isFinished = decisions.length >= maxIter || pct >= 100;

    return { pct, phaseLabel, isFinished, currentIter, maxIter };
  }, [state, decisions]);

  /* ---- Pause / Resume ---- */
  const [paused, setPaused] = useState(false);
  const pausePollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Poll pause status every 2s
  useEffect(() => {
    if (!activeRun || offline) return;
    const poll = () => {
      fetch(`/api/run/${activeRun}/status`)
        .then((r) => r.json())
        .then((d: { paused: boolean }) => setPaused(d.paused))
        .catch(() => {});
    };
    poll();
    pausePollRef.current = setInterval(poll, 2000);
    return () => {
      if (pausePollRef.current) clearInterval(pausePollRef.current);
    };
  }, [activeRun, offline]);

  const togglePause = useCallback(async () => {
    if (!activeRun || offline) return;
    const action = paused ? "resume" : "pause";
    try {
      const res = await fetch(`/api/run/${activeRun}/control`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action }),
      });
      const data = (await res.json()) as { ok: boolean; paused: boolean };
      if (data.ok) setPaused(data.paused);
    } catch { /* ignore */ }
  }, [activeRun, offline, paused]);

  /* ---- Save snapshot ---- */
  const [saveMsg, setSaveMsg] = useState("");
  const saveSnapshot = async () => {
    if (!activeRun || offline) return;
    setSaveMsg("Saving…");
    try {
      const res = await fetch(`/api/run/${activeRun}/export`, { method: "POST" });
      const data = (await res.json()) as { ok: boolean; index: string };
      setSaveMsg(data.ok ? `Saved: ${data.index}` : "Failed");
    } catch {
      setSaveMsg("Error");
    }
  };

  return (
    <div className="page">
      {/* -------- Header -------- */}
      <header className="app-header">
        <div className="header-left">
          <img
            src={theme === "dark" ? `${import.meta.env.BASE_URL}logo_dark.png` : `${import.meta.env.BASE_URL}logo_light.png`}
            alt="BNNR"
            className="logo-img"
          />
          <span className={`live-badge ${offline ? "offline" : dashboardMode === "serve" ? "replay" : "live"}`}>
            {offline ? "OFFLINE" : dashboardMode === "serve" ? "REPLAY" : "LIVE"}
          </span>
        </div>
        <div className="header-right">
          <select
            value={activeRun}
            disabled={offline}
            onChange={(e) => setActiveRun(e.target.value)}
          >
            {runs.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id}
              </option>
            ))}
          </select>
          <button onClick={saveSnapshot} disabled={offline} className="btn-save">
            Export Report
          </button>
          {saveMsg && <span className="save-msg">{saveMsg}</span>}
          <button
            onClick={toggleTheme}
            className="theme-toggle"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
          >
            {theme === "dark" ? (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/>
                <line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/>
                <line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
              </svg>
            ) : (
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
              </svg>
            )}
          </button>
        </div>
      </header>

      {/* -------- Tabs -------- */}
      <nav className="tab-bar">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`tab-btn ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
          >
            {t.icon}
            {t.label}
          </button>
        ))}
      </nav>

      {/* -------- Content -------- */}
      {loading && <div className="card skeleton">Loading run state…</div>}
      {!loading && !state && (
        <div className="card empty-state">
          <h2>No data</h2>
          <p>Start training with event logging enabled, or open an existing run.</p>
        </div>
      )}

      {state && (
        <>
          {/* ====== OVERVIEW ====== */}
          {tab === "overview" && (
            <div className="tab-content">
              {/* ── Replay banner — shown when dashboard is in serve/replay mode ── */}
              {dashboardMode === "serve" && (
                <div className="card" style={{
                  padding: "12px 16px",
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  borderLeft: "3px solid var(--accent)",
                  background: "color-mix(in srgb, var(--accent) 6%, var(--card-bg))",
                }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                    <polyline points="1 4 1 10 7 10"/>
                    <path d="M3.51 15a9 9 0 1 0 .49-4.5"/>
                  </svg>
                  <div style={{ minWidth: 0 }}>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "var(--accent)" }}>
                      Replay
                    </span>
                    <span style={{ fontSize: 13, fontWeight: 600, marginLeft: 6 }}>
                      — Dashboard replay for run:
                    </span>
                    {" "}
                    <code style={{
                      fontSize: 12,
                      background: "var(--bg)",
                      padding: "2px 6px",
                      borderRadius: 4,
                      fontFamily: "monospace",
                      wordBreak: "break-all",
                    }}>
                      {activeRun}
                    </code>
                  </div>
                  <div style={{ marginLeft: "auto", textAlign: "right", flexShrink: 0 }}>
                    {(progress?.isFinished || state?.pipeline_complete) ? (
                      <div style={{ fontSize: 12, color: "var(--green)", fontWeight: 600 }}>
                        ✓ Training complete
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: "var(--muted)", fontWeight: 600 }}>
                        ⚠ Training interrupted
                      </div>
                    )}
                    {progress && (
                      <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                        {progress.phaseLabel} · {progress.pct}%
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* ── Pre-training phase banner — visible during live precomputing ── */}
              {dashboardMode === "live" && state?.pipeline_phase && (
                <div className="card" style={{
                  padding: "12px 16px",
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  borderLeft: "3px solid var(--accent)",
                  animation: "pulse 2s infinite",
                }}>
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0, animation: "spin 2s linear infinite" }}>
                    <path d="M21 12a9 9 0 1 1-6.219-8.56"/>
                  </svg>
                  <div>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>
                      {(state.pipeline_phase as { message?: string })?.message || "Preparing..."}
                    </span>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                      Phase: {(state.pipeline_phase as { phase?: string })?.phase?.replace(/_/g, " ") ?? "unknown"}
                    </div>
                  </div>
                </div>
              )}

              {/* ── Training progress bar — only during active (in-progress) live training ── */}
              {dashboardMode === "live" && progress && !progress.isFinished && (
                <div className="card" style={{ padding: "12px 16px" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
                    <span style={{ fontSize: 13, fontWeight: 600 }}>
                      {paused ? "Training Paused" : "Training Progress"}
                    </span>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontSize: 12, color: "var(--muted)" }}>
                        {progress.phaseLabel}
                      </span>
                      {!offline && (
                        <button
                          className={`btn-pause ${paused ? "btn-resume" : ""}`}
                          onClick={togglePause}
                          title={paused ? "Resume training" : "Pause training"}
                        >
                          {paused ? (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                              <polygon points="5 3 19 12 5 21 5 3"/>
                            </svg>
                          ) : (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none">
                              <rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>
                            </svg>
                          )}
                          {paused ? " Resume" : " Pause"}
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="progress-bar-track">
                    <div
                      className={`progress-bar-fill ${paused ? "paused" : ""}`}
                      style={{ width: `${progress.pct}%` }}
                    />
                  </div>
                  <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 4, textAlign: "right" }}>
                    {paused && <span style={{ color: "var(--accent)", fontWeight: 600, marginRight: 8 }}>PAUSED</span>}
                    {progress.pct}%
                  </div>
                </div>
              )}

              {/* ── Live mode: Training Complete banner ── */}
              {dashboardMode === "live" && progress && progress.isFinished && (
                <div className="card" style={{
                  padding: "12px 16px",
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  borderLeft: "3px solid var(--green)",
                  background: "color-mix(in srgb, var(--green) 6%, var(--card-bg))",
                }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="var(--green)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ flexShrink: 0 }}>
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                    <polyline points="22 4 12 14.01 9 11.01"/>
                  </svg>
                  <div>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "var(--green)" }}>
                      Training Complete
                    </span>
                    <div style={{ fontSize: 11, color: "var(--muted)", marginTop: 2 }}>
                      {progress.maxIter} iteration{progress.maxIter !== 1 ? "s" : ""} · {progress.pct}%
                    </div>
                  </div>
                </div>
              )}

              {/* KPI cards */}
              {kpi && (
                <div className="kpi-row">
                  <div className="kpi-card best">
                    <div className="kpi-value">
                      {(kpi.bestAccuracy * 100).toFixed(1) + "%"}
                    </div>
                    <div className="kpi-label">
                      {isMultilabel ? "Best F1 (samples) ★" : "Best Accuracy ★"}
                    </div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-value">
                      {(kpi.currentAccuracy * 100).toFixed(1) + "%"}
                    </div>
                    <div className="kpi-label">
                      {isMultilabel ? "Current F1 (samples)" : "Current Accuracy"}
                    </div>
                  </div>
                  <div className="kpi-card best">
                    <div className="kpi-value">
                      {(kpi.bestF1 * 100).toFixed(1) + "%"}
                    </div>
                    <div className="kpi-label">Best F1 (macro) ★</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-value"
                      style={{ color: kpi.gain > 0 ? "var(--green)" : kpi.gain < 0 ? "var(--red)" : undefined }}
                    >
                      {kpi.gain > 0 ? "+" : ""}{(kpi.gain * 100).toFixed(1)}pp
                    </div>
                    <div className="kpi-label">BNNR Gain vs Baseline</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-value">{kpi.decisionCount}</div>
                    <div className="kpi-label">Decisions Made</div>
                  </div>
                  <div className="kpi-card">
                    <div className="kpi-value">{kpi.branchCount}</div>
                    <div className="kpi-label">Branches Evaluated</div>
                  </div>
                </div>
              )}

              {/* Best path */}
              <div className="card">
                <h3>Selected Path</h3>
                <div className="path-chips">
                  {selectedPath.map((nodeId, i) => {
                    const label =
                      nodeId === "root:baseline"
                        ? "baseline"
                        : nodeId.includes(":")
                          ? nodeId.split(":").slice(1).join(":")
                          : nodeId;
                    return (
                      <span key={nodeId} className="path-chip">
                        {label}
                        {i < selectedPath.length - 1 && <span className="path-arrow">&rarr;</span>}
                      </span>
                    );
                  })}
                </div>
              </div>

              {/* Decision ledger */}
              <div className="card">
                <h3>Branch Decision Ledger</h3>
                <DecisionLedger decisions={decisions} />
              </div>

              {/* Metrics chart — trunk-only + branch comparison */}
              <div className="card">
                <MetricsCharts timeline={state.metrics_timeline} selectedPath={state.selected_path} task={runTask} />
              </div>
            </div>
          )}

          {/* ====== BRANCH TREE ====== */}
          {tab === "tree" && (
            <div className="tab-content">
              <div className="card">
                <h3>Branch Decision Tree</h3>
                <p className="muted" style={{ fontSize: 12, marginBottom: 8 }}>
                  Green = selected path. Click a node for details. Hover for quick metrics.
                </p>
                <BranchGraph state={state} activeRun={activeRun} offline={offline} />
              </div>
            </div>
          )}

          {/* ====== METRICS ====== */}
          {tab === "metrics" && (
            <div className="tab-content">
              <div className="card">
                <h3>Training Metrics</h3>
                <MetricsCharts timeline={state.metrics_timeline} selectedPath={state.selected_path} task={runTask} />
              </div>
              <div className="grid-2">
                <div className="card">
                  <h3>{isMultilabel ? "Per-Label F1" : "Per-Class Accuracy"}</h3>
                  <PerClassMetrics perClassTimeline={state.per_class_timeline} task={runTask} classNames={state.dataset_profile?.class_names} />
                </div>
                <div className="card">
                  <h3>Confusion Matrix</h3>
                  <ConfusionMatrix
                    confusionTimeline={state.confusion_timeline}
                    classNames={classNames}
                  />
                </div>
              </div>
            </div>
          )}

          {/* ====== SAMPLES & XAI ====== */}
          {tab === "samples" && (
            <div className="tab-content">
              <div className="card">
                <h3>XAI Preview </h3>
                <AugmentationPreview
                  state={state}
                  activeRun={activeRun}
                  offline={offline}
                />
              </div>
              <div className="card">
                <h3>XAI Insights — Per-Class Explanations</h3>
                <p className="muted" style={{ fontSize: 11, marginBottom: 10 }}>
                  Human-readable analysis of what the model focuses on for each class, generated from saliency maps.
                </p>
                <XAIInsights
                  perClassTimeline={state.per_class_timeline}
                  xaiInsightsTimeline={state.xai_insights_timeline}
                  classNames={classNames}
                  task={runTask}
                />
              </div>
              <div className="card">
                <h3>XAI Quality Trend</h3>
                <p className="muted" style={{ fontSize: 11, marginBottom: 10 }}>
                  How the model's attention quality evolves over training — overall average and per-class breakdown.
                </p>
                <XAIQualityTrend
                  xaiInsightsTimeline={state.xai_insights_timeline}
                  classNames={classNames}
                />
              </div>
            </div>
          )}

          {/* ====== ANALYSIS ====== */}
          {tab === "analysis" && (
            <div className="tab-content">
              {/* Augmentation Impact Summary */}
              <div className="card">
                <h3>Augmentation Impact Summary</h3>
                <AugmentationImpact
                  decisions={decisions}
                  timeline={state.metrics_timeline}
                  branchNodes={state.branch_graph.nodes}
                />
              </div>

              {/* Candidate Comparison + Loss Landscape */}
              <div className="grid-2">
                <div className="card">
                  <h3>Candidate Comparison (Radar)</h3>
                  <CandidateRadar decisions={decisions} task={runTask} />
                </div>
                <div className="card">
                  <h3>Loss & Accuracy Landscape</h3>
                  <LossLandscape
                    timeline={state.metrics_timeline}
                    decisions={decisions}
                    task={runTask}
                  />
                </div>
              </div>

              {/* Per-Class Delta Heatmap */}
              <div className="card">
                <h3>Per-Class Δ Heatmap</h3>
                <p className="muted" style={{ fontSize: 11, marginBottom: 8 }}>
                  Accuracy change per class after each augmentation decision. Green = improvement, red = regression.
                </p>
                <PerClassDeltaHeatmap
                  decisions={decisions}
                  perClassTimeline={state.per_class_timeline}
                />
              </div>

              {/* Augmentation Stack + Time Breakdown */}
              <div className="grid-2">
                <div className="card">
                  <h3>Augmentation Pipeline</h3>
                  <AugmentationStack
                    decisions={decisions}
                    selectedPath={state.selected_path}
                  />
                </div>
                <div className="card">
                  <h3>Accuracy Gain Rate</h3>
                  <AccuracyGainRate
                    timeline={state.metrics_timeline}
                    selectedPath={state.selected_path}
                    task={runTask}
                  />
                </div>
              </div>

              {/* Full Decision Ledger */}
              <div className="card">
                <h3>Decision Ledger (Full)</h3>
                <DecisionLedger decisions={decisions} />
              </div>

              {/* Confusion Matrix */}
              <div className="card">
                <h3>Confusion Matrix</h3>
                <ConfusionMatrix
                  confusionTimeline={state.confusion_timeline}
                  classNames={classNames}
                />
              </div>
            </div>
          )}

          {/* ====== DATASET INSIGHT ====== */}
          {tab === "insight" && (
            <DatasetInsight
              datasetProfile={state.dataset_profile}
              perClassTimeline={state.per_class_timeline}
              confusionTimeline={state.confusion_timeline}
              classNames={classNames}
              activeRun={activeRun}
              offline={offline}
            />
          )}
        </>
      )}
    </div>
  );
}
