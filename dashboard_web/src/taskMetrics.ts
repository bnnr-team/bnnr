import type { MetricPoint } from "./types";

export type RunTask = "classification" | "multilabel" | "detection";

/** Normalize backend task string for dashboard UI. */
export function normalizeRunTask(raw: string | undefined): RunTask {
  if (raw === "multilabel" || raw === "detection") return raw;
  return "classification";
}

/** Primary selection metric for charts / KPIs (0–1 scale for accuracy, F1, mAP). */
export function primaryMetricValue(p: MetricPoint, task: RunTask): number {
  if (task === "multilabel") {
    return p.f1_samples ?? p.f1_macro ?? 0;
  }
  if (task === "detection") {
    return p.map_50 ?? p.map_50_95 ?? 0;
  }
  return p.accuracy ?? 0;
}

/** Secondary line (macro F1) — meaningful for classification; for detection use mAP@50:95. */
export function secondaryMetricValue(p: MetricPoint, task: RunTask): number {
  if (task === "detection") {
    return p.map_50_95 ?? p.map_50 ?? 0;
  }
  return p.f1_macro ?? 0;
}
